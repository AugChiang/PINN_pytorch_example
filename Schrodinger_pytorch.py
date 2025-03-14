import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pyDOE import lhs

class SchrodingerDataset():
    def __init__(
            self,
            path:str="data/NLS.mat"
        ):
        self.path = path
        self.X, self.T, self.GT = self.get_data()

    def get_data(self):
        data = scipy.io.loadmat(self.path)
        T = data['tt'].flatten()[:,None] # (201,1)
        X = data['x'].flatten()[:,None]  # (256,1)
        GT = data['uu'] # ground truth, complex number, (256,201) along (x,t)
        return X, T, GT

class InitDataset(Dataset):
    """The initial data, {(x,h), ...}, where h = h(t,x) = [u(t,x), v(t,x)]"""
    def __init__(
            self,
            SchrodingerDataset:SchrodingerDataset,
            N0:int = 50,
        ):
        super().__init__()
        self.X = SchrodingerDataset.X
        self.T = SchrodingerDataset.T
        self.GT = SchrodingerDataset.GT
        self.N0 = N0 # number of samples

        self.X0, self.H0 = self.get_init_data()

    def get_init_data(self):
        U = torch.from_numpy(self.GT.real.astype(np.float32, copy=False))
        V = torch.from_numpy(self.GT.imag.astype(np.float32, copy=False))

        random_indices = np.random.choice(self.X.shape[0], self.N0, replace=False)

        X0 = self.X[random_indices,:].astype(np.float32, copy=False)
        X0 = torch.from_numpy(X0)
        X0 = torch.concat((X0, torch.zeros_like(X0)), dim=1)

        H0 = torch.concat((U[random_indices, 0:1], V[random_indices, 0:1]), dim=1) # t = 0

        return X0, H0

    def __len__(self):
        return len(self.X0)

    def __getitem__(self, idx):
        return self.X0[idx], self.H0[idx]

class BoundaryDataset(Dataset):
    """Collocation points on the boundary."""
    def __init__(
            self,
            SchrodingerDataset:SchrodingerDataset,
            lb:np.ndarray = np.array([-5.0, 0.0]),
            ub:np.ndarray = np.array([5.0, np.pi/2]),
            N_b:int = 50,
        ):
        super().__init__()
        self.T = SchrodingerDataset.T
        self.N_b = N_b # number of samples
        self.lb = lb
        self.ub = ub
        self.x_lb, self.x_ub = self.get_boundary_data()

    def get_boundary_data(self):
        random_indices = np.random.choice(self.T.shape[0], self.N_b, replace=False)
        tb = self.T[random_indices,:].astype(np.float32, copy=False)
        tb = torch.from_numpy(tb) #(B,1)
        x_lb = torch.concat((torch.zeros_like(tb) + self.lb[0], tb), dim=1) # (lb[0], tb)
        x_ub = torch.concat((torch.zeros_like(tb) + self.ub[0], tb), dim=1) # (ub[0], tb)
        return x_lb, x_ub

    def __len__(self):
        return len(self.x_lb)

    def __getitem__(self, idx):
        return self.x_lb[idx], self.x_ub[idx]

class CollocationDataset(Dataset):
    """Collocation points on f(t,x)"""
    def __init__(
            self,
            lb = np.array([-5.0, 0.0]),
            ub= np.array([5.0, np.pi/2]),
            N_f:int = 20000,
        ):
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.N_f = N_f # number of samples

        self.X_f = self.get_data() # (N_f, 2)

    def get_data(self):
        X_f = self.lb + (self.ub-self.lb)*lhs(2, self.N_f)
        X_f = X_f.astype(np.float32, copy=False)
        X_f = torch.from_numpy(X_f)
        return X_f

    def __len__(self):
        return len(self.X_f)

    def __getitem__(self, idx):
        return self.X_f[idx]

class PINN(nn.Module):
    """ Model """
    def __init__(self, in_ch, chs, out_ch):
        super().__init__()
        self.chs = chs
        self.fc_in = nn.Linear(in_ch, chs[0])
        self.ffns = nn.ModuleList([
            nn.Linear(chs[i], chs[i+1]) for i in range(0,len(chs)-1)
        ])
        self.fc_out = nn.Linear(chs[-1], out_ch)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)
        return

    def forward(self, x):
        x = self.fc_in(x)
        for l in self.ffns:
            x = l(x)
            x = F.tanh(x)
        x = self.fc_out(x)
        return x

def get_grad(y, x, n=1):
    if not x.requires_grad:
        x = x.requires_grad_()
    grad = autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=torch.ones_like(y),
                create_graph=True
            )[0]
    if n > 1:
        for _ in range(n-1):
            grad = autograd.grad(
                outputs=grad,
                inputs=x,
                grad_outputs=torch.ones_like(grad),
                create_graph=True
            )[0]
    return grad

if __name__ == "__main__": 
     
    noise = 0.0     

    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    # number of samples
    N0 = 50
    N_b = 50
    N_f = 20000

    # batch sizes
    batch_0 = 16
    batch_b = 16
    batch_f = 256

    split = [0.8, 0.2] # [train, val]

    # model
    in_ch = 2
    chs = [100, 100, 100, 100]
    out_ch = 2
    model = PINN(in_ch, chs, out_ch)

    epochs = 2
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # datasets
    schrodinger_dataset = SchrodingerDataset()
    init_dataset = InitDataset(schrodinger_dataset, N0=N0)
    boundary_dataset = BoundaryDataset(schrodinger_dataset, lb=lb, ub=ub, N_b=N_b)
    collocation_dataset = CollocationDataset(lb=lb, ub=ub, N_f=N_f)

    # dataloader
    init_dataloader = DataLoader(init_dataset, batch_size=batch_0, drop_last=True)
    boundary_dataloader = DataLoader(boundary_dataset, batch_size=batch_b, drop_last=True)
    tr_dataset, val_dataset = random_split(collocation_dataset, split)
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_f, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_f, drop_last=True)

    num_iter = len(tr_dataloader)

    for epoch in range(epochs):
        model.train()

        n_steps = 0
        loss_0_epoch = 0.
        loss_b_epoch = 0.
        loss_f_epoch = 0.
    
        # init iterator
        init_iterator = iter(init_dataloader)
        boundary_iterator = iter(boundary_dataloader)
        tr_iterator = iter(tr_dataloader)

        tr_loss_histroy = {
            '0':[],
            'lb':[],
            'ub':[],
            'fu':[],
            'fv':[],
        }
        val_loss_history = {'u':[], 'v':[]}
        # use iterator to fetch the batched data
        # and calculate the loss_u and loss_f at the same time for backpropogation
        for step in range(num_iter):
            # init data
            try:
                x0, h0 = next(init_iterator)
            except StopIteration:
                # re-init boundary iterator when exhausted
                init_iterator = iter(init_dataloader)
                x0, h0 = next(init_iterator)

            # boundary points
            try:
                x_lb, x_ub = next(boundary_iterator)
            except StopIteration:
                # re-init boundary iterator when exhausted
                boundary_iterator = iter(boundary_dataloader)
                x_lb, x_ub = next(boundary_iterator)
 
            # collocation points
            try:
                x_f = next(tr_iterator) # (B,2)
            except StopIteration:
                break

            optimizer.zero_grad()
            # MSE_0
            uv0_pred = model(x0)
            loss_0 = loss_fn(uv0_pred, h0)

            # MSE_b
            x_lb = x_lb.requires_grad_()
            x_ub = x_ub.requires_grad_()
            uv_lb_pred = model(x_lb)
            uv_ub_pred = model(x_ub)
            uv_lb_pred_x = get_grad(uv_lb_pred, x_lb)
            uv_ub_pred_x = get_grad(uv_ub_pred, x_ub)
            loss_uv_b = loss_fn(uv_lb_pred, uv_ub_pred)
            loss_uv_b_x = loss_fn(uv_lb_pred_x[:,0], uv_ub_pred_x[:,0])
            loss_b = loss_uv_b + loss_uv_b_x

            # MSE_f
            x_f = x_f.requires_grad_()
            uv_pred = model(x_f)
            uv_pred_amp = torch.sum(torch.square(uv_pred), dim=1) # |h|^2
            u_xt = get_grad(uv_pred[:,0], x_f, n=1) # grad sum of every elements, [x's sum, t's sum]
            v_xt = get_grad(uv_pred[:,1], x_f, n=1) # grad sum of every elements, [x's sum, t's sum]
            u_xxtt = get_grad(u_xt, x_f, n=1)
            v_xxtt = get_grad(v_xt, x_f, n=1)
            f_u = u_xt[:,1] + 0.5*v_xxtt[:,0] + uv_pred_amp*uv_pred[:,1]
            f_v = v_xt[:,1] - 0.5*u_xxtt[:,0] + uv_pred_amp*uv_pred[:,0]
            
            loss_f_u = loss_fn(f_u, torch.zeros_like(f_u))
            loss_f_v = loss_fn(f_v, torch.zeros_like(f_v))
            loss_f = loss_f_u + loss_f_v
            
            loss = loss_0 + loss_b + loss_f

            n_steps += 1
            loss_0_epoch += loss_0.item()
            loss_b_epoch += loss_b.item()
            loss_f_epoch += loss_f.item()
            tr_loss_histroy['0'].append(loss_0.item())
            tr_loss_histroy['lb'].append(loss_uv_b.item())
            tr_loss_histroy['ub'].append(loss_uv_b_x.item())
            tr_loss_histroy['fu'].append(loss_f_u.item())
            tr_loss_histroy['fv'].append(loss_f_v.item())

            if step % 10 == 0:
                print(f"step: {step} | loss_0: {loss_0.item():.4f} | loss_b: {loss_b.item():.4f} | loss_f: {loss_f.item():.4f} | total_loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()

        loss_0_epoch /= n_steps
        loss_b_epoch /= n_steps
        loss_f_epoch /= n_steps

        print(f"Epoch: {epoch+1} | total_loss: {loss_0_epoch+loss_f_epoch:.4f} | loss_0: {loss_0_epoch:.4f} | loss_b: {loss_b_epoch:.4f} | loss_f: {loss_f_epoch:.4f}")
