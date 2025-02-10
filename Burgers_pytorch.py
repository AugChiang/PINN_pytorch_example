import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from pyDOE import lhs

class BurgerDataset:
    def __init__(
            self,
            path:str="appendix/Data/burgers_shock.mat"
        ):
        self.path = path
        self.X, self.T, self.GT = self.get_data()

    def get_data(self):
        data = scipy.io.loadmat(self.path)
        T = data['t'].flatten()[:,None]
        X = data['x'].flatten()[:,None]
        GT = np.real(data['usol']).T # ground truth
        return X, T, GT
    
class BurgerBoundaryDataset(Dataset):
    def __init__(
            self,
            data:BurgerDataset,
            data_trans = None,
            N_u:int=100
        ):
        self.X = data.X
        self.T = data.T
        self.GT = data.GT
        self.N_u = N_u
        self.X_u, self.U = self.get_boundary_data()

    def get_boundary_data(self):
        X, T = np.meshgrid(self.X, self.T)

        # boundary points
        xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # t = 0 (left)
        uu1 = self.GT[0:1,:].T
        xx2 = np.hstack((X[:,0:1], T[:,0:1])) # x = -1 (bottom)
        uu2 = self.GT[:,0:1]
        xx3 = np.hstack((X[:,-1:], T[:,-1:])) # x = 1 (upper)
        uu3 = self.GT[:,-1:]
        X_u = np.vstack([xx1, xx2, xx3])
        U = np.vstack([uu1, uu2, uu3])

        # sample num of N_u points at boundary
        sample_indice = np.random.choice(X_u.shape[0], self.N_u, replace=False)
        X_u = X_u[sample_indice,:]
        U = U[sample_indice,:]

        X_u = X_u.astype(np.float32, copy=False)
        U = U.astype(np.float32, copy=False)
        return X_u, U

    def __len__(self):
        return len(self.X_u)

    def __getitem__(self, idx):
        return self.X_u[idx], self.U[idx]

class BurgerCollocationDataset(Dataset):
    def __init__(
            self,
            data:BurgerDataset,
            boundary_data:BurgerBoundaryDataset,
            data_trans = None,
            N_f:int=10000
        ):
        self.X = data.X
        self.T = data.T
        self.GT = data.GT
        self.N_f = N_f

        # boundary pts
        self.X_u = boundary_data.X_u
        self.U = boundary_data.U

        self.X_f, self.U_f = self.get_collocation_data()

    def get_collocation_data(self):
        X, T = np.meshgrid(self.X, self.T)

        # (25600, 2)
        X_flat = np.hstack((X.flatten()[:,None],
                            T.flatten()[:,None]))
        GT_flat = self.GT.flatten()[:,None]

        # Domain bounds [x's, t's]
        lb = X_flat.min(0)
        ub = X_flat.max(0)

        # sample collocation points
        
        X_f = lb + (ub-lb)*lhs(2, self.N_f) # sample [x,t]s

        #! ground truth may NOT exist, thus, some index would be [] (empty).
        # indices = [np.where((X_flat == X_f[i]).all(axis=1))[0][0] \
        #            for i in range(X_f.shape[0])]

        # X_f = X_flat[indices,:] # [[x1,t1], ...]
        # U_f = GT_flat[indices, :]
        
        # The original paper also stacks the boundary points.
        X_f = np.vstack((X_f, self.X_u))
        # U_f = np.vstack((U_f, self.U))
        # return X_f, U_f
        X_f = X_f.astype(np.float32, copy=False)
        return X_f, None

    def __len__(self):
        return len(self.X_f)

    def __getitem__(self, idx):
        # return self.X_f[idx], self.U_f[idx]
        return self.X_f[idx]


class FFN(nn.Module):
    def __init__(self, in_ch, hidden_dim, out_ch):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_dim = hidden_dim
        self.out_ch = out_ch

        self.fc1 = nn.Linear(in_ch, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_ch)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class PINN(nn.Module):
    def __init__(self, in_ch, chs, out_ch):
        super().__init__()
        self.chs = chs
        self.fc_in = nn.Linear(in_ch, chs[0])
        # self.ffn_blocks = nn.ModuleList([
        #     FFN(chs[i], chs[i]*2, chs[i+1]) for i in range(0,len(chs)-1)
        # ])
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
    N_u = 128
    N_f = 16384
    nu = 0.01/np.pi

    epochs = 100
    batch_u = 64
    batch_f = 256
    split = [0.8, 0.1, 0.1] # [train, test]

    in_ch = 2
    chs = [20, 20, 20, 20, 20, 20, 20, 20]
    out_ch = 1

    
    data_trans = transforms.Compose([
        transforms.Lambda(lambda x: x.float()) # float64 to float32
    ])
    burger_dataset = BurgerDataset()
    boundary_dataset = BurgerBoundaryDataset(burger_dataset,
                                             N_u=N_u,
                                             data_trans=data_trans)
    boundary_data_loader = DataLoader(boundary_dataset,
                                      batch_size=batch_u,
                                      drop_last=True)
    collocation_dataset = BurgerCollocationDataset(burger_dataset,
                                                   boundary_dataset,
                                                   N_f = N_f,
                                                   data_trans=data_trans)
    tr_dataset, val_dataset, test_dataset = random_split(collocation_dataset, split)
    tr_dataloader = DataLoader(tr_dataset,
                               batch_size=batch_f,
                               drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_f,
                                drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=16,
                                 drop_last=True)

    model = PINN(in_ch=in_ch, chs=chs, out_ch=out_ch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tr_loss_histroy = {'u':[], 'f':[]}
    val_loss_histroy = []

    num_iter = len(tr_dataloader)

    
    for epoch in range(epochs):
        model.train()

        n_steps = 0
        loss_u_epoch = 0.
        loss_f_epoch = 0.
    
        # init iterator
        boundary_iterator = iter(boundary_data_loader)
        tr_iterator = iter(tr_dataloader)

        # use iterator to fetch the batched data
        # and calculate the loss_u and loss_f at the same time for backpropogation
        for step in range(num_iter):
            try:
                x_u, u = next(boundary_iterator)
            except StopIteration:
                # re-init boundary iterator when exhausted
                boundary_iterator = iter(boundary_data_loader)
                x_u, u = next(boundary_iterator)

            try:
                x_f = next(tr_iterator)
            except StopIteration:
                break

            optimizer.zero_grad()
            x_f = x_f.requires_grad_()
            u_pred = model(x_u)
            uf_pred = model(x_f)
            u_xt = get_grad(uf_pred, x_f, n=1)
            u_xxtt = get_grad(u_xt, x_f, n=1)
            f = u_xt[:,1] + u*u_xt[:,0] - nu*u_xxtt[:,0]
            
            loss_u = loss_fn(u_pred, u)
            loss_f = loss_fn(f, torch.zeros_like(f))
            loss = loss_u + loss_f

            n_steps += 1
            loss_u_epoch += loss_u.item()
            loss_f_epoch += loss_f.item()
            
            tr_loss_histroy['u'].append(loss_u.item())
            tr_loss_histroy['f'].append(loss_f.item())
            if step % 10 == 0:
                print(f"step: {step} | loss_u: {loss_u.item():.4f} | loss_f: {loss_f.item():.4f} | total_loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()

        loss_u_epoch /= n_steps
        loss_f_epoch /= n_steps

        print(f"Epoch: {epoch+1} | total_loss: {loss_u_epoch+loss_f_epoch:.4f} | loss_u: {loss_u_epoch:.4f} | loss_f: {loss_f_epoch:.4f}")

        model.eval()        
        n_steps = 0.
        val_loss = 0.

        for step, x in enumerate(val_dataloader):
            x = x.requires_grad_()
            uf_pred = model(x)
            u_xt = get_grad(uf_pred, x)
            u_xxtt = get_grad(u_xt, x)

            f = u_xt[:,1] + u*u_xt[:,0] - nu*u_xxtt[:,0]
            loss = loss_fn(f, torch.zeros_like(f))
            val_loss_histroy.append(loss.item())
            val_loss += loss
            n_steps += 1

        val_loss /= n_steps
        print("Val f_loss: {:.4f}".format(val_loss.item()))