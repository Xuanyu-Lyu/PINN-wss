# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# torch / torch.nn / torch.optim : core PyTorch — tensors, layers, optimizers
# torch.autograd                 : automatic differentiation for velocity/pressure
# numpy                          : array handling for mesh and BC data
# matplotlib.pyplot              : plotting and saving results
# vtk / vtk.util.numpy_support  : reading .vtu/.vtk mesh and data files
# DataLoader / TensorDataset     : mini-batch training infrastructure
# time                           : wall-clock timing of training
# os                             : path management for results directory
# =============================================================================
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import vtk
from vtk.util import numpy_support as VN


def geo_train(device, x_in, y_in, z_in,
              xb, yb, zb, ub, vb, wb,
              xd, yd, zd, ud, vd, wd,
              batchsize, learning_rate, epochs, path,
              Flag_batch, Diff, rho, Flag_BC_exact, Lambda_BC,
              nPt, T,
              xb_inlet, yb_inlet, zb_inlet,
              ub_inlet, vb_inlet, wb_inlet):
    # -------------------------------------------------------------------------
    # Data setup: convert NumPy arrays to PyTorch float tensors on the target
    # device. In batch mode wrap interior points in a DataLoader; BCs and data
    # points are small enough to keep as full tensors throughout training.
    # -------------------------------------------------------------------------
    if Flag_batch:
        x        = torch.FloatTensor(x_in).to(device)
        y        = torch.FloatTensor(y_in).to(device)
        z        = torch.FloatTensor(z_in).to(device)
        xb       = torch.FloatTensor(xb).to(device)
        yb       = torch.FloatTensor(yb).to(device)
        zb       = torch.FloatTensor(zb).to(device)
        ub       = torch.FloatTensor(ub).to(device)
        vb       = torch.FloatTensor(vb).to(device)
        wb       = torch.FloatTensor(wb).to(device)
        xd       = torch.FloatTensor(xd).to(device)
        yd       = torch.FloatTensor(yd).to(device)
        zd       = torch.FloatTensor(zd).to(device)
        ud       = torch.FloatTensor(ud).to(device)
        vd       = torch.FloatTensor(vd).to(device)
        wd       = torch.FloatTensor(wd).to(device)
        xb_inlet = torch.FloatTensor(xb_inlet).to(device)
        yb_inlet = torch.FloatTensor(yb_inlet).to(device)
        zb_inlet = torch.FloatTensor(zb_inlet).to(device)
        ub_inlet = torch.FloatTensor(ub_inlet).to(device)
        vb_inlet = torch.FloatTensor(vb_inlet).to(device)
        wb_inlet = torch.FloatTensor(wb_inlet).to(device)

        dataset    = TensorDataset(x, y, z)
        dataloader = DataLoader(dataset, batch_size=batchsize,
                                shuffle=True, num_workers=0, drop_last=True)
    else:
        x = torch.FloatTensor(x_in).to(device)
        y = torch.FloatTensor(y_in).to(device)
        z = torch.FloatTensor(z_in).to(device)

    h_n     = 200  # neurons per hidden layer
    input_n = 3    # spatial inputs: x, y, z

    # -------------------------------------------------------------------------
    # Swish activation: f(x) = x * sigmoid(x).
    # Smooth and non-monotonic — preserves well-behaved higher-order derivatives
    # needed for the second-order PDE residuals. inplace=True saves memory.
    # -------------------------------------------------------------------------
    class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)

    # -------------------------------------------------------------------------
    # Network for u (x-velocity): 9 hidden layers of width h_n with Swish.
    # -------------------------------------------------------------------------
    class Net2_u(nn.Module):
        def __init__(self):
            super(Net2_u, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_n), Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, 1),
            )

        def forward(self, x):
            output = self.main(x)
            if Flag_BC_exact:
                output = (output * (x - xStart) * (y - yStart) * (y - yEnd)
                          + U_BC_in + (y - yStart) * (y - yEnd))
            return output

    # -------------------------------------------------------------------------
    # Network for v (y-velocity): same architecture as Net2_u.
    # -------------------------------------------------------------------------
    class Net2_v(nn.Module):
        def __init__(self):
            super(Net2_v, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_n), Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, 1),
            )

        def forward(self, x):
            output = self.main(x)
            if Flag_BC_exact:
                output = (output * (x - xStart) * (x - xEnd)
                          * (y - yStart) * (y - yEnd) + (-0.9 * x + 1.))
            return output

    # -------------------------------------------------------------------------
    # Network for w (z-velocity): same architecture as Net2_u.
    # -------------------------------------------------------------------------
    class Net2_w(nn.Module):
        def __init__(self):
            super(Net2_w, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_n), Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, 1),
            )

        def forward(self, x):
            output = self.main(x)
            if Flag_BC_exact:
                output = (output * (x - xStart) * (x - xEnd)
                          * (y - yStart) * (y - yEnd) + (-0.9 * x + 1.))
            return output

    # -------------------------------------------------------------------------
    # Network for p (pressure): same architecture as Net2_u/v/w.
    # -------------------------------------------------------------------------
    class Net2_p(nn.Module):
        def __init__(self):
            super(Net2_p, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n, h_n), Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, h_n),    Swish(),
                nn.Linear(h_n, 1),
            )

        def forward(self, x):
            output = self.main(x)
            if Flag_BC_exact:
                output = (output * (x - xStart) * (x - xEnd)
                          * (y - yStart) * (y - yEnd) + (-0.9 * x + 1.))
            return output

    # -------------------------------------------------------------------------
    # Instantiate networks and move to device
    # -------------------------------------------------------------------------
    net2_u = Net2_u().to(device)
    net2_v = Net2_v().to(device)
    net2_w = Net2_w().to(device)
    net2_p = Net2_p().to(device)

    # Kaiming (He) normal initialisation — suited for Swish / ReLU-like activations
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)

    net2_u.apply(init_normal)
    net2_v.apply(init_normal)
    net2_w.apply(init_normal)
    net2_p.apply(init_normal)

    # Adam optimisers with tight epsilon for numerical stability
    optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate,
                             betas=(0.9, 0.99), eps=1e-15)
    optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate,
                             betas=(0.9, 0.99), eps=1e-15)
    optimizer_w = optim.Adam(net2_w.parameters(), lr=learning_rate,
                             betas=(0.9, 0.99), eps=1e-15)
    optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate,
                             betas=(0.9, 0.99), eps=1e-15)

    # -------------------------------------------------------------------------
    # PDE residual loss (3D Navier-Stokes + continuity).
    # Computes steady incompressible NS residuals at collocation points (x,y,z):
    #   x-momentum : u*u_x + v*u_y + w*u_z - nu*(u_xx+u_yy+u_zz) + (1/rho)*P_x = 0
    #   y-momentum : u*v_x + v*v_y + w*v_z - nu*(v_xx+v_yy+v_zz) + (1/rho)*P_y = 0
    #   z-momentum : u*w_x + v*w_y + w*w_z - nu*(w_xx+w_yy+w_zz) + (1/rho)*P_z = 0
    #   continuity :  u_x + v_y + w_z = 0
    # Scale factors non-dimensionalise so all residuals are O(1).
    # -------------------------------------------------------------------------
    def criterion(x, y, z):
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        net_in = torch.cat((x, y, z), 1)
        u = net2_u(net_in).view(len(x), -1)
        v = net2_v(net_in).view(len(x), -1)
        w = net2_w(net_in).view(len(x), -1)
        P = net2_p(net_in).view(len(x), -1)

        # --- First and second spatial derivatives via autograd ---
        u_x  = torch.autograd.grad(u,   x, grad_outputs=torch.ones_like(x),
                                   create_graph=True, only_inputs=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x),
                                   create_graph=True, only_inputs=True)[0]
        u_y  = torch.autograd.grad(u,   y, grad_outputs=torch.ones_like(y),
                                   create_graph=True, only_inputs=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y),
                                   create_graph=True, only_inputs=True)[0]
        u_z  = torch.autograd.grad(u,   z, grad_outputs=torch.ones_like(z),
                                   create_graph=True, only_inputs=True)[0]
        u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(z),
                                   create_graph=True, only_inputs=True)[0]

        v_x  = torch.autograd.grad(v,   x, grad_outputs=torch.ones_like(x),
                                   create_graph=True, only_inputs=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x),
                                   create_graph=True, only_inputs=True)[0]
        v_y  = torch.autograd.grad(v,   y, grad_outputs=torch.ones_like(y),
                                   create_graph=True, only_inputs=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y),
                                   create_graph=True, only_inputs=True)[0]
        v_z  = torch.autograd.grad(v,   z, grad_outputs=torch.ones_like(z),
                                   create_graph=True, only_inputs=True)[0]
        v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(z),
                                   create_graph=True, only_inputs=True)[0]

        w_x  = torch.autograd.grad(w,   x, grad_outputs=torch.ones_like(x),
                                   create_graph=True, only_inputs=True)[0]
        w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x),
                                   create_graph=True, only_inputs=True)[0]
        w_y  = torch.autograd.grad(w,   y, grad_outputs=torch.ones_like(y),
                                   create_graph=True, only_inputs=True)[0]
        w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y),
                                   create_graph=True, only_inputs=True)[0]
        w_z  = torch.autograd.grad(w,   z, grad_outputs=torch.ones_like(z),
                                   create_graph=True, only_inputs=True)[0]
        w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(z),
                                   create_graph=True, only_inputs=True)[0]

        P_x  = torch.autograd.grad(P,   x, grad_outputs=torch.ones_like(x),
                                   create_graph=True, only_inputs=True)[0]
        P_y  = torch.autograd.grad(P,   y, grad_outputs=torch.ones_like(y),
                                   create_graph=True, only_inputs=True)[0]
        P_z  = torch.autograd.grad(P,   z, grad_outputs=torch.ones_like(z),
                                   create_graph=True, only_inputs=True)[0]

        # Combined scale factors for non-dimensionalised derivatives
        XX_scale = U_scale * (X_scale  ** 2)
        YY_scale = U_scale * (YZ_scale ** 2)
        UU_scale = U_scale ** 2

        # NS residuals (target = 0)
        res_x = (u * u_x / X_scale  + v * u_y / YZ_scale + w * u_z / YZ_scale
                 - Diff * (u_xx / XX_scale + u_yy / YY_scale + u_zz / YY_scale)
                 + (1.0 / rho) * P_x / (X_scale  * UU_scale))   # x-momentum
        res_y = (u * v_x / X_scale  + v * v_y / YZ_scale + w * v_z / YZ_scale
                 - Diff * (v_xx / XX_scale + v_yy / YY_scale + v_zz / YY_scale)
                 + (1.0 / rho) * P_y / (YZ_scale * UU_scale))   # y-momentum
        res_z = (u * w_x / X_scale  + v * w_y / YZ_scale + w * w_z / YZ_scale
                 - Diff * (w_xx / XX_scale + w_yy / YY_scale + w_zz / YY_scale)
                 + (1.0 / rho) * P_z / (YZ_scale * UU_scale))   # z-momentum
        res_c = u_x / X_scale + v_y / YZ_scale + w_z / YZ_scale # continuity

        loss_f = nn.MSELoss()
        loss = (loss_f(res_x, torch.zeros_like(res_x))
                + loss_f(res_y, torch.zeros_like(res_y))
                + loss_f(res_z, torch.zeros_like(res_z))
                + loss_f(res_c, torch.zeros_like(res_c)))
        return loss

    # -------------------------------------------------------------------------
    # No-slip wall boundary condition loss.
    # Penalises u, v, w deviating from zero on wall points (xb, yb, zb).
    # -------------------------------------------------------------------------
    def Loss_BC(xb, yb, zb, ub, vb, wb,
                xb_inlet, yb_inlet, ub_inlet, x, y, z):
        net_in1 = torch.cat((xb, yb, zb), 1)
        out1_u  = net2_u(net_in1).view(len(xb), -1)
        out1_v  = net2_v(net_in1).view(len(xb), -1)
        out1_w  = net2_w(net_in1).view(len(xb), -1)

        loss_f      = nn.MSELoss()
        loss_noslip = (loss_f(out1_u, torch.zeros_like(out1_u))
                       + loss_f(out1_v, torch.zeros_like(out1_v))
                       + loss_f(out1_w, torch.zeros_like(out1_w)))
        return loss_noslip

    # -------------------------------------------------------------------------
    # Sparse velocity data loss.
    # Penalises u, v, w deviating from known values at interior data points.
    # -------------------------------------------------------------------------
    def Loss_data(xd, yd, zd, ud, vd, wd):
        net_in1 = torch.cat((xd, yd, zd), 1)
        out1_u  = net2_u(net_in1).view(len(xd), -1)
        out1_v  = net2_v(net_in1).view(len(xd), -1)
        out1_w  = net2_w(net_in1).view(len(xd), -1)

        loss_f = nn.MSELoss()
        loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd) + loss_f(out1_w, wd)
        return loss_d

    tic = time.time()

    # Optionally load pre-trained weights to continue a previous run
    if Flag_pretrain:
        print('Reading (pretrain) weights...')
        net2_u.load_state_dict(torch.load(os.path.join(path, 'IA3D_data3velsmall_u.pt')))
        net2_v.load_state_dict(torch.load(os.path.join(path, 'IA3D_data3velsmall_v.pt')))
        net2_w.load_state_dict(torch.load(os.path.join(path, 'IA3D_data3velsmall_w.pt')))
        net2_p.load_state_dict(torch.load(os.path.join(path, 'IA3D_data3velsmall_p.pt')))

    # StepLR scheduler: multiply LR by decay_rate every step_epoch epochs
    if Flag_schedule:
        scheduler_u = torch.optim.lr_scheduler.StepLR(
            optimizer_u, step_size=step_epoch, gamma=decay_rate)
        scheduler_v = torch.optim.lr_scheduler.StepLR(
            optimizer_v, step_size=step_epoch, gamma=decay_rate)
        scheduler_w = torch.optim.lr_scheduler.StepLR(
            optimizer_w, step_size=step_epoch, gamma=decay_rate)
        scheduler_p = torch.optim.lr_scheduler.StepLR(
            optimizer_p, step_size=step_epoch, gamma=decay_rate)

    # Loss history: [epoch, loss_eqn_avg, loss_bc_avg, loss_data_avg, loss_total_avg]
    LOSS = []

    # -------------------------------------------------------------------------
    # Training loop — mini-batch mode (Flag_batch=True):
    #   DataLoader yields batchsize random interior points each step.
    #   BC and data losses use all points (small tensors) each step.
    #   Total loss = PDE_residual + Lambda_BC * wall_BC + Lambda_data * data
    # -------------------------------------------------------------------------
    if Flag_batch:
        for epoch in range(epochs):
            loss_eqn_tot  = 0.
            loss_bc_tot   = 0.
            loss_data_tot = 0.
            n = 0

            for batch_idx, (x_in, y_in, z_in) in enumerate(dataloader):
                net2_u.zero_grad()
                net2_v.zero_grad()
                net2_w.zero_grad()
                net2_p.zero_grad()

                loss_eqn  = criterion(x_in, y_in, z_in)
                loss_bc   = Loss_BC(xb, yb, zb, ub, vb, wb,
                                    xb_inlet, yb_inlet, ub_inlet, x, y, z)
                loss_data = Loss_data(xd, yd, zd, ud, vd, wd)
                loss = loss_eqn + Lambda_BC * loss_bc + Lambda_data * loss_data

                loss.backward()
                optimizer_u.step()
                optimizer_v.step()
                optimizer_w.step()
                optimizer_p.step()

                loss_eqn_tot  += loss_eqn.item()
                loss_bc_tot   += loss_bc.item()
                loss_data_tot += loss_data.item()
                n += 1

                if batch_idx % 40 == 0:
                    print('Epoch {:4d} [{}/{} ({:.0f}%)]  '
                          'eqn {:.6e}  BC {:.6e}  data {:.6e}'.format(
                              epoch,
                              batch_idx * len(x_in), len(dataloader.dataset),
                              100. * batch_idx / len(dataloader),
                              loss_eqn.item(), loss_bc.item(), loss_data.item()))

            if Flag_schedule:
                scheduler_u.step()
                scheduler_v.step()
                scheduler_w.step()
                scheduler_p.step()

            loss_eqn_avg   = loss_eqn_tot  / n
            loss_bc_avg    = loss_bc_tot   / n
            loss_data_avg  = loss_data_tot / n
            loss_total_avg = (loss_eqn_avg
                              + Lambda_BC   * loss_bc_avg
                              + Lambda_data * loss_data_avg)
            LOSS.append([epoch, loss_eqn_avg, loss_bc_avg,
                         loss_data_avg, loss_total_avg])

            print('*** Epoch {:4d}  avg — eqn: {:.4e}  BC: {:.4e}  '
                  'data: {:.4e}  LR: {:.2e} ***'.format(
                      epoch, loss_eqn_avg, loss_bc_avg, loss_data_avg,
                      optimizer_u.param_groups[0]['lr']))

            # Periodic checkpoint (every 1000 epochs after epoch 3000)
            if epoch % 1000 == 0 and epoch > 3000:
                _ckpt = os.path.join(path, 'IA3D_data3velsmall_{}.pt')
                torch.save(net2_u.state_dict(), _ckpt.format('u'))
                torch.save(net2_v.state_dict(), _ckpt.format('v'))
                torch.save(net2_w.state_dict(), _ckpt.format('w'))
                torch.save(net2_p.state_dict(), _ckpt.format('p'))
                print(f'  Checkpoint saved at epoch {epoch}')

    # -------------------------------------------------------------------------
    # Training loop — full-batch mode (Flag_batch=False):
    #   All interior points used every epoch. Simpler but may be slow for
    #   large meshes. Kept for completeness / debugging.
    # -------------------------------------------------------------------------
    else:
        for epoch in range(epochs):
            net2_u.zero_grad()
            net2_v.zero_grad()
            net2_w.zero_grad()
            net2_p.zero_grad()

            loss_eqn  = criterion(x, y, z)
            loss_bc   = Loss_BC(xb, yb, zb, ub, vb, wb,
                                xb_inlet, yb_inlet, ub_inlet, x, y, z)
            loss_data = Loss_data(xd, yd, zd, ud, vd, wd)

            if Flag_BC_exact:
                loss = loss_eqn + Lambda_data * loss_data
            else:
                loss = loss_eqn + Lambda_BC * loss_bc + Lambda_data * loss_data

            loss.backward()
            optimizer_u.step()
            optimizer_v.step()
            optimizer_w.step()
            optimizer_p.step()

            LOSS.append([epoch, loss_eqn.item(), loss_bc.item(),
                         loss_data.item(), loss.item()])

            if Flag_schedule:
                scheduler_u.step()
                scheduler_v.step()
                scheduler_w.step()
                scheduler_p.step()

            if epoch % 50 == 0:
                print('Epoch {:5d}/{}  LR: {:.2e}  '
                      'PDE: {:.4e}  BC: {:.4e}  data: {:.4e}'.format(
                          epoch, epochs,
                          optimizer_u.param_groups[0]['lr'],
                          loss_eqn.item(), loss_bc.item(), loss_data.item()))

    toc = time.time()
    elapseTime = toc - tic

    # =========================================================================
    # Post-training: save final model weights
    # =========================================================================
    os.makedirs(path, exist_ok=True)
    torch.save(net2_u.state_dict(), os.path.join(path, 'IA3D_data3velsmall_u.pt'))
    torch.save(net2_v.state_dict(), os.path.join(path, 'IA3D_data3velsmall_v.pt'))
    torch.save(net2_w.state_dict(), os.path.join(path, 'IA3D_data3velsmall_w.pt'))
    torch.save(net2_p.state_dict(), os.path.join(path, 'IA3D_data3velsmall_p.pt'))
    print('Model weights saved.')

    # =========================================================================
    # Post-training: save loss history as CSV
    # =========================================================================
    LOSS_arr = np.array(LOSS)
    csv_path = os.path.join(path, 'loss_history.csv')
    np.savetxt(csv_path, LOSS_arr, delimiter=',',
               header='epoch,loss_eqn,loss_bc,loss_data,loss_total', comments='')
    print(f'Loss CSV saved -> {csv_path}')

    # =========================================================================
    # Post-training: print and save training summary
    # =========================================================================
    print('\n' + '=' * 60)
    print('  POST-TRAINING SUMMARY')
    print('=' * 60)
    print(f'  Training time   : {elapseTime:.2f} s')
    print(f'  Device          : {device}')
    print(f'  Final PDE loss  : {LOSS_arr[-1, 1]:.4e}')
    print(f'  Final BC  loss  : {LOSS_arr[-1, 2]:.4e}')
    print(f'  Final data loss : {LOSS_arr[-1, 3]:.4e}')
    print(f'  Final total loss: {LOSS_arr[-1, 4]:.4e}')
    print('=' * 60 + '\n')

    log_path = os.path.join(path, 'training_summary.txt')
    with open(log_path, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('  POST-TRAINING SUMMARY\n')
        f.write('=' * 60 + '\n')
        f.write(f'  Training time   : {elapseTime:.2f} s\n')
        f.write(f'  Device          : {device}\n')
        f.write(f'  Epochs          : {epochs}\n')
        f.write(f'  Batch mode      : {Flag_batch}\n')
        f.write(f'  Diff (nu)       : {Diff}\n')
        f.write(f'  rho             : {rho}\n')
        f.write(f'  Lambda_BC       : {Lambda_BC}\n')
        f.write(f'  Lambda_data     : {Lambda_data}\n')
        f.write(f'  Final PDE loss  : {LOSS_arr[-1, 1]:.4e}\n')
        f.write(f'  Final BC  loss  : {LOSS_arr[-1, 2]:.4e}\n')
        f.write(f'  Final data loss : {LOSS_arr[-1, 3]:.4e}\n')
        f.write(f'  Final total loss: {LOSS_arr[-1, 4]:.4e}\n')
        f.write('=' * 60 + '\n')
    print(f'Training summary saved -> {log_path}')

    # =========================================================================
    # Post-training: plot loss history and save figure
    # =========================================================================
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 1], label='PDE residual loss')
    ax_loss.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 2],
                     label=f'BC loss (lambda={Lambda_BC})')
    ax_loss.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 3],
                     label=f'Data loss (lambda={Lambda_data})')
    ax_loss.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 4],
                     'k--', lw=1, label='Total loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss (log scale)')
    ax_loss.set_title('Training Loss History - 3D IA PINN')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    loss_fig_path = os.path.join(path, 'loss_history.png')
    fig_loss.savefig(loss_fig_path, dpi=150, bbox_inches='tight')
    print(f'Loss figure saved -> {loss_fig_path}')
    plt.show()

    return net2_u, net2_v, net2_w, net2_p, LOSS


# =============================================================================
# Main configuration
# -----------------------------------------------------------------------------
# Device: automatically selects MPS (Apple Silicon GPU) if available,
#         otherwise falls back to CUDA, then CPU.
# =============================================================================
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using device: MPS (Apple Silicon GPU)')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device: CUDA GPU')
else:
    device = torch.device('cpu')
    print('Using device: CPU')

Flag_batch    = True   # mini-batch training (recommended for large meshes)
Flag_BC_exact = False  # hard BC enforcement via output transform (not active)
Lambda_BC     = 20.    # weight for wall no-slip BC loss
Lambda_data   = 20.    # weight for sparse velocity data loss

# -----------------------------------------------------------------------------
# Data paths: update Directory to point to your 3D IA data files.
# -----------------------------------------------------------------------------
Directory        = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Data', '3D-aneurysm') + os.sep
mesh_file        = Directory + 'IA_mesh3D_nearwall_small_physical.vtu'
outer_wall_file  = Directory + 'IA_nearwall_outer_small.vtk'
bc_file_wall     = Directory + 'IA_nearwall_wall_small.vtk'
File_data        = Directory + 'IA_3D_unsteady3.vtu'
fieldname        = 'f_17'   # velocity field name in VTK file (verify in ParaView)

batchsize     = 512
learning_rate = 1e-5
epochs        = 8500       # 5500 + 3000
Flag_pretrain = False      # if True, loads weights from a previous run

# Physical parameters
Diff  = 0.00125  # kinematic viscosity nu  (Re ~ U_scale * L / Diff)
rho   = 1.0
T     = 0.5      # placeholder total duration (unused in steady solver)

# Domain scaling: non-dimensionalise coordinates by physical domain size
Flag_x_length = True   # if True, use X_scale / YZ_scale below
X_scale  = 3.0         # physical x-length of domain
YZ_scale = 2.0         # physical y/z-length of domain
U_scale  = 1.0         # reference velocity scale
U_BC_in  = 0.5         # inlet centreline velocity

nPt    = 130
xStart = 0.
xEnd   = 1.
yStart = 0.
yEnd   = 1.0

if not Flag_x_length:
    X_scale  = 1.
    YZ_scale = 1.

# LR schedule: StepLR — multiply by decay_rate every step_epoch epochs
Flag_schedule = True
if Flag_schedule:
    learning_rate = 5e-4
    step_epoch    = 800
    decay_rate    = 0.5

# Results directory (sibling to this script)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results') + os.sep

# =============================================================================
# Load interior mesh (collocation points)
# =============================================================================
print('Loading mesh:', mesh_file)
reader   = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print('  n_points (mesh):', n_points)

x_vtk_mesh = np.zeros((n_points, 1))
y_vtk_mesh = np.zeros((n_points, 1))
z_vtk_mesh = np.zeros((n_points, 1))
VTKpoints  = vtk.vtkPoints()
for i in range(n_points):
    pt = data_vtk.GetPoint(i)
    x_vtk_mesh[i] = pt[0]
    y_vtk_mesh[i] = pt[1]
    z_vtk_mesh[i] = pt[2]
    VTKpoints.InsertPoint(i, pt[0], pt[1], pt[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

# Scale coordinates into non-dimensional space
x = np.reshape(x_vtk_mesh, (-1, 1)) / X_scale
y = np.reshape(y_vtk_mesh, (-1, 1)) / YZ_scale
z = np.reshape(z_vtk_mesh, (-1, 1)) / YZ_scale

t = np.linspace(0., T, nPt * nPt).reshape(-1, 1)
print('shape of x:', x.shape)
print('shape of y:', y.shape)
print('shape of z:', z.shape)

# =============================================================================
# Load outer wall boundary points (used as inlet BC locations)
# =============================================================================
print('Loading outer wall:', outer_wall_file)
reader   = vtk.vtkPolyDataReader()
reader.SetFileName(outer_wall_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print('  n_points (outer wall):', n_points)

x_vtk_mesh = np.zeros((n_points, 1))
y_vtk_mesh = np.zeros((n_points, 1))
z_vtk_mesh = np.zeros((n_points, 1))
VTKpoints  = vtk.vtkPoints()
for i in range(n_points):
    pt = data_vtk.GetPoint(i)
    x_vtk_mesh[i] = pt[0]
    y_vtk_mesh[i] = pt[1]
    z_vtk_mesh[i] = pt[2]
    VTKpoints.InsertPoint(i, pt[0], pt[1], pt[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb_in = np.reshape(x_vtk_mesh, (-1, 1)) / X_scale
yb_in = np.reshape(y_vtk_mesh, (-1, 1)) / YZ_scale
zb_in = np.reshape(z_vtk_mesh, (-1, 1)) / YZ_scale

# =============================================================================
# Load wall (no-slip) boundary points
# =============================================================================
print('Loading wall BC:', bc_file_wall)
reader    = vtk.vtkPolyDataReader()
reader.SetFileName(bc_file_wall)
reader.Update()
data_vtk  = reader.GetOutput()
n_pointsw = data_vtk.GetNumberOfPoints()
print('  n_points (wall):', n_pointsw)

x_vtk_mesh = np.zeros((n_pointsw, 1))
y_vtk_mesh = np.zeros((n_pointsw, 1))
z_vtk_mesh = np.zeros((n_pointsw, 1))
VTKpoints  = vtk.vtkPoints()
for i in range(n_pointsw):
    pt = data_vtk.GetPoint(i)
    x_vtk_mesh[i] = pt[0]
    y_vtk_mesh[i] = pt[1]
    z_vtk_mesh[i] = pt[2]
    VTKpoints.InsertPoint(i, pt[0], pt[1], pt[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb_wall = np.reshape(x_vtk_mesh, (-1, 1)) / X_scale
yb_wall = np.reshape(y_vtk_mesh, (-1, 1)) / YZ_scale
zb_wall = np.reshape(z_vtk_mesh, (-1, 1)) / YZ_scale

# =============================================================================
# Set up boundary condition arrays
# Inlet: parabolic profile  u = y*(0.3 - y)/0.0225 * U_BC_in
# Wall : no-slip  u = v = w = 0
# =============================================================================
u_in_BC   = yb_in * (0.3 - yb_in) / 0.0225 * U_BC_in  # parabolic inlet profile
v_in_BC   = np.zeros((n_points,  1))
w_in_BC   = np.zeros((n_points,  1))
u_wall_BC = np.zeros((n_pointsw, 1))
v_wall_BC = np.zeros((n_pointsw, 1))
w_wall_BC = np.zeros((n_pointsw, 1))

xb = xb_wall.reshape(-1, 1)
yb = yb_wall.reshape(-1, 1)
zb = zb_wall.reshape(-1, 1)
ub = u_wall_BC.reshape(-1, 1)
vb = v_wall_BC.reshape(-1, 1)
wb = w_wall_BC.reshape(-1, 1)

xb_inlet = xb_in.reshape(-1, 1)
yb_inlet = yb_in.reshape(-1, 1)
zb_inlet = zb_in.reshape(-1, 1)
ub_inlet = u_in_BC.reshape(-1, 1)
vb_inlet = v_in_BC.reshape(-1, 1)
wb_inlet = w_in_BC.reshape(-1, 1)

print('shape of xb:', xb.shape)
print('shape of yb:', yb.shape)
print('shape of ub:', ub.shape)

# =============================================================================
# Load sparse interior velocity data points via vtkProbeFilter.
# Sample every N_sample-th point from the outer wall mesh as data locations;
# the probe filter interpolates the CFD velocity field to these locations.
# =============================================================================
N_sample = 200  # subsample stride: take every N_sample-th outer-wall point

print('Loading outer wall for data sampling:', outer_wall_file)
reader   = vtk.vtkPolyDataReader()
reader.SetFileName(outer_wall_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print('  n_points (outer wall, for data probe):', n_points)

x_vtk_mesh = np.zeros((n_points, 1))
y_vtk_mesh = np.zeros((n_points, 1))
z_vtk_mesh = np.zeros((n_points, 1))
N_pts_data = 0
for i in range(n_points):
    if i % N_sample == 0:
        pt = data_vtk.GetPoint(i)
        x_vtk_mesh[N_pts_data] = pt[0]
        y_vtk_mesh[N_pts_data] = pt[1]
        z_vtk_mesh[N_pts_data] = pt[2]
        N_pts_data += 1

print('  n_points sampled for data:', N_pts_data)

x_data = x_vtk_mesh[:N_pts_data, :]
y_data = y_vtk_mesh[:N_pts_data, :]
z_data = z_vtk_mesh[:N_pts_data, :]

print('Loading velocity data file:', File_data)
reader   = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(File_data)
reader.Update()
data_vtk = reader.GetOutput()
print('  n_points (data file):', data_vtk.GetNumberOfPoints())

# Probe CFD field at the sampled data locations
VTKpoints = vtk.vtkPoints()
for i in range(N_pts_data):
    VTKpoints.InsertPoint(i, x_data[i, 0], y_data[i, 0], z_data[i, 0])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

probe = vtk.vtkProbeFilter()
probe.SetInputData(point_data)
probe.SetSourceData(data_vtk)
probe.Update()
array    = probe.GetOutput().GetPointData().GetArray(fieldname)
data_vel = VN.vtk_to_numpy(array)

# Non-dimensionalise velocity and scale coordinates
data_vel_u = data_vel[:, 0] / U_scale
data_vel_v = data_vel[:, 1] / U_scale
data_vel_w = data_vel[:, 2] / U_scale
x_data     = x_data / X_scale
y_data     = y_data / YZ_scale
z_data     = z_data / YZ_scale

print('Data pts (x,y,z):', x_data.T, y_data.T, z_data.T)
print('Data vel  u     :', data_vel_u)
print('Data vel  v     :', data_vel_v)
print('Data vel  w     :', data_vel_w)

xd = x_data.reshape(-1, 1)
yd = y_data.reshape(-1, 1)
zd = z_data.reshape(-1, 1)
ud = data_vel_u.reshape(-1, 1)
vd = data_vel_v.reshape(-1, 1)
wd = data_vel_w.reshape(-1, 1)

# =============================================================================
# Print configuration summary before training
# =============================================================================
print('\n' + '=' * 60)
print('  PROBLEM SETUP')
print('=' * 60)
print(f'  PDE     : Steady incompressible Navier-Stokes (3D)')
print(f'  nu(Diff): {Diff}  |  rho: {rho}  |  Re ~ {int(U_scale / Diff)}')
print(f'  X_scale : {X_scale}  |  YZ_scale: {YZ_scale}  |  U_scale: {U_scale}')
print(f'  Mesh pts: {x.shape[0]}')
print(f'  Data pts: {N_pts_data}')
print('=' * 60)
print('  TRAINING CONFIG')
print('=' * 60)
print(f'  Device      : {device}')
print(f'  Epochs      : {epochs}')
print(f'  Batch mode  : {Flag_batch}  (batch size: {batchsize})')
print(f'  Lambda_BC   : {Lambda_BC}  |  Lambda_data: {Lambda_data}')
if Flag_schedule:
    print(f'  LR schedule : StepLR  step={step_epoch},  gamma={decay_rate}')
else:
    print('  LR schedule : None')
print('=' * 60 + '\n')

# =============================================================================
# Run training
# =============================================================================
net2_u, net2_v, net2_w, net2_p, LOSS_history = geo_train(
    device, x, y, z,
    xb, yb, zb, ub, vb, wb,
    xd, yd, zd, ud, vd, wd,
    batchsize, learning_rate, epochs, path,
    Flag_batch, Diff, rho, Flag_BC_exact, Lambda_BC,
    nPt, T,
    xb_inlet, yb_inlet, zb_inlet,
    ub_inlet, vb_inlet, wb_inlet)