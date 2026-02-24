# =============================================================================
# Imports
# -----------------------------------------------------------------------------
# torch / torch.nn / torch.optim : core PyTorch — tensors, layers, optimizers
# torch.autograd                 : automatic differentiation for dC/dx, d2C/dx2
# numpy                          : array construction for collocation pts & analytical soln
# matplotlib.pyplot              : plotting results after training
# DataLoader / TensorDataset     : mini-batch training infrastructure
# math.exp                       : evaluating the exact analytical solution
# time                           : wall-clock timing of training
# =============================================================================
import torch
import numpy as np
#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import math
import os


def geo_train(device,x_in,xb,cb,batchsize,learning_rate,epochs,path,Flag_batch,C_analytical,Vel,Diff):
	# -------------------------------------------------------------------------
	# Data setup: convert NumPy collocation array to PyTorch tensor.
	# In batch mode, wrap in a DataLoader for mini-batch iteration (batch_size
	# random points per step). In full-batch mode, use all points every epoch.
	# -------------------------------------------------------------------------
	if (Flag_batch):
		x = torch.Tensor(x_in)
		dataset = TensorDataset(x,x)
		dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
	else:
		x = torch.Tensor(x_in)
	h_n = 100   # number of neurons per hidden layer
	input_n = 1 # this is what our answer is a function of. 
	# -------------------------------------------------------------------------
	# Swish activation: f(x) = x * sigmoid(x)
	# Smooth and non-monotonic — preserves well-behaved higher-order derivatives
	# (needed for d2C/dx2), generally outperforming tanh/ReLU in PINNs.
	# inplace=True modifies the tensor in memory for efficiency.
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
	# Network architecture: fully-connected feed-forward neural network.
	#   Input  : 1D spatial coordinate x
	#   Hidden : 10 layers x 100 neurons, Swish activation
	#   Output : 1D predicted concentration C(x)
	# The forward() method optionally applies a hard BC transformation:
	#   output * (1 - x) + 0.1  =>  guarantees C(1) = 0.1 exactly,
	#   eliminating the need to train that BC into the weights.
	# -------------------------------------------------------------------------
	class Net2(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),


				Swish(),
				nn.Linear(h_n,h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				Swish(),

				#######
				nn.Linear(h_n,h_n),

				Swish(),


				nn.Linear(h_n,h_n),

				Swish(),


				#######
				#nn.Linear(h_n,h_n),

				#Swish(),
				
				#nn.Linear(h_n,h_n),

				#Swish(),



				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			if (Flag_BC_near_wall):
				#output = output*x*(1-x) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
				return  output * (1-x) + 0.1 #enforce BC at the wall region of interest
			else:
				return  output #* (x) + 1.0
				
	
	################################################################

	net2 = Net2().to(device)
	
	# -------------------------------------------------------------------------
	# Weight initialization: Kaiming (He) normal initialization.
	# Sets weights from N(0, sqrt(2/n_in)), preventing vanishing/exploding
	# gradients in the 10-layer deep network.
	# -------------------------------------------------------------------------
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net2.apply(init_normal)


	############################################################################

	# -------------------------------------------------------------------------
	# Optimizer: Adam with betas=(0.9, 0.99) and eps=1e-15.
	# Higher beta2 (0.99 vs default 0.999) makes adaptive scaling react faster.
	# Very small eps prevents division by near-zero, important for PINN gradients.
	# -------------------------------------------------------------------------
	optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	

	def criterion(x):

		#print (x)
		#x = torch.Tensor(x).to(device)
	

		x.requires_grad = True
		

		net_in = x
		C = net2(net_in)
		C = C.view(len(C),-1)



		
		c_x = torch.autograd.grad(
    		C,                          # output  : what to differentiate
    		x,                          # input   : differentiate w.r.t. this
    		grad_outputs=torch.ones_like(x),  # explained below
    		create_graph=True,          # keep graph alive for 2nd derivative
    		only_inputs=True            # only compute grad w.r.t. x, not all params
			)[0]
		c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		
		loss_1 = Vel * c_x - Diff * c_xx




		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss

	###################################################################
	# Boundary condition loss (soft enforcement):
	#   Evaluates the network at the two boundary points x=0 and x=1
	#   and penalizes deviation from the known Dirichlet values C(0)=1, C(1)=0.1.
	#   NOTE: currently defined but commented out of the training loop —
	#   BCs are instead enforced via Loss_data or hard-enforcement (Flag_BC_near_wall).
	def Loss_BC(xb,cb):
		xb = torch.FloatTensor(xb).to(device)
		cb = torch.FloatTensor(cb).to(device)

		#xb.requires_grad = True
		
		#net_in = torch.cat((xb),1)
		out = net2(xb)
		cNN = out.view(len(out), -1)
		#cNN = cNN*(1.-xb) + cb    #cNN*xb*(1-xb) + cb
		loss_f = nn.MSELoss()
		loss_bc = loss_f(cNN, cb)
		return loss_bc

	###################################################################
	# Sparse data loss:
	#   Penalizes the network for deviating from known true values at a small
	#   number of interior points (xd, cd). These anchor points are sampled
	#   from the analytical solution at x~0.5, 0.75, 0.98 — clustered near
	#   the sharp boundary layer — giving the network a signal to locate it
	#   without needing dense data everywhere. Weighted by Lambda_data=10.
	def Loss_data(xd,cd):
		xd = torch.FloatTensor(xd).to(device)
		cd = torch.FloatTensor(cd).to(device)

		#xb.requires_grad = True
		

		out = net2(xd)
		out = out.view(len(out), -1)
		loss_f = nn.MSELoss()
		loss_bc = loss_f(out, cd)

		return loss_bc



	# -------------------------------------------------------------------------
	# Training setup:
	#   xd, cd : 3 sparse supervision points drawn from the analytical solution
	#            at x ~ 0.3, 0.90, 0.98 (near the boundary layer at x=1).
	#   scheduler: StepLR step-decay — LR multiplied by decay_rate every
	#              step_epoch epochs (e.g. 1e-3 -> 2e-4 -> 4e-5 ...), allowing
	#              large steps early and fine-tuning later.
	# -------------------------------------------------------------------------
	# Main loop
	LOSS = []
	tic = time.time()



	xd = x_in[ [ int(3*nPt/10), int(9*nPt/10), int(9.8*nPt/10)    ] ] 
	cd = C_analytical[ [ int(3*nPt/10), int(9*nPt/10), int(9.8*nPt/10)   ] ]


	if (Flag_schedule):
		scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_epoch, gamma=decay_rate)


	# -------------------------------------------------------------------------
	# Training loop — two modes:
	#   Mini-batch (Flag_batch=True) : DataLoader yields batchsize random pts
	#     per step; can escape local minima but introduces gradient noise.
	#   Full-batch (Flag_batch=False): all nPt collocation points used every
	#     epoch; simpler and more stable for 1D problems.
	# Total loss = PDE_residual_loss + Lambda_data * sparse_data_loss
	# -------------------------------------------------------------------------
	if(Flag_batch):# This one uses dataloader
		for epoch in range(epochs):
			for batch_idx, (x_in2,x_in2) in enumerate(dataloader):
				#zero gradient
				#net1.zero_grad()
				##Closure function for LBFGS loop:
				#def closure():
				net2.zero_grad()
				#x_in2 = torch.FloatTensor(x_in2)
				#print (x_in2)
				#print (x)
				#print('shape of x_in2',x_in2.shape)
				loss_eqn = criterion(x_in2)
				loss_data = Loss_data(xd,cd)
				loss = loss_eqn + Lambda_data * loss_data
				loss.backward()
				optimizer2.step()
			LOSS.append([epoch, loss_eqn.item(), loss_data.item(), loss.item()])
			if (Flag_schedule):
					scheduler_c.step()

			if epoch % 50  ==0:
					print('Epoch {:5d}/{} | LR: {:.2e} | Loss PDE: {:.4e} | Loss data: {:.4e} | Loss total: {:.4e}'.format(
						epoch, epochs,
						optimizer2.param_groups[0]['lr'],
						loss_eqn.item(), loss_data.item(), loss.item()))
	else:
		for epoch in range(epochs):
			#zero gradient
			#net1.zero_grad()
			##Closure function for LBFGS loop:
			#def closure():
			net2.zero_grad()
			loss_eqn = criterion(x)
			#loss_bc = Loss_BC(xb,cb)
			loss_data = Loss_data(xd,cd)
			loss = loss_eqn + Lambda_data * loss_data
			
			loss.backward()

			optimizer2.step()
			LOSS.append([epoch, loss_eqn.item(), loss_data.item(), loss.item()])

			if (Flag_schedule):
				scheduler_c.step()

			if epoch % 50 ==0:
				print('Epoch {:5d}/{} | LR: {:.2e} | Loss PDE: {:.4e} | Loss data: {:.4e} | Loss total: {:.4e}'.format(
					epoch, epochs,
					optimizer2.param_groups[0]['lr'],
					loss_eqn.item(), loss_data.item(), loss.item()))
		

	toc = time.time()
	elapseTime = toc - tic

	# -------------------------------------------------------------------------
	# Post-training evaluation: error metrics
	# -------------------------------------------------------------------------
	with torch.no_grad():
		C_Result = net2(x).numpy()          # shape (nPt, 1)

	L2_abs = np.sqrt(np.mean((C_Result - C_analytical)**2))
	L2_rel = L2_abs / np.sqrt(np.mean(C_analytical**2))
	max_err = np.max(np.abs(C_Result - C_analytical))

	with torch.no_grad():
		C_at_0 = net2(torch.FloatTensor([[0.]])).item()
		C_at_1 = net2(torch.FloatTensor([[1.]])).item()

	print("\n" + "="*60)
	print("  POST-TRAINING SUMMARY")
	print("="*60)
	print(f"  Training time         : {elapseTime:.2f} s")
	print(f"  Final PDE loss        : {LOSS[-1][1]:.4e}")
	print(f"  Final data loss       : {LOSS[-1][2]:.4e}")
	print(f"  Final total loss      : {LOSS[-1][3]:.4e}")
	print(f"  L2 absolute error     : {L2_abs:.4e}")
	print(f"  L2 relative error     : {L2_rel:.4e}")
	print(f"  Max pointwise error   : {max_err:.4e}")
	print(f"  BC check  C(0)={C_at_0:.6f}  (target {C_BC1:.1f})")
	print(f"  BC check  C(1)={C_at_1:.6f}  (target {C_BC2:.1f})")
	print("="*60 + "\n")

	# -------------------------------------------------------------------------
	# Save summary log to file
	# -------------------------------------------------------------------------
	os.makedirs(path, exist_ok=True)
	log_path = os.path.join(path, 'training_summary.txt')
	with open(log_path, 'w') as f:
		f.write("=" * 60 + "\n")
		f.write("  POST-TRAINING SUMMARY\n")
		f.write("=" * 60 + "\n")
		f.write(f"  Training time         : {elapseTime:.2f} s\n")
		f.write(f"  Epochs                : {epochs}\n")
		f.write(f"  nPt (collocation)     : {nPt}\n")
		f.write(f"  Vel={Vel},  Diff={Diff},  Pe={Vel/Diff:.0f}\n")
		f.write(f"  Lambda_data           : {Lambda_data}\n")
		f.write(f"  Final PDE loss        : {LOSS[-1][1]:.4e}\n")
		f.write(f"  Final data loss       : {LOSS[-1][2]:.4e}\n")
		f.write(f"  Final total loss      : {LOSS[-1][3]:.4e}\n")
		f.write(f"  L2 absolute error     : {L2_abs:.4e}\n")
		f.write(f"  L2 relative error     : {L2_rel:.4e}\n")
		f.write(f"  Max pointwise error   : {max_err:.4e}\n")
		f.write(f"  BC check  C(0)={C_at_0:.6f}  (target {C_BC1:.1f})\n")
		f.write(f"  BC check  C(1)={C_at_1:.6f}  (target {C_BC2:.1f})\n")
		f.write("=" * 60 + "\n")
	print(f"  Log saved  -> {log_path}")
	#   [0,0] Solution comparison   [0,1] Loss history (log)
	#   [1,0] Pointwise abs. error  [1,1] First-derivative comparison
	# -------------------------------------------------------------------------
	LOSS_arr = np.array(LOSS)   # columns: epoch, loss_eqn, loss_data, loss_total
	x_np     = x.detach().numpy()

	fig, axes = plt.subplots(2, 2, figsize=(12, 9))
	fig.suptitle(
		f'PINN — 1D Steady Advection-Diffusion  '
		f'(Pe = {Vel/Diff:.0f},  {epochs} epochs,  {nPt} pts)',
		fontsize=12, fontweight='bold'
	)

	# --- [0,0] Solution comparison ---
	ax = axes[0, 0]
	ax.plot(x_np, C_analytical, 'b--', lw=2, label='Analytical', alpha=0.85)
	ax.plot(x_np, C_Result,     'go',  ms=4, label='PINN',       alpha=0.7)
	ax.plot(xd,   cd,           'r+',  ms=10, mew=2, label='Sparse data (3 pts)')
	ax.set_xlabel('x')
	ax.set_ylabel('C(x)')
	ax.set_title(f'Solution  |  L2 rel. error = {L2_rel:.2e}')
	ax.legend()
	ax.grid(True, alpha=0.3)

	# --- [0,1] Loss history (log scale) ---
	ax = axes[0, 1]
	ax.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 1], label='PDE residual loss', alpha=0.85)
	ax.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 2], label=f'Data loss  (λ={Lambda_data})', alpha=0.85)
	ax.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 3], 'k--', lw=1, label='Total loss', alpha=0.6)
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Loss  (log scale)')
	ax.set_title('Training Loss History')
	ax.legend()
	ax.grid(True, alpha=0.3)

	# --- [1,0] Pointwise absolute error (log scale) ---
	ax = axes[1, 0]
	abs_err = np.abs(C_Result - C_analytical).flatten()
	ax.semilogy(x_np, abs_err, 'r-', lw=1.5)
	ax.set_xlabel('x')
	ax.set_ylabel('|C_PINN − C_exact|  (log scale)')
	ax.set_title(f'Pointwise Absolute Error  |  Max = {max_err:.2e}')
	ax.grid(True, alpha=0.3)

	# --- [1,1] First-derivative comparison dC/dx ---
	ax = axes[1, 1]
	x_t  = torch.FloatTensor(x).requires_grad_(True)
	C_t  = net2(x_t)
	dCdx_pinn = torch.autograd.grad(
		C_t, x_t,
		grad_outputs=torch.ones_like(x_t),
		create_graph=False
	)[0].detach().numpy()
	dCdx_exact = A * (Vel / Diff) * np.exp(Vel / Diff * x_np)
	ax.plot(x_np, dCdx_exact, 'b--', lw=2, label='Analytical dC/dx', alpha=0.85)
	ax.plot(x_np, dCdx_pinn,  'g-',  lw=1.5, label='PINN dC/dx',     alpha=0.85)
	ax.set_xlabel('x')
	ax.set_ylabel('dC/dx')
	ax.set_title('First Derivative Comparison')
	ax.legend()
	ax.grid(True, alpha=0.3)

	plt.tight_layout()
	fig_path = os.path.join(path, 'results_plot.png')
	plt.savefig(fig_path, dpi=150, bbox_inches='tight')
	print(f"  Figure saved -> {fig_path}")
	plt.show()

	# -------------------------------------------------------------------------
	# Save loss history as CSV
	# -------------------------------------------------------------------------
	csv_path = os.path.join(path, 'loss_history.csv')
	np.savetxt(csv_path, LOSS_arr, delimiter=',',
			   header='epoch,loss_pde,loss_data,loss_total', comments='')
	print(f"  Loss CSV saved -> {csv_path}\n")

	return net2, LOSS




# =============================================================================
# Main configuration
# -----------------------------------------------------------------------------
# Physical problem: 1D steady advection-diffusion on x in [0,1]
#   Vel*dC/dx - Diff*d2C/dx2 = 0,  C(0)=1,  C(1)=0.1
#   Peclet number Pe = Vel/Diff = 100  =>  sharp boundary layer near x=1
#
# Flags:
#   Flag_batch       : mini-batch (True) vs full-batch (False) training
#   Flag_Chebyshev   : Chebyshev collocation pts (cluster near BL) vs uniform
#   Flag_BC_near_wall: hard-enforce C(1)=0.1 in network output transform
#   Flag_schedule    : use step-decay LR scheduler
#
# Loss weighting:
#   Lambda_data = 10  =>  sparse data loss weighted 10x vs PDE residual loss
# =============================================================================
#Main code:
device = torch.device("cpu")
epochs  = 5000 

Flag_batch =  False #USe batch or not  
Flag_Chebyshev = False   #Use Chebyshev pts for more accurcy in BL region
Flag_BC_near_wall =False  # True #If True sets BC at just the boundary of interet

Lambda_data = 10.  #Data lambda

Vel = 1.0
Diff= 0.01 

nPt = 100 
xStart = 0.
xEnd = 1.

batchsize = 32 
learning_rate = 1e-6 


Flag_schedule = True  #If true change the learning rate 
if (Flag_schedule):
	learning_rate = 1e-3  #starting learning rate
	step_epoch = 1000 
	decay_rate = 0.2  





# =============================================================================
# Collocation point generation
# Constructs the interior training points where the PDE residual is enforced.
# Chebyshev points cluster near boundaries (useful for resolving the BL) but
# are disabled by default — uniform spacing is used instead.
# Final shape is (nPt, 1) because nn.Linear expects 2D input.
# =============================================================================
if(Flag_Chebyshev): #!!!Not a very good idea (makes even the simpler case worse)
	x = np.polynomial.chebyshev.chebpts1(2*nPt)
	x = x[nPt:]  # take the right half, mapped to [0, 1]
	if(0):#Mannually place more pts at the BL
		x = np.linspace(0.95, xEnd, nPt)
		x[1] = 0.2
		x[2] = 0.5
	x[0] = 0.
	x[-1] = xEnd
	x = np.reshape(x, (nPt,1))
else:
	x = np.linspace(xStart, xEnd, nPt)
	x = np.reshape(x, (nPt,1))


print('shape of x',x.shape)

#boundary pt and boundary condition
#X_BC_loc = 1.
#C_BC = 1.
#xb = np.array([X_BC_loc],dtype=np.float32)
#cb = np.array([C_BC ], dtype=np.float32)
# =============================================================================
# Boundary conditions
# xb, cb: the two Dirichlet BC locations and values, reshaped to (2,1) for
# compatibility with the network input format.
# =============================================================================
C_BC1 = 1.
C_BC2 = 0.1
xb = np.array([0.,1.],dtype=np.float32)
cb = np.array([C_BC1,C_BC2 ], dtype=np.float32)
xb= xb.reshape(-1, 1) #need to reshape to get 2D array
cb= cb.reshape(-1, 1) #need to reshape to get 2D array
#xb = np.transpose(xb)  #transpose because of the order that NN expects instances of training data
#cb = np.transpose(cb)




path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")

# =============================================================================
# Analytical solution: C(x) = A*exp(Vel/Diff * x) + B
#   A = (C_BC2 - C_BC1) / (exp(Vel/Diff) - 1)
#   B = C_BC1 - A
# Evaluated at all collocation points. Used as ground truth for plotting
# and to provide the sparse supervision data points (xd, cd) during training.
# =============================================================================
#Analytical soln
A = (C_BC2 - C_BC1) / (exp(Vel/Diff) - 1)
B = C_BC1 - A
C_analytical = A*np.exp(Vel/Diff*x[:] ) + B



# =============================================================================
# Print problem and training configuration summary before starting
# =============================================================================
Pe = Vel / Diff
print("\n" + "="*60)
print("  PROBLEM SETUP")
print("="*60)
print(f"  PDE         : Vel*dC/dx - Diff*d2C/dx2 = 0")
print(f"  Domain      : x in [{xStart}, {xEnd}]")
print(f"  BCs         : C({xStart}) = {C_BC1},  C({xEnd}) = {C_BC2}")
print(f"  Vel={Vel},  Diff={Diff},  Peclet number Pe = {Pe:.0f}")
print(f"  Collocation : {nPt} pts  ({'Chebyshev' if Flag_Chebyshev else 'uniform'})")
print("="*60)
print("  TRAINING CONFIG")
print("="*60)
print(f"  Epochs      : {epochs}")
print(f"  Optimizer   : Adam  (lr0={learning_rate:.2e}, beta1=0.9, beta2=0.99, eps=1e-15)")
if Flag_schedule:
	print(f"  LR schedule : StepLR  step={step_epoch},  gamma={decay_rate}")
else:
	print(f"  LR schedule : None")
print(f"  Batch mode  : {'mini-batch (size=' + str(batchsize) + ')' if Flag_batch else 'full-batch'}")
print(f"  Lambda_data : {Lambda_data}  (data loss weight)")
print(f"  Hard BC     : {'on (Flag_BC_near_wall)' if Flag_BC_near_wall else 'off'}")
print("="*60 + "\n")

#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
net2_final, LOSS_history = geo_train(device,x,xb,cb,batchsize,learning_rate,epochs,path,Flag_batch,C_analytical,Vel,Diff)

 






