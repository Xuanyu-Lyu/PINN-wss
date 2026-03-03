# =============================================================================
# LP Allele Spread — PINN with Fisher-KPP equation
# -----------------------------------------------------------------------------
# Models the geographic and temporal spread of the lactase-persistence (LP)
# allele using a Physics-Informed Neural Network.
#
# Governing PDE (Fisher-KPP in 2-D geographic space):
#
#   ∂p/∂τ = D · (∂²p/∂lat² + ∂²p/∂lon²) + s · p · (1 − p)
#
#   p   : LP allele frequency ∈ [0, 1]
#   τ   : forward time  (τ = T_max − mean_date_BP,  τ=0 at oldest sample)
#   D   : diffusion coefficient  (deg² yr⁻¹)  — inferred
#   s   : selection coefficient  (yr⁻¹)        — inferred
#
# Inputs  : (lat, lon, τ)  scaled to [0, 1]
# Output  : p ∈ (0, 1)  via Sigmoid
#
# D and s are parameterised as exp(log_D), exp(log_s) to guarantee positivity.
#
# Data source: cleaned_adna_pinn.csv
#   Columns: lat, long, mean_date (yr BP), LP_allele_count (0 / 1 / 2)
#
# Each sample is a single diploid individual carrying 0, 1, or 2 copies of the
# LP allele.  The network output p is the latent *population* allele frequency.
# Individuals are modelled as Binomial(n=2, p) draws, so the data loss is the
# negative Binomial log-likelihood — statistically correct for single-individual
# ancient DNA data rather than the MSE on count/2.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time


# =============================================================================
# Training function
# =============================================================================
def lp_train(device,
             lat_col, long_col, tau_col,          # collocation pts (scaled)
             lat_d, long_d, tau_d, p_d,           # observation pts (scaled) + raw allele counts
             batchsize, learning_rate, epochs, path,
             Flag_batch, Lambda_data,
             lat_scale, long_scale, tau_scale):
    """
    Train a PINN to infer D and s for the Fisher-KPP model of LP allele spread.

    Parameters
    ----------
    lat_col, long_col, tau_col : (N,1) float32 arrays  — PDE collocation points
    lat_d, long_d, tau_d       : (M,1) float32 arrays  — observation locations (scaled)
    p_d                        : (M,1) float32 array   — raw LP allele counts {0, 1, 2}
                                 Each individual is treated as a Binomial(n=2, p) draw.
                                 The network infers the underlying population frequency p.
    lat_scale, long_scale, tau_scale : float            — physical scaling factors
    """
    # -------------------------------------------------------------------------
    # Tensors
    # -------------------------------------------------------------------------
    lat_d_t   = torch.FloatTensor(lat_d).to(device)
    long_d_t  = torch.FloatTensor(long_d).to(device)
    tau_d_t   = torch.FloatTensor(tau_d).to(device)
    p_d_t     = torch.FloatTensor(p_d).to(device)

    if Flag_batch:
        lat_col_t  = torch.FloatTensor(lat_col).to(device)
        long_col_t = torch.FloatTensor(long_col).to(device)
        tau_col_t  = torch.FloatTensor(tau_col).to(device)
        dataset    = TensorDataset(lat_col_t, long_col_t, tau_col_t)
        dataloader = DataLoader(dataset, batch_size=batchsize,
                                shuffle=True, num_workers=0, drop_last=True)
    else:
        lat_col_t  = torch.FloatTensor(lat_col).to(device)
        long_col_t = torch.FloatTensor(long_col).to(device)
        tau_col_t  = torch.FloatTensor(tau_col).to(device)

    # -------------------------------------------------------------------------
    # Smooth activation: f(x) = x · σ(x)  (non-inplace, safe for 2nd derivatives)
    # -------------------------------------------------------------------------
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    # -------------------------------------------------------------------------
    # Network  NetP : (lat_nd, lon_nd, τ_nd) → p ∈ (0, 1)
    # 6 hidden layers × 64 neurons, Sigmoid output enforces physical range.
    # -------------------------------------------------------------------------
    h_n = 64

    class NetP(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Sequential(
                nn.Linear(3, h_n),   Swish(),
                nn.Linear(h_n, h_n), Swish(),
                nn.Linear(h_n, h_n), Swish(),
                nn.Linear(h_n, h_n), Swish(),
                nn.Linear(h_n, h_n), Swish(),
                nn.Linear(h_n, h_n), Swish(),
                nn.Linear(h_n, 1),
                nn.Sigmoid(),           # p ∈ (0, 1)
            )

        def forward(self, x):
            return self.main(x)

    net_p = NetP().to(device)

    def init_normal(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    net_p.apply(init_normal)

    # -------------------------------------------------------------------------
    # Trainable log-parameters  (exp ensures D, s > 0 throughout training)
    # Initial guesses: D ≈ 1  deg²/yr,  s ≈ 0.001 yr⁻¹
    # -------------------------------------------------------------------------
    log_D = nn.Parameter(torch.tensor([0.0],  device=device))   # D = exp(0) = 1
    log_s = nn.Parameter(torch.tensor([-3.0], device=device))   # s = exp(-3) ≈ 5e-4

    optimizer = optim.Adam(
        list(net_p.parameters()) + [log_D, log_s],
        lr=learning_rate, betas=(0.9, 0.99), eps=1e-15)

    # -------------------------------------------------------------------------
    # KPP PDE residual loss
    #
    # Network inputs are non-dimensional:
    #   lat_nd  = lat  / lat_scale
    #   lon_nd  = lon  / lon_scale
    #   τ_nd    = τ    / τ_scale
    #
    # Chain-rule back to physical space:
    #   ∂p/∂τ       = (1/τ_scale)  · ∂p/∂τ_nd
    #   ∂²p/∂lat²   = (1/lat_scale²) · ∂²p/∂lat_nd²
    #   ∂²p/∂lon²   = (1/lon_scale²) · ∂²p/∂lon_nd²
    #
    # Residual (zero at exact solution):
    #   r = ∂p/∂τ_nd / τ_scale  −  D·(p_ll/lat_s² + p_gg/lon_s²)  −  s·p·(1−p)
    # -------------------------------------------------------------------------
    def criterion(lat, lon, tau):
        lat.requires_grad  = True
        lon.requires_grad  = True
        tau.requires_grad  = True

        D = torch.exp(log_D)
        s = torch.exp(log_s)

        net_in = torch.cat((lat, lon, tau), dim=1)
        p = net_p(net_in)

        # First-order derivatives
        p_tau  = torch.autograd.grad(p, tau,
                                     grad_outputs=torch.ones_like(p),
                                     create_graph=True, only_inputs=True)[0]
        p_lat  = torch.autograd.grad(p, lat,
                                     grad_outputs=torch.ones_like(p),
                                     create_graph=True, only_inputs=True)[0]
        p_lon  = torch.autograd.grad(p, lon,
                                     grad_outputs=torch.ones_like(p),
                                     create_graph=True, only_inputs=True)[0]
        # Second-order spatial derivatives (Laplacian)
        p_ll   = torch.autograd.grad(p_lat, lat,
                                     grad_outputs=torch.ones_like(p_lat),
                                     create_graph=True, only_inputs=True)[0]
        p_gg   = torch.autograd.grad(p_lon, lon,
                                     grad_outputs=torch.ones_like(p_lon),
                                     create_graph=True, only_inputs=True)[0]

        # KPP residual in physical units
        residual = (p_tau  / tau_scale
                    - D * (p_ll / lat_scale**2 + p_gg / long_scale**2)
                    - s * p * (1.0 - p))

        loss_f = nn.MSELoss()
        return loss_f(residual, torch.zeros_like(residual))

    # -------------------------------------------------------------------------
    # Data loss: negative Binomial log-likelihood
    #
    # Each individual is a diploid sample with n=2 allele draws from a
    # population with true LP allele frequency p (the network output).
    # The Binomial likelihood for observed count k ∈ {0, 1, 2} is:
    #
    #   log L_i = k · log p  +  (2 − k) · log(1 − p)
    #
    # (The combinatorial constant C(2,k) is the same for all samples and
    #  can be dropped without affecting the gradient.)
    #
    # Minimising the mean NLL is statistically efficient for this data type,
    # unlike MSE on k/2 which is unbiased but high-variance.
    # -------------------------------------------------------------------------
    def loss_data_fn(lat_d, lon_d, tau_d, allele_count_d):
        net_in  = torch.cat((lat_d, lon_d, tau_d), dim=1)
        p_pred  = net_p(net_in).clamp(1e-6, 1.0 - 1e-6)   # numerical safety for log
        # Binomial NLL for n=2 trials
        nll = -(allele_count_d           * torch.log(p_pred)
                + (2.0 - allele_count_d) * torch.log(1.0 - p_pred))
        return nll.mean()

    # -------------------------------------------------------------------------
    # LR scheduler
    # -------------------------------------------------------------------------
    tic = time.time()

    if Flag_schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_epoch, gamma=decay_rate)

    # -------------------------------------------------------------------------
    # Training loop — mini-batch mode
    # -------------------------------------------------------------------------
    LOSS = []

    # Early stopping
    es_patience  = 200    # epochs without improvement before stopping
    es_min_delta = 1e-7   # minimum improvement to reset counter
    best_loss    = np.inf
    es_counter   = 0

    if Flag_batch:
        for epoch in range(epochs):
            loss_pde_tot  = 0.
            loss_data_tot = 0.
            n = 0

            for batch_idx, (lat_b, lon_b, tau_b) in enumerate(dataloader):
                optimizer.zero_grad()

                loss_pde  = criterion(lat_b, lon_b, tau_b)
                loss_obs  = loss_data_fn(lat_d_t, long_d_t, tau_d_t, p_d_t)
                loss      = loss_pde + Lambda_data * loss_obs

                loss.backward()
                optimizer.step()

                loss_pde_tot  += loss_pde.item()
                loss_data_tot += loss_obs.item()
                n += 1

            if Flag_schedule:
                scheduler.step()

            D_val = torch.exp(log_D).item()
            s_val = torch.exp(log_s).item()
            LOSS.append([epoch,
                         loss_pde_tot  / n,
                         loss_data_tot / n,
                         (loss_pde_tot + Lambda_data * loss_data_tot) / n,
                         D_val, s_val])

            if epoch % 200 == 0:
                print('Epoch {:4d}  PDE: {:.4e}  data: {:.4e}  '
                      'D: {:.4e}  s: {:.4e}  LR: {:.2e}'.format(
                          epoch, LOSS[-1][1], LOSS[-1][2], D_val, s_val,
                          optimizer.param_groups[0]['lr']))

            # Early stopping
            current_loss = LOSS[-1][3]
            if best_loss - current_loss > es_min_delta:
                best_loss  = current_loss
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= es_patience:
                    print(f'Early stopping at epoch {epoch}  '
                          f'(no improvement for {es_patience} epochs)')
                    break

    # -------------------------------------------------------------------------
    # Training loop — full-batch mode
    # -------------------------------------------------------------------------
    else:
        for epoch in range(epochs):
            optimizer.zero_grad()

            loss_pde  = criterion(lat_col_t, long_col_t, tau_col_t)
            loss_obs  = loss_data_fn(lat_d_t, long_d_t, tau_d_t, p_d_t)
            loss      = loss_pde + Lambda_data * loss_obs

            loss.backward()
            optimizer.step()

            if Flag_schedule:
                scheduler.step()

            D_val = torch.exp(log_D).item()
            s_val = torch.exp(log_s).item()
            LOSS.append([epoch, loss_pde.item(), loss_obs.item(),
                         loss.item(), D_val, s_val])

            if epoch % 200 == 0:
                print('Epoch {:5d}  LR: {:.2e}  '
                      'PDE: {:.4e}  data: {:.4e}  D: {:.4e}  s: {:.4e}'.format(
                          epoch,
                          optimizer.param_groups[0]['lr'],
                          loss_pde.item(), loss_obs.item(), D_val, s_val))

            # Early stopping
            current_loss = loss.item()
            if best_loss - current_loss > es_min_delta:
                best_loss  = current_loss
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= es_patience:
                    print(f'Early stopping at epoch {epoch}  '
                          f'(no improvement for {es_patience} epochs)')
                    break

    toc = time.time()
    elapsed = toc - tic

    # =========================================================================
    # Save model weights and inferred parameters
    # =========================================================================
    os.makedirs(path, exist_ok=True)
    torch.save(net_p.state_dict(), os.path.join(path, 'lp_p.pt'))
    torch.save({'log_D': log_D.detach().cpu(),
                'log_s': log_s.detach().cpu()},
               os.path.join(path, 'lp_params.pt'))
    print("Model weights saved.")

    # =========================================================================
    # Save loss history CSV
    # =========================================================================
    LOSS_arr = np.array(LOSS)
    csv_path = os.path.join(path, 'loss_history.csv')
    np.savetxt(csv_path, LOSS_arr, delimiter=',',
               header='epoch,loss_pde,loss_data,loss_total,D,s', comments='')
    print(f"Loss CSV saved -> {csv_path}")

    # =========================================================================
    # Print and save training summary
    # =========================================================================
    D_final = torch.exp(log_D).item()
    s_final = torch.exp(log_s).item()

    # Per-generation conversions  (1 generation = 25 yr)
    GEN           = 25.0
    D_gen         = D_final * GEN            # deg²  generation⁻¹
    s_gen         = s_final * GEN            # dimensionless per generation
    # RMS 2-D radial dispersal distance in one generation: sqrt(4·D·T_gen)
    dispersal_deg = np.sqrt(4.0 * D_gen)    # degrees per generation
    dispersal_km  = dispersal_deg * 111.0   # km per generation (1°≈111 km)

    print("\n" + "=" * 60)
    print("  POST-TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Training time       : {elapsed:.2f} s")
    print(f"  Device              : {device}")
    print(f"  Final PDE loss      : {LOSS_arr[-1, 1]:.4e}")
    print(f"  Final data loss     : {LOSS_arr[-1, 2]:.4e}")
    print(f"  Final total loss    : {LOSS_arr[-1, 3]:.4e}")
    print(f"  Inferred D          : {D_final:.6e}  deg² yr⁻¹")
    print(f"  Inferred s          : {s_final:.6e}  yr⁻¹")
    print("  --- Per generation (25 yr) ---")
    print(f"  D per generation    : {D_gen:.6e}  deg² gen⁻¹")
    print(f"  Dispersal distance  : {dispersal_deg:.4f}  deg gen⁻¹  ({dispersal_km:.1f} km gen⁻¹)")
    print(f"  s per generation    : {s_gen:.6e}  gen⁻¹")
    print("=" * 60 + "\n")

    log_path = os.path.join(path, 'training_summary.txt')
    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  POST-TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Training time       : {elapsed:.2f} s\n")
        f.write(f"  Device              : {device}\n")
        f.write(f"  Epochs              : {epochs}\n")
        f.write(f"  Batch mode          : {Flag_batch}\n")
        f.write(f"  Lambda_data         : {Lambda_data}\n")
        f.write(f"  Final PDE loss      : {LOSS_arr[-1, 1]:.4e}\n")
        f.write(f"  Final data loss     : {LOSS_arr[-1, 2]:.4e}\n")
        f.write(f"  Final total loss    : {LOSS_arr[-1, 3]:.4e}\n")
        f.write(f"  Inferred D          : {D_final:.6e}  deg2 yr-1\n")
        f.write(f"  Inferred s          : {s_final:.6e}  yr-1\n")
        f.write("  --- Per generation (25 yr) ---\n")
        f.write(f"  D per generation    : {D_gen:.6e}  deg2 gen-1\n")
        f.write(f"  Dispersal distance  : {dispersal_deg:.4f}  deg gen-1  ({dispersal_km:.1f} km gen-1)\n")
        f.write(f"  s per generation    : {s_gen:.6e}  gen-1\n")
        f.write("=" * 60 + "\n")
    print(f"Training summary saved -> {log_path}")

    # =========================================================================
    # Plots
    # =========================================================================

    # --- Loss history ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 1], label='PDE (KPP) residual')
    ax.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 2],
                label=f'Data loss  (λ={Lambda_data})')
    ax.semilogy(LOSS_arr[:, 0], LOSS_arr[:, 3],
                'k--', lw=1, label='Total loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss History — LP Allele Spread PINN')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'loss_history.png'), dpi=150, bbox_inches='tight')
    print("Loss figure saved.")

    # --- Evolution of inferred D and s ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(LOSS_arr[:, 0], LOSS_arr[:, 4])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('D  (deg² yr⁻¹)')
    ax1.set_title('Inferred diffusion rate D')
    ax1.grid(True, alpha=0.3)

    ax2.plot(LOSS_arr[:, 0], LOSS_arr[:, 5])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('s  (yr⁻¹)')
    ax2.set_title('Inferred selection coefficient s')
    ax2.grid(True, alpha=0.3)

    fig2.suptitle(f'Final: D = {D_final:.4e}  |  s = {s_final:.4e}', fontsize=11)
    fig2.tight_layout()
    fig2.savefig(os.path.join(path, 'inferred_params.png'),
                 dpi=150, bbox_inches='tight')
    print("Parameter evolution figure saved.")

    # --- Predicted frequency map at present day (τ = τ_max) ---
    lat_grid  = np.linspace(lat_min_data,  lat_max_data,  80)
    lon_grid  = np.linspace(long_min_data, long_max_data, 80)
    LAT, LON  = np.meshgrid(lat_grid, lon_grid)

    tau_present = tau_max_col  # τ = τ_max_col ~ present-day

    with torch.no_grad():
        lat_t = torch.FloatTensor(
            (LAT.ravel() / lat_scale).reshape(-1, 1)).to(device)
        lon_t = torch.FloatTensor(
            (LON.ravel() / long_scale).reshape(-1, 1)).to(device)
        tau_t = torch.FloatTensor(
            np.full((LAT.size, 1), tau_present / tau_scale,
                    dtype=np.float32)).to(device)
        net_in  = torch.cat((lat_t, lon_t, tau_t), dim=1)
        p_map   = net_p(net_in).cpu().numpy().reshape(LAT.shape)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    cm = ax3.contourf(LON, LAT, p_map, levels=20, cmap='viridis',
                      vmin=0, vmax=1)
    fig3.colorbar(cm, ax=ax3, label='LP allele frequency  p')

    # Overlay observations (scatter coloured by observed frequency).
    # p_d contains raw counts {0,1,2}; divide by 2 to map onto [0,1] for display.
    p_obs = p_d.ravel() / 2.0
    sc = ax3.scatter(long_d.ravel() * long_scale,
                     lat_d.ravel()  * lat_scale,
                     c=p_obs, cmap='Reds', vmin=0, vmax=1,
                     edgecolors='k', linewidths=0.5, s=40, zorder=5,
                     label='Observations (individual allele freq)')
    fig3.colorbar(sc, ax=ax3, label='Individual allele frequency  (count / 2)')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Predicted LP allele frequency  (present day,  τ = τ_max)')
    ax3.legend(loc='upper left')
    fig3.tight_layout()
    fig3.savefig(os.path.join(path, 'p_map_present.png'),
                 dpi=150, bbox_inches='tight')
    print("Frequency map saved.")

    plt.show()

    return net_p, log_D, log_s, LOSS_arr


# =============================================================================
# Main configuration
# =============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# ---- Flags ------------------------------------------------------------------
Flag_batch    = True     # mini-batch training (recommended)
Lambda_data   = 20.0     # weight for data loss  (sparse → higher weight)

# ---- Training hyper-parameters ----------------------------------------------
batchsize     = 512
learning_rate = 5e-4
epochs        = 3000
Flag_pretrain = False    # load pre-trained weights if True

Flag_schedule = True
if Flag_schedule:
    step_epoch  = 1000
    decay_rate  = 0.2

# ---- Results directory ------------------------------------------------------
path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "Results_KPP") + os.sep

# =============================================================================
# Load and preprocess data
# =============================================================================
data_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(data_dir, 'cleaned_adna_pinn.csv')

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} records from {csv_path}")

# Remove rows with clearly erroneous coordinates
# (some entries appear to be encoded integers rather than decimal degrees)
df = df[(df['lat'].abs()  <= 90) &
        (df['long'].abs() <= 180)]
df = df.dropna(subset=['lat', 'long', 'mean_date', 'LP_allele_count'])
df = df.reset_index(drop=True)
print(f"  After coordinate filtering: {len(df)} records")

# Keep raw allele counts {0, 1, 2} as the training target.
# The Binomial NLL loss treats each individual as a Binomial(n=2, p) draw,
# so p is the latent population frequency the network infers — no division by 2.
# lp_freq (= count/2) is retained only for scatter-plot display purposes.
df['lp_freq'] = df['LP_allele_count'] / 2.0

# Forward time τ = T_max − mean_date_BP
# τ = 0  at oldest sample  →  allele has not yet spread
# τ = τ_max at most recent sample  →  present day
T_max_data   = df['mean_date'].max()
df['tau']    = T_max_data - df['mean_date']

# Domain bounds (from data, used for collocation sampling and visualisation)
lat_min_data  = df['lat'].min()
lat_max_data  = df['lat'].max()
long_min_data = df['long'].min()
long_max_data = df['long'].max()
tau_min_data  = df['tau'].min()    # = 0
tau_max_data  = df['tau'].max()    # = T_max_data − min(mean_date)

# Physical scaling factors (non-dimensionalise inputs to ~[0, 1])
lat_scale  = lat_max_data  - lat_min_data
long_scale = long_max_data - long_min_data
tau_scale  = tau_max_data  - tau_min_data

lat_scale  = lat_scale  if lat_scale  != 0 else 1.0
long_scale = long_scale if long_scale != 0 else 1.0
tau_scale  = tau_scale  if tau_scale  != 0 else 1.0

print(f"\n  Lat  range : [{lat_min_data:.2f}, {lat_max_data:.2f}]   "
      f"scale = {lat_scale:.2f}°")
print(f"  Lon  range : [{long_min_data:.2f}, {long_max_data:.2f}]   "
      f"scale = {long_scale:.2f}°")
print(f"  τ    range : [{tau_min_data:.1f}, {tau_max_data:.1f}] yr   "
      f"scale = {tau_scale:.1f} yr\n")

# Scaled observation arrays.
# p_d holds the raw allele counts {0, 1, 2} — passed directly to the
# Binomial NLL loss.  Coordinates are non-dimensionalised to ~[0, 1].
lat_d  = (df['lat'].values  / lat_scale).reshape(-1, 1).astype(np.float32)
long_d = (df['long'].values / long_scale).reshape(-1, 1).astype(np.float32)
tau_d  = (df['tau'].values  / tau_scale).reshape(-1, 1).astype(np.float32)
p_d    =  df['LP_allele_count'].values.reshape(-1, 1).astype(np.float32)   # raw counts {0,1,2}

# =============================================================================
# Collocation points: uniformly random over geographic × temporal domain
# =============================================================================
n_col = 20000
rng   = np.random.default_rng(42)

lat_col  = (rng.uniform(lat_min_data,  lat_max_data,  (n_col, 1))
            / lat_scale).astype(np.float32)
long_col = (rng.uniform(long_min_data, long_max_data, (n_col, 1))
            / long_scale).astype(np.float32)
tau_col  = (rng.uniform(tau_min_data,  tau_max_data,  (n_col, 1))
            / tau_scale).astype(np.float32)

tau_max_col = tau_max_data   # unscaled τ at present, used for visualisation

# =============================================================================
# Configuration summary
# =============================================================================
print("=" * 60)
print("  PROBLEM SETUP")
print("=" * 60)
print(f"  PDE          : Fisher-KPP  ∂p/∂τ = D·∇²p + s·p·(1−p)")
print(f"  Inferred     : D  (diffusion, deg² yr⁻¹),  s  (selection, yr⁻¹)")
print(f"  Data points  : {len(df)}")
print(f"  Colloc. pts  : {n_col}")
print("=" * 60)
print("  TRAINING CONFIG")
print("=" * 60)
print(f"  Device       : {device}")
print(f"  Epochs       : {epochs}")
print(f"  Batch mode   : {Flag_batch}  (batch size: {batchsize})")
print(f"  Lambda_data  : {Lambda_data}")
if Flag_schedule:
    print(f"  LR schedule  : StepLR  step={step_epoch}, gamma={decay_rate}")
else:
    print("  LR schedule  : None")
print("=" * 60 + "\n")

# =============================================================================
# Optionally load pre-trained weights
# =============================================================================
# (handled inside lp_train — set Flag_pretrain above if needed)

# =============================================================================
# Run training
# =============================================================================
net_p, log_D, log_s, LOSS_history = lp_train(
    device,
    lat_col, long_col, tau_col,
    lat_d, long_d, tau_d, p_d,
    batchsize, learning_rate, epochs, path,
    Flag_batch, Lambda_data,
    lat_scale, long_scale, tau_scale)
