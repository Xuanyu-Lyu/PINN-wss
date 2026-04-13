import pandas as pd
import tskit
import pyslim
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the Tree Sequence
ts = tskit.load("Bonus-LPgene/fisher_kpp_spatial_betterReso_25k.trees")

# In tskit, time is measured in "generations ago". 
# SLiM ran for 2000 generations, so Gen 2000 = Time 0.
max_gen = 2000

# 2. Grab our single injected mutation
try:
    variant = next(ts.variants())
    genotypes = variant.genotypes 
except StopIteration:
    print("No mutations found! The mutation may have died out before sweeping.")
    exit()

# --- THE FIX: Create a Node-to-Sample Translator ---
# tskit's genotype array is indexed by sample ID, not absolute node ID.
# We build a quick translation array to map absolute IDs to sample IDs.
node_to_sample = np.full(ts.num_nodes, -1, dtype=int)
node_to_sample[ts.samples()] = np.arange(ts.num_samples)
# ---------------------------------------------------

# 3. Setup the Spatial Grid 
grid_size = 50 
bins = np.linspace(0, 100, grid_size + 1)

gens_to_plot = [200, 600, 1000, 1400]
fig, axes = plt.subplots(1, len(gens_to_plot), figsize=(16, 4))

# 4. Loop through time and calculate Allele Frequencies
for i, gen in enumerate(gens_to_plot):
    time_ago = max_gen - gen 
    alive_inds = pyslim.individuals_alive_at(ts, time_ago)
    
    xs, ys, mutant_counts = [], [], []
    
    for ind_id in alive_inds:
        ind = ts.individual(ind_id)
        x, y, z = ind.location
        xs.append(x)
        ys.append(y)
        
        # Get absolute node IDs
        node_1, node_2 = ind.nodes
        
        # Translate absolute Node IDs to Sample IDs
        sample_1 = node_to_sample[node_1]
        sample_2 = node_to_sample[node_2]
        
        # Count copies of the mutation using the correct Sample ID
        copies = genotypes[sample_1] + genotypes[sample_2]
        mutant_counts.append(copies)

    xs = np.array(xs)
    ys = np.array(ys)
    mutant_counts = np.array(mutant_counts)

    # 5. Bin the coordinates into our 50x50 grid
    mut_grid, _, _ = np.histogram2d(xs, ys, bins=bins, weights=mutant_counts)
    total_inds, _, _ = np.histogram2d(xs, ys, bins=bins)
    
    freq_grid = np.divide(mut_grid, 2 * total_inds, out=np.zeros_like(mut_grid), where=total_inds!=0)
    
    # 6. Plot the wave
    ax = axes[i]
    im = ax.imshow(freq_grid.T, origin='lower', extent=[0, 100, 0, 100], 
                   cmap='magma', vmin=0, vmax=1)
    ax.set_title(f"Generation {gen}")
    ax.set_xlabel("X Coordinate")
    if i == 0: ax.set_ylabel("Y Coordinate")
# --- Compute average allele frequency per grid cell across time ---
bin_centers = 0.5 * (bins[:-1] + bins[1:])  # center of each 2-unit cell

# ── Temporal sampling mode ────────────────────────────────────────────────────
# "uniform"  : one time point every `uniform_step` generations (regular grid)
# "random"   : stratified random — gen 1–2000 split into 4 equal periods,
#              `n_per_period` points drawn uniformly at random from each period
SAMPLE_MODE   = "random"   # "uniform" or "random"
uniform_step  = 10         # used only when SAMPLE_MODE == "uniform"
n_per_period  = 20          # draws per period when SAMPLE_MODE == "random"
rng           = np.random.default_rng(11)

if SAMPLE_MODE == "uniform":
    gens_all = np.arange(uniform_step, max_gen + 1, uniform_step)
else:  # stratified random across 4 periods
    # SLiM retains individuals at multiples of 10 (gen 10..2000)
    available = np.arange(10, max_gen + 1, 10)     # 200 retained snapshots
    period_size = len(available) // 4              # 50 snapshots per period
    gens_all = np.concatenate([
        rng.choice(available[p * period_size:(p + 1) * period_size],
                   size=min(n_per_period, period_size), replace=False)
        for p in range(4)
    ])
    gens_all = np.sort(gens_all)

print(f"Sampling {len(gens_all)} time points ({SAMPLE_MODE} mode): {gens_all}")
rows = []

for gen in gens_all:
    time_ago   = max_gen - gen
    alive_inds = pyslim.individuals_alive_at(ts, time_ago)

    xs, ys, mutant_counts = [], [], []
    for ind_id in alive_inds:
        ind = ts.individual(ind_id)
        x, y, _ = ind.location
        node_1, node_2 = ind.nodes
        s1, s2 = node_to_sample[node_1], node_to_sample[node_2]
        if s1 < 0 or s2 < 0:
            continue
        xs.append(x)
        ys.append(y)
        mutant_counts.append(int(genotypes[s1]) + int(genotypes[s2]))

    xs = np.array(xs)
    ys = np.array(ys)
    mutant_counts = np.array(mutant_counts)

    # Bin every individual into its grid cell
    ix_all = np.clip(np.digitize(xs, bins) - 1, 0, grid_size - 1)
    iy_all = np.clip(np.digitize(ys, bins) - 1, 0, grid_size - 1)

    for ix in range(grid_size):
        for iy in range(grid_size):
            mask = (ix_all == ix) & (iy_all == iy)
            cell_counts = mutant_counts[mask]
            if len(cell_counts) == 0:
                continue

            # Randomly drop individuals: triangular(left=0.2, mode=0.4, right=1.0)
            # Equivalent to sum of two scaled uniforms; right-skewed so small
            # drops (≈0.2) are most common and large drops (up to 1.0) are rare.
            drop_frac   = rng.triangular(0.2, 0.4, 1.0)
            keep_n      = max(1, int(round(len(cell_counts) * (1 - drop_frac))))
            kept         = rng.choice(cell_counts, size=keep_n, replace=False)

            rows.append({
                "x":          bin_centers[ix],
                "y":          bin_centers[iy],
                "generation": gen,
                "n_inds":     keep_n,
                "freq":       kept.sum() / (2 * keep_n),
            })

df_freq = pd.DataFrame(rows)
df_freq.to_csv("Bonus-LPgene/cell_freq_data_random.csv", index=False)
print(f"Saved {len(df_freq)} rows → cell_freq_data_random.csv")

# --- Convert to WSINDy .mat format ---
# U_data: 3D array [n_x, n_y, n_t], NaN where no observation was sampled
from scipy.io import savemat

x_grid = np.array(sorted(df_freq["x"].unique()), dtype=float)       # 20 cell centers
y_grid = np.array(sorted(df_freq["y"].unique()), dtype=float)        # 20 cell centers
t_grid = np.array(sorted(df_freq["generation"].unique()), dtype=float)

U_data = np.full((len(x_grid), len(y_grid), len(t_grid)), np.nan)

x_idx = {v: i for i, v in enumerate(x_grid)}
y_idx = {v: i for i, v in enumerate(y_grid)}
t_idx = {v: i for i, v in enumerate(t_grid)}

for _, row in df_freq.iterrows():
    U_data[x_idx[row["x"]], y_idx[row["y"]], t_idx[row["generation"]]] = row["freq"]

# WSINDy expects xs_obs as a 1x3 object array and U_obs as a 1x1 object array
xs_obs = np.empty((1, 3), dtype=object)
xs_obs[0, 0] = x_grid.reshape(-1, 1)
xs_obs[0, 1] = y_grid.reshape(-1, 1)
xs_obs[0, 2] = t_grid.reshape(-1, 1)

U_data_imputed = np.where(np.isnan(U_data), 0.0, U_data)

U_obs = np.empty((1, 1), dtype=object)
U_obs[0, 0] = U_data_imputed

savemat(
    "/Users/xuly4739/Library/CloudStorage/OneDrive-UCB-O365/Documents/coding/MatlabProject/datasets/lactose_data_25k.mat",
    {"xs_obs": xs_obs, "U_obs": U_obs},
)
