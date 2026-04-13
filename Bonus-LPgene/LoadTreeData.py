import tskit
import pyslim
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the Tree Sequence
ts = tskit.load("Bonus-LPgene/fisher_kpp_spatial.trees")

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
grid_size = 20 
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

    # 5. Bin the coordinates into our 20x20 grid
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

plt.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="Lactose Persistence Frequency")
plt.suptitle("Fisher-KPP Diffusion Wave of the LP Gene", fontsize=16)

# save the figure
plt.savefig("Bonus-LPgene/fisher_kpp_wave.png", dpi=300)

plt.show()

