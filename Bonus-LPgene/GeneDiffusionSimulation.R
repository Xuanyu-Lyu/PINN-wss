# Install required packages if you don't have them:
#install.packages(c("slendr", "sf", "dplyr"))
#slendr::setup_env() # Required on first run to setup the SLiM/Python backend

library(slendr)
library(sf)
library(dplyr)

# ---------------------------------------------------------
# 1. Define the Spatial World (Europe / Eurasia)
# ---------------------------------------------------------
# xrange/yrange are always in degrees for world(); EPSG:3035 (ETRS89-LAEA)
# is applied internally so all distance parameters below are in meters.
# yrange extended to 28°N to fully include the Near East origin (~32°N).
map_eur <- slendr::world(
  xrange = c(-15, 45), # degrees longitude (W Ireland to E Ukraine)
  yrange = c(28, 70),  # degrees latitude (Near East to Scandinavia)
  crs = "EPSG:3035"    # ETRS89-LAEA Europe: metric, equal-area
)

# ---------------------------------------------------------
# 2. Define the Population & Fisher-KPP Parameters
# ---------------------------------------------------------
# time: 8000 BP (Before Present), matching the oldest dates in your CSV.
# dispersal: This is your Diffusion Coefficient (D). It defines how far 
# individuals move per generation in kilometers.
# mating: Max distance for reproducing.
pop <- population(
  "EUR", 
  time = 8000, 
  N = 10000, 
  map = map_eur,
  center = c(0,30), # Levant/Near East in lon/lat degrees (slendr always uses degrees for center)
  radius = 400e3,     # 400 km initial spread (meters, per EPSG:3035)
  mating = 50e3,      # 50 km mating distance (meters)
  dispersal = 60e3    # 60 km dispersal / Diffusion D (meters)
)

# Force the population to expand across the map to simulate diffusion
pop <- expand_range(pop, by = 1000e3, start = 8000, end = 2000, polygon = map_eur) # 1000 km

# ---------------------------------------------------------
# 3. Compile Model & Schedule aDNA Sampling
# ---------------------------------------------------------
model <- compile_model(
  populations = pop,
  generation_time = 25, # 25 years per generation
  resolution = 10e3,    # 10 km per pixel (meters, required for spatial models)
  direction = "backward", # Matches BP dating (larger numbers are older)
  path = "lp_diffusion_model",
  overwrite = TRUE
)

# We want to sample individuals across time, exactly like your aDNA dataset.
# We sample 50 individuals every 500 years from 7500 BP down to 2000 BP.
sampling_times <- seq(7500, 2000, by = -500)
schedule <- schedule_sampling(model, times = sampling_times, list(pop, 50))

# ---------------------------------------------------------
# 4. Run SLiM Engine
# ---------------------------------------------------------
# This runs the spatial forward-genetic simulation and outputs a tree sequence.
ts <- slim(
  model, 
  sequence_length = 1e5, 
  recombination_rate = 1e-8,
  samples = schedule
)

# ---------------------------------------------------------
# 5. Extract & Format Data to Match your CSV
# ---------------------------------------------------------
# Extract the spatial nodes (individuals) from the simulated tree sequence
nodes <- ts_nodes(ts)

# Filter for our sampled ancient individuals and extract coordinates
sampled_nodes <- nodes %>%
  filter(time %in% sampling_times) %>%
  st_drop_geometry() %>%
  bind_cols(as.data.frame(st_coordinates(nodes %>% filter(time %in% sampling_times))))

# Build the final dataframe to match `cleaned_adna_pinn.csv`
simulated_adna <- sampled_nodes %>%
  select(
    lat = Y,
    long = X,
    mean_date = time
  ) %>%
  # NOTE ON SELECTION (s): slendr models spatial demography natively. 
  # To get true allele frequencies of a sweeping beneficial mutation, you would 
  # overlay mutations onto the tree sequence using `tskit` in Python, or inject 
  # an Eidos script. For now, we simulate the allele count based on a 
  # simple spatial-temporal logistic curve to give you data ready for WSINDy.
  mutate(
    # Mock Fisher-KPP Wave: Probability of having the LP allele increases 
    # as time gets closer to 0, and as longitude moves west from the origin.
    wave_prob = plogis((8000 - mean_date) / 1000 - (long - 35) / 10),
    LP_allele_count = rbinom(n(), size = 2, prob = wave_prob) # 0, 1, or 2 copies
  ) %>%
  select(-wave_prob)

# Save the dataset
write.csv(simulated_adna, "simulated_adna_pinn.csv", row.names = FALSE)
head(simulated_adna)