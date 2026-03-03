import pandas as pd
import os

# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'glad_adna_15-8-22.xlsx')
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Select the minimal columns needed for the neural network
cols_to_keep = ['lat', 'long', 'mean_date', 'rs4988235_most_likely_genotype']
df_minimal = df[cols_to_keep].copy()

# Drop any rows missing spatial, temporal, or genotype data
df_minimal = df_minimal.dropna()

# Fix lat/lon values that were stored without decimal point (e.g. 51793 -> 51.793, -2425 -> -2.425)
# Valid range: lat in [-90, 90], lon in [-180, 180]; anything outside is scaled by 1/1000
df_minimal['lat'] = df_minimal['lat'].apply(lambda x: x / 1000 if abs(x) > 90 else x)
df_minimal['long'] = df_minimal['long'].apply(lambda x: x / 1000 if abs(x) > 180 else x)

# Convert the string genotype into the number of effective alleles ('A')
def count_lp_alleles(genotype):
    # Ensure uppercase string format
    genotype = str(genotype).upper()
    # Count how many 'A' alleles are present (0, 1, or 2)
    return genotype.count('A')

# Apply the function to create our target numeric column
df_minimal['LP_allele_count'] = df_minimal['rs4988235_most_likely_genotype'].apply(count_lp_alleles)

# Drop the old categorical column to keep the dataframe purely numerical
df_minimal = df_minimal.drop(columns=['rs4988235_most_likely_genotype'])

# Save the resulting minimal dataframe for the PINN
output_path = os.path.join(script_dir, 'cleaned_adna_pinn.csv')
df_minimal.to_csv(output_path, index=False)

print(df_minimal.head())