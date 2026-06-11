import requests
import pandas as pd
from io import StringIO

url = "https://www.bv-brc.org/api/genome_amr/?eq(taxon_id,1280)&eq(evidence,Laboratory%20Method)&limit(50000)"
headers = {"Accept": "text/csv"}

response = requests.get(url, headers=headers)
df = pd.read_csv(StringIO(response.text))

# Filter to methicillin, drop Intermediate (keep binary classification clean)
methicillin_df = df[df['Antibiotic'] == 'methicillin'].copy()
methicillin_df = methicillin_df[methicillin_df['Resistant Phenotype'].isin(['Resistant', 'Susceptible'])]

print("Shape:", methicillin_df.shape)
print("\nClass balance:")
print(methicillin_df['Resistant Phenotype'].value_counts())
print("\nSample genome IDs:")
print(methicillin_df['Genome ID'].head(10).tolist())

# Save it
methicillin_df.to_csv("methicillin_labels.csv", index=False)
print("\nSaved methicillin_labels.csv")