import requests
import pandas as pd
from io import StringIO
from sklearn.preprocessing import MultiLabelBinarizer


# The URL to pull Staph aureus, make sure that the resistance was tested in wet lab
url = "https://www.bv-brc.org/api/genome_amr/?eq(taxon_id,1280)&eq(evidence,Laboratory%20Method)&limit(50000)"
headers = {"Accept": "text/csv"}

# API request
response = requests.get(url, headers=headers)
df = pd.read_csv(StringIO(response.text))

# Filter to methicillin, only get the Resistant or susceptible genomes
methicillin_df = df[df['Antibiotic'] == 'methicillin'].copy()
methicillin_df = methicillin_df[methicillin_df['Resistant Phenotype'].isin(['Resistant', 'Susceptible'])]

# Convert the genome IDs to a flat string that can be fed into an API call
genome_list = methicillin_df['Genome ID'].tolist()
genome_string = ",".join([str(x) for x in genome_list])

# Set up batches since there are a lot of genome IDs
all_batches = []
cursor = "*"

# Loop until all batches are completed
while True:
    # API request done in batches
    response = requests.post(
        "https://www.bv-brc.org/api/sp_gene/",
        headers={"Accept": "text/csv", "Content-Type": "application/rqlquery+x-www-form-urlencoded"},
        data=f'in(genome_id,({genome_string}))&eq(property,%22Antibiotic%20Resistance%22)&limit(25000)&cursor({cursor})'
    )

    # Convert the batch API call into a csv and then make into a DF. Append to final batch df
    batch_df = pd.read_csv(StringIO(response.text))
    all_batches.append(batch_df)

    # Save the next_cursor to know where to start for the next API call
    next_cursor = response.headers.get("X-Cursor-Mark")

    # Stop when cursor stops changing
    if next_cursor == cursor:
        break

    # update the Cursor and run the loop again
    cursor = next_cursor

genes_df = pd.concat(all_batches, ignore_index=True)

# Use Gene where available, fall back to Product when Gene is missing
genes_df['feature_name'] = genes_df['Gene'].fillna(genes_df['Product'])

genome_genes = genes_df.groupby('Genome ID')['feature_name'].apply(list)

mlb = MultiLabelBinarizer()
encoded_genes = mlb.fit_transform(genome_genes)

df_encoded_genes = pd.DataFrame(encoded_genes, index = genome_genes.index,columns=mlb.classes_)

labels_df = methicillin_df.set_index("Genome ID")

dataset = df_encoded_genes.join(labels_df["Resistant Phenotype"], how="right")

gene_cols = dataset.columns.drop("Resistant Phenotype")
dataset[gene_cols] = dataset[gene_cols].fillna(0)
dataset.to_csv("../data/processed_dataset.csv")