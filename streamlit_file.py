import streamlit as st
from pyarrow.lib import output_stream
from streamlit import button
import import_ipynb
from AntibioticResistancePredictor.ipynb import

st.title("S. Aureus Bacterial Resistance Predictor")

# To do:
# Make a multi select list of genes, with search function
# Run prediction button
# Show resistant or susceptible as well as the probability output
# list number of genomes, model AUC and Accuracy
# provide SHAP waterfall of the output
