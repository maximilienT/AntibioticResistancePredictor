import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load in information app needs to run
model = joblib.load("models/xgboost_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
explainer = joblib.load("models/shap_explainer.pkl")
metrics = joblib.load("models/metrics.pkl")

# UI

st.title("S. Aureus Bacterial Resistance Predictor")

st.subheader("model Performance")

col1, col2, col3, col4 = st.columns(4)

# list number of genomes, model AUC and Accuracy
col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
col2.metric("F1 Score", f"{metrics['f1']:.1%}")
col3.metric("AUC", f"{metrics['auc']:.1%}")
col4.metric("Number of Genomes", f"{metrics['num_genomes']}")

st.subheader("Predict from gene profile")

# Make a multi select list of genes, with search function
selected_genes = st.multiselect("Select genes present in this genome:", feature_columns )

# Run prediction button
if st.button("Run Prediction"):

    # Create a feature dataframe to feed into the model
    row = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)
    for gene in selected_genes:
        if gene in row.columns:
            row[gene] = 1

    # Run prediction
    proba = model.predict_proba(row)[0][1]
    label = "Resistant" if proba > 0.5 else "Susceptible"

    # Show resistant or susceptible as well as the probability output
    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability of resistance:** {proba:.2%}")

    # provide SHAP waterfall of the output
    shap_values = explainer(row)
    st.subheader("Why the model made this prediction")
    shap.plots.waterfall(shap_values[0], show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()

