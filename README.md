# Predicting Methicillin Resistance in *Staphylococcus aureus* from Genomic Data

## Overview

This project predicts whether a *Staphylococcus aureus* genome is resistant or susceptible to methicillin, using presence/absence of known antibiotic resistance genes as features. Methicillin resistance (MRSA) is one of the most clinically significant antimicrobial resistance problems worldwide, and is primarily driven by acquisition of the *mecA* gene and its regulatory elements.

The goal of this project was to build an interpretable, biologically grounded classifier and to use model explainability (SHAP) to confirm that the model's predictions are driven by known resistance mechanisms rather than spurious correlations.

## Data Source

All data was pulled programmatically from the [BV-BRC](https://www.bv-brc.org/) (Bacterial and Viral Bioinformatics Resource Center) API, the successor to PATRIC.

- **Labels**: `genome_amr` endpoint, filtered to *S. aureus* (taxon ID 1280), methicillin, lab-confirmed phenotypes only (`evidence = Laboratory Method`). This yielded 1,082 genomes: 574 Susceptible, 508 Resistant.
- **Features**: `sp_gene` endpoint, filtered to `property = Antibiotic Resistance`. Retrieved via POST requests with cursor-based pagination (86,803 gene records across 980 genomes).

## Feature Engineering

Each genome's resistance genes were encoded as a multi-hot matrix (one column per known gene/product, 1 = present, 0 = absent) using `MultiLabelBinarizer`. Genomes with no detected resistance genes (~100 genomes) were retained as all-zero feature rows rather than dropped, since the absence of detectable resistance genes is itself meaningful signal.

Where a specific gene name (`mecA`, `blaZ`, etc.) wasn't available, the gene's `Product` description was used as a fallback label. This decision turned out to matter: many *mecA*-driven records were annotated only by their product description ("Penicillin-binding protein PBP2a, methicillin resistance determinant MecA, transpeptidase"), not the gene symbol itself.

This produced a final feature matrix of 1,082 genomes × 279 gene/product features.

## Modeling

Two models were trained and compared:

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | 94.5% | 0.938 | 0.984 |
| XGBoost | 94.5% | 0.938 | 0.984 |

Both models produced **identical classifications** on the held-out test set (217 genomes), with only marginal differences in predicted probabilities. This convergence is itself a finding: it suggests the genomic signal for methicillin resistance is strong and close to linearly separable, driven predominantly by a small number of features rather than complex non-linear interactions; consistent with the known biology, where *mecA* presence is close to a deterministic switch for resistance.

Confusion matrix (both models):

```
              Predicted Susceptible   Predicted Resistant
Actual Susceptible      115                    0
Actual Resistant         12                   90
```

Notably, **zero false positives**
 the model never incorrectly flagged a susceptible genome as resistant. All 12 errors were false negatives (resistant genomes predicted susceptible), which is the more clinically concerning error type, since it could lead to under-treatment.

## Interpretability (SHAP)

SHAP analysis on the XGBoost model confirmed that the top predictive features align with established MRSA biology:

1. **PBP2a / mecA** (the canonical methicillin resistance determinant)
2. **MecR1** (the regulatory sensor-transducer controlling *mecA* expression)
3. **MecI** (the *mecA* repressor)

These three features — which together make up the *mec* regulatory cassette — dominate the model's decision-making, exactly as expected from known resistance mechanisms.

### A correctly classified example

Genome `1280.8162` was correctly predicted Susceptible. SHAP attribution showed this was driven almost entirely by the *absence* of PBP2a and MecR1 — a clean example of the model reasoning the way a microbiologist would.
<img width="1069" height="416" alt="Screenshot 2026-06-24 at 7 19 53 PM" src="https://github.com/user-attachments/assets/f8204a05-bbf8-4d76-92ca-b5b15ca648b5" />

### The most informative error

Genome `1280.2514` (strain *12673_8_61*) was lab-confirmed Resistant (via broth dilution/VITEK 2, gold-standard clinical methods) but predicted Susceptible by the model. SHAP showed this genome had **zero detected resistance genes of any kind** — including no PBP2a, MecR1, or MecI.
<img width="1065" height="425" alt="Screenshot 2026-06-24 at 7 20 51 PM" src="https://github.com/user-attachments/assets/b53ff073-7e8d-41a1-86a6-8112a87367df" />

This genome originates from a real clinical surveillance dataset: a 2019 study by Toleman et al. (*Eurosurveillance*) on prospective genomic surveillance of MRSA bloodstream infections in England. The resistance phenotype is well-established and trustworthy; the gap is on the genomic annotation side.

This points to a genuine limitation of gene-presence-based prediction: it can only detect resistance mechanisms that have already been catalogued. Possible explanations include an unannotated SCCmec variant, a point mutation in an existing gene rather than acquisition of a new one, or an assembly/annotation gap specific to this isolate. Rather than a model failure, this is the model correctly reporting "no known resistance mechanism detected".

## Key Takeaways

- A gene presence/absence feature set, combined with a simple model, can predict methicillin resistance with high accuracy (94.5%, AUC 0.984) because the underlying biology is strongly deterministic.
- Model agreement between a linear model and a tree ensemble is itself informative it indicates the signal is close to linearly separable rather than reliant on complex feature interactions.
- SHAP-based interpretability is essential for distinguishing genuine model limitations (missing biological signal) from modeling errors a distinction that matters most in a clinical context, where understanding *why* a model is wrong is as important as knowing *that* it's wrong.

## Tech Stack

- **Data acquisition**: Python, `requests` (BV-BRC REST API, cursor pagination)
- **Data processing**: `pandas`, `scikit-learn` (`MultiLabelBinarizer`)
- **Modeling**: `scikit-learn` (Logistic Regression), `xgboost`
- **Interpretability**: `shap`

## Future Work

- Extend to other antibiotics (vancomycin, ciprofloxacin) tested in the same dataset
- Incorporate k-mer-based sequence features as a reference-free alternative to gene annotation, to capture resistance mechanisms missed by curated gene panels
- Build an interactive dashboard (Streamlit) for exploring predictions and SHAP explanations on new genomes
