import pandas as pd
import re
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier


dataset = pd.read_csv("../data/processed_dataset.csv", index_col=0)

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

y = y.replace({"Resistant" : 1, "Susceptible": 0}).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify= y
)
# Remove [, ], and < from column names
X_train.columns = [re.sub(r'[\[\]<]', '_', str(col)) for col in X_train.columns]
X_test.columns = [re.sub(r'[\[\]<]', '_', str(col)) for col in X_test.columns]

logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

xgboost_model = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=20,
    random_state=0
)

xgboost_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate
y_pred = xgboost_model.predict(X_test)
y_pred_proba = xgboost_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, y_pred_proba),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "num_genomes" : len(y)
}
print(metrics)

# SHAP explainer
explainer = shap.TreeExplainer(xgboost_model)

# Get list of features
feature_list = X_train.columns.tolist()

# Save everything Streamlit will need
joblib.dump(xgboost_model, '../Models/xgboost_model.pkl')
joblib.dump(feature_list, '../Models/feature_columns.pkl')
joblib.dump(explainer, '../Models/shap_explainer.pkl')
joblib.dump(metrics, '../Models/metrics.pkl')
