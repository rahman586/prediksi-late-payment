import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv("hasil/dataset_model.csv", parse_dates=["tanggal_invoice"])
X = df[["tenor_hari", "hari_ke", "bulan", "dow", "hist_late_rate", "hist_n_invoice"]]
y = df["late_payment"]

cutoff = df["tanggal_invoice"].quantile(0.8)
tr = df["tanggal_invoice"] <= cutoff
X_tr, X_te, y_tr, y_te = X[tr], X[~tr], y[tr], y[~tr]

X_tr_sm, y_tr_sm = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
}

hasil = []
for nama, model in models.items():
    model.fit(X_tr_sm, y_tr_sm)
    pred = model.predict(X_te)
    rep = classification_report(y_te, pred, output_dict=True, zero_division=0)
    hasil.append({
        "model": nama,
        "recall_late": rep["1"]["recall"],
        "precision_late": rep["1"]["precision"],
        "roc_auc": roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    })

hasil_df = pd.DataFrame(hasil).sort_values("roc_auc", ascending=False)
print("\n--- Hasil Model ---")
print(hasil_df.round(2))
hasil_df.to_csv("hasil/perbandingan_model.csv", index=False)
