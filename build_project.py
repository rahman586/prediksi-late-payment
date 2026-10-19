import os

print("Membuat struktur folder...")
folders = ['data', 'notebook', 'hasil']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

file_contents = {
    ".gitignore": r"""venv/
__pycache__/
.ipynb_checkpoints/
*.pyc""",
    
    "requirements.txt": r"""pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
jupyter""",
    
    "README.md": r"""# Prediksi Late Payment - PT Maju Bersama

## Tujuan
Mendeteksi invoice yang berpotensi telat bayar (>30 hari setelah due date) menggunakan dataset sintetik (120 customer, 24 produk, ~4.000 invoice).

## Struktur Proyek
- `data/`: Berisi dataset mentah dan sintetik.
- `notebook/`: Skrip Python untuk ekstraksi, pemrosesan, dan pemodelan.
- `hasil/`: Output data agregat, matriks model, dan visualisasi grafis.

## Metodologi
- **Kerangka Kerja**: CRISP-DM.
- **Validasi Model**: Time-based split (menghindari data masa depan bocor ke masa lalu).
- **Penanganan Imbalance**: Pendekatan hibrida SMOTE dan algoritma class weighting.
- **Pencegahan Leakage**: Fitur historis hanya dihitung dari data sebelum invoice diterbitkan.""",

    "notebook/01_generate_data.py": r"""import numpy as np
import pandas as pd
from datetime import timedelta

np.random.seed(42)

N_CUST, N_PROD, N_INV = 120, 24, 4000
START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2026-06-30")

customer = pd.DataFrame({
    "customer_id": [f"C{i:03d}" for i in range(1, N_CUST+1)],
    "nama": [f"Customer {i}" for i in range(1, N_CUST+1)],
    "segmen": np.random.choice(["Retail", "Grosir", "Korporat"], N_CUST, p=[0.6, 0.3, 0.1]),
    "kota": np.random.choice(["Bandung", "Jakarta", "Surabaya", "Medan"], N_CUST),
})

product = pd.DataFrame({
    "product_id": [f"P{i:02d}" for i in range(1, N_PROD+1)],
    "nama_produk": [f"Produk {i}" for i in range(1, N_PROD+1)],
    "harga_beli": np.random.randint(20_000, 200_000, N_PROD),
    "harga_jual": np.random.randint(25_000, 300_000, N_PROD),
})
product["harga_jual"] = np.maximum(product["harga_jual"], product["harga_beli"] * 1.15)

tgl_invoice = pd.to_datetime(np.random.randint(START.value, END.value, N_INV))
invoice = pd.DataFrame({
    "invoice_id": [f"INV{i:05d}" for i in range(1, N_INV+1)],
    "customer_id": np.random.choice(customer["customer_id"], N_INV),
    "tanggal_invoice": tgl_invoice,
})
invoice["due_date"] = invoice["tanggal_invoice"] + pd.to_timedelta(np.random.choice([14, 30, 45], N_INV), unit="D")

lines = []
for inv in invoice.itertuples():
    for _ in range(np.random.randint(1, 9)):
        p = product.sample(1).iloc[0]
        lines.append({
            "invoice_id": inv.invoice_id,
            "product_id": p.product_id,
            "qty": np.random.randint(1, 10),
            "harga_satuan": p.harga_jual,
            "harga_beli": p.harga_beli,
        })
invoice_line = pd.DataFrame(lines)

late = np.random.random(N_INV) < 0.13
invoice["late_payment"] = late.astype(int)
invoice["tanggal_bayar"] = invoice.apply(
    lambda r: r.due_date + timedelta(days=np.random.randint(1, 90)) if r.late_payment
    else r.due_date - timedelta(days=np.random.randint(0, 10)), axis=1
)

customer.to_csv("data/customer.csv", index=False)
product.to_csv("data/product.csv", index=False)
invoice.to_csv("data/invoice.csv", index=False)
invoice_line.to_csv("data/invoice_line.csv", index=False)
print("Data sintetik berhasil dibuat!")""",

    "notebook/02_preprocessing.py": r"""import pandas as pd

invoice = pd.read_csv("data/invoice.csv", parse_dates=["tanggal_invoice", "due_date", "tanggal_bayar"])
customer = pd.read_csv("data/customer.csv")
product = pd.read_csv("data/product.csv")
lines = pd.read_csv("data/invoice_line.csv")

invoice = invoice.drop_duplicates("invoice_id")
invoice = invoice.drop(invoice[invoice.tanggal_bayar < invoice.tanggal_invoice].index)
customer["segmen"] = customer["segmen"].str.title()

lines = lines.merge(product[["product_id","harga_beli"]], on="product_id", suffixes=("","_p"))
lines["revenue"] = lines["qty"] * lines["harga_satuan"]
lines["gross_margin"] = lines["qty"] * (lines["harga_satuan"] - lines["harga_beli_p"])

harian = lines.merge(invoice[["invoice_id", "tanggal_invoice", "late_payment", "customer_id"]], on="invoice_id")
monthly = harian.set_index("tanggal_invoice").resample("ME")[["revenue", "gross_margin"]].sum()
monthly["late_rate"] = harian.set_index("tanggal_invoice").resample("ME")["late_payment"].mean()

monthly.to_csv("hasil/ringkasan_bulanan.csv")
print("Preprocessing selesai!")""",

    "notebook/03_features.py": r"""import pandas as pd

invoice = pd.read_csv("data/invoice.csv", parse_dates=["tanggal_invoice", "due_date", "tanggal_bayar"])

invoice["tenor_hari"] = (invoice["due_date"] - invoice["tanggal_invoice"]).dt.days
invoice["hari_ke"] = invoice["tanggal_invoice"].dt.day
invoice["bulan"] = invoice["tanggal_invoice"].dt.month
invoice["dow"] = invoice["tanggal_invoice"].dt.dayofweek

inv_sorted = invoice.sort_values("tanggal_invoice")
inv_sorted["hist_late_rate"] = inv_sorted.groupby("customer_id")["late_payment"].transform(lambda s: s.shift().expanding().mean())
inv_sorted["hist_n_invoice"] = inv_sorted.groupby("customer_id").cumcount()

df = inv_sorted.dropna(subset=["hist_late_rate"])
cols = ["tenor_hari", "hari_ke", "bulan", "dow", "hist_late_rate", "hist_n_invoice", "late_payment", "invoice_id", "tanggal_invoice"]
df[cols].to_csv("hasil/dataset_model.csv", index=False)
print("Fitur berhasil direkayasa!")""",

    "notebook/04_modeling.py": r"""import pandas as pd
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
hasil_df.to_csv("hasil/perbandingan_model.csv", index=False)""",

    "notebook/05_visualisasi.py": r"""import pandas as pd
import matplotlib.pyplot as plt

hasil = pd.read_csv("hasil/perbandingan_model.csv")
monthly = pd.read_csv("hasil/ringkasan_bulanan.csv", parse_dates=[0])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(monthly.iloc[:,0], monthly["revenue"], label="Revenue")
ax[0].plot(monthly.iloc[:,0], monthly["gross_margin"], label="Gross Margin")
ax[0].set_title("Revenue vs Gross Margin per Bulan"); ax[0].legend()
ax[1].barh(hasil["model"], hasil["recall_late"], color='skyblue')
ax[1].set_title("Recall Kelas Late Payment"); ax[1].set_xlabel("Recall Score")

plt.tight_layout()
plt.savefig("hasil/grafik_analisis.png", dpi=150)
print("\nVisualisasi selesai dan disimpan di hasil/grafik_analisis.png!")"""
}

for path, content in file_contents.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

print("\n✅ BERHASIL! Semua file dan struktur proyek siap.")