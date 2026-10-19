import pandas as pd

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
print("Fitur berhasil direkayasa!")
