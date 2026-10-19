import pandas as pd

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
print("Preprocessing selesai!")
