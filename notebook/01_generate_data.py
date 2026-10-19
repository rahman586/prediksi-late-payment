import numpy as np
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

# PERBAIKAN DI SINI: Menggunakan selisih hari agar aman dari limit int32 Windows
delta_days = (END - START).days
tgl_invoice = START + pd.to_timedelta(np.random.randint(0, delta_days, N_INV), unit="D")

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
print("Data sintetik berhasil dibuat!")