import pandas as pd
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
print("\nVisualisasi selesai dan disimpan di hasil/grafik_analisis.png!")
