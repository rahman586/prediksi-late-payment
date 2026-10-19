# Prediksi Late Payment - PT Maju Bersama

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
- **Pencegahan Leakage**: Fitur historis hanya dihitung dari data sebelum invoice diterbitkan.
