# BBCA Stock Analysis: Financial Data Preprocessing & Stationarity Testing

## Overview
Analisis preprocessing data saham BBCA (Bank Central Asia) periode 
Januari 2022 – Januari 2024 menggunakan Python. Proyek ini mencakup 
eksplorasi distribusi returns, deteksi outlier berbasis makroekonomi, 
dan uji stasioneritas sebagai fondasi time series modeling.

## Key Findings
- **Rata-rata volatilitas tahunan:** 19.95%
- **Skewness:** -0.145 (sedikit miring kiri — kerugian ekstrem 
  lebih mungkin dari keuntungan ekstrem)
- **Kurtosis:** 2.09 (platykurtic — distribusi lebih tipis dari normal, 
  kejadian ekstrem relatif jarang)
- **Outlier terdeteksi:** 3 tanggal, semua berkorelasi dengan kejadian makro:
  - Mei 2022: krisis energi Eropa + Fed rate hike 50bps
  - Desember 2023: sinyal Fed pivot + Red Sea crisis
- **ADF Test:** Harga Close tidak stasioner (p=0.518), 
  Log Returns stasioner (p=0.000) ✓
- **Volatility persistence (α+β):** 0.9277 — volatilitas sangat persisten
- **ARIMA baseline:** tidak signifikan, mengkonfirmasi log returns mendekati white noise
- **Volatility forecast 5 hari:** rata-rata 1.17% per hari 
  (di bawah baseline 1.28% — kondisi diprediksi lebih tenang)
- **Mean reversion terdeteksi:** volatilitas naik bertahap 
  1.15% → 1.19% mencerminkan persistence 0.9277
- **Dollar bars:** 355 bars dari 485 time bars — setiap bar 
  merepresentasikan 500 miliar rupiah diperdagangkan
- **Triple barrier labels:** Buy 49.3% / Sell 39.7% / Hold 10.7%
  — class imbalance terdeteksi, perlu penanganan sebelum ML training
- **BBCA karakteristik:** saham aktif dengan bias upward, 
  jarang stagnan lebih dari 10 hari
- **Feature terpenting:** ma_dist (0.253) — jarak harga dari MA20 
  adalah sinyal paling informatif untuk prediksi arah BBCA
- **Feature terlemah:** log_return (0.074) — konfirmasi final 
  bahwa returns BBCA mendekati white noise
- **Model accuracy:** 20–51% per fold — tidak konsisten, 
  belum reliable untuk trading nyata
- **Volume sebagai sinyal:** vol_ratio (0.202) — volume relatif 
  terhadap rata-rata historis adalah sinyal kedua terkuat

## Pipeline
1. Data acquisition — yfinance API
2. Preprocessing — missing values, duplicate check
3. Log returns computation
4. Distribusi analysis — skewness, kurtosis
5. Outlier detection — z-score method
6. Rolling volatility — 21-day annualized
7. Stationarity testing — Augmented Dickey-Fuller
8. ARIMA(1,0,1) baseline modeling
9. GARCH(1,1) volatility modeling & persistence analysis
10. Volatility forecasting — GARCH 5-day ahead forecast
11. Dollar bars — de Prado alternative data structure
12. Triple barrier method — proper ML labeling
13. Feature engineering — momentum, volatility, volume ratio, mean reversion
14. Random Forest classifier — TimeSeriesSplit 5-fold cross validation
15. Feature importance analysis

## Tech Stack
- Python 3
- pandas, numpy, matplotlib
- statsmodels
- yfinance

## Key Insight
Semua model time series (ARIMA, GARCH) harus dibangun di atas 
log returns, bukan harga — karena harga tidak stasioner dan 
menghasilkan spurious regression. ADF test mengonfirmasi 
log returns BBCA stasioner dengan p-value < 0.001.
De Prado (2018): model ML yang dilatih di time bars biasa 
menghasilkan spurious results. Dollar bars + triple barrier 
labeling adalah fondasi yang benar untuk quant ML strategy.
Feature importance mengkonfirmasi temuan statistik sebelumnya:
log_return BBCA mendekati white noise (ADF, ACF/PACF, ARIMA) —
model sederhana tidak cukup untuk mengalahkan pasar. Improvement
membutuhkan fitur yang lebih sophisticated dan data yang lebih banyak.

## Next Steps
- ~~ACF/PACF analysis~~ ✓
- ~~ARIMA baseline modeling~~ ✓
- ~~GARCH volatility modeling~~ ✓
- ~~Volatility forecasting~~ ✓
- ~~Dollar bars (de Prado)~~ ✓
- ~~Triple barrier labeling~~ ✓
- ~~Feature engineering~~ ✓
- ~~Random Forest + feature importance~~ ✓
- SMOTE untuk handle class imbalance
- XGBoost sebagai model alternatif
- Integrasi GARCH volatility sebagai fitur ML
- Backtesting & Sharpe Ratio evaluation
- Ekspansi ke multi-saham BEI

## Model Performance Summary

| Model | Keterangan | Hasil |
|-------|------------|-------|
| ARIMA(1,0,1) | Baseline time series | Tidak signifikan (p>0.05) |
| GARCH(1,1) | Volatility modeling | Beta signifikan, persistence 0.9277 |
| Random Forest | ML classifier | Akurasi 20–51%, belum konsisten |

Ketiga model mengkonfirmasi: BBCA adalah saham efisien yang 
sulit diprediksi dengan fitur teknikal sederhana. 
Pendekatan yang lebih sophisticated diperlukan.

## Author
Aril Satrio Saputro  
Data Science Student | Universitas Airlangga  
www.linkedin.com/in/aril-saputro-530168323 | satrioaril34@gmail.com
