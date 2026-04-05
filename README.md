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

## Next Steps
- ~~ACF/PACF analysis~~ ✓
- ~~ARIMA baseline modeling~~ ✓
- ~~GARCH volatility modeling~~ ✓
- ~~Volatility forecasting~~ ✓
- ~~Dollar bars (de Prado)~~ ✓
- ~~Triple barrier labeling~~ ✓
- Feature engineering untuk ML
- Random Forest / ML model training
- Backtesting & performance evaluation

## Author
Aril Satrio Saputro  
Data Science Student | Universitas Airlangga  
www.linkedin.com/in/aril-saputro-530168323 | satrioaril34@gmail.com
