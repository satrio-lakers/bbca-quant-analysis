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

## Next Steps
- ~~ACF/PACF analysis~~ ✓
- ~~ARIMA baseline modeling~~ ✓  
- ~~GARCH volatility modeling~~ ✓
- ~~Volatility forecasting~~ ✓
- Financial data structures (de Prado approach)
- Triple barrier method & labeling
- ML model di atas fitur quant

## Author
Aril Satrio Saputro  
Data Science Student | Universitas Airlangga  
www.linkedin.com/in/aril-saputro-530168323 | satrioaril34@gmail.com
