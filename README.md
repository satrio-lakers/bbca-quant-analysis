# BBCA Quant Analysis: From Preprocessing to ML Strategy

## Overview
Analisis kuantitatif komprehensif saham BBCA (Bank Central Asia) 
periode Januari 2022 – Januari 2024. Proyek ini mengimplementasikan 
pipeline quant lengkap dari preprocessing hingga ML strategy, 
mengikuti pendekatan de Prado (2018) dalam Advances in Financial 
Machine Learning.

## Key Findings
- **Volatilitas tahunan:** 19.95% rata-rata
- **Skewness:** -0.145 — sedikit miring kiri, kerugian ekstrem 
  lebih mungkin dari keuntungan ekstrem
- **Kurtosis:** 2.09 — distribusi lebih tipis dari normal, 
  kejadian ekstrem relatif jarang
- **Outlier:** 3 tanggal terdeteksi, semua berkorelasi dengan 
  kejadian makro (Fed rate hike Mei 2022, Red Sea crisis Des 2023)
- **ADF Test:** Harga Close tidak stasioner (p=0.518), 
  Log Returns stasioner (p=0.000) ✓
- **Volatility persistence (α+β):** 0.9277 — volatilitas sangat 
  persisten, sekali bergejolak bertahan lama
- **Volatility forecast 5 hari:** rata-rata 1.17% per hari 
  (di bawah baseline 1.28% — kondisi diprediksi lebih tenang)
- **Dollar bars:** 355 bars dari 485 time bars
- **Triple barrier labels:** Buy 49.3% / Sell 39.7% / Hold 10.7%
- **Feature terpenting:** ma_dist (0.253) — mean reversion signal
- **Feature terlemah:** log_return (0.074) — konfirmasi white noise
- **Model accuracy:** 40-42% — tidak konsisten lintas fold

## Pipeline
1. Data acquisition — yfinance API
2. Preprocessing — missing values, duplicate check
3. Log returns computation
4. Distribusi analysis — skewness, kurtosis
5. Outlier detection — z-score + konteks makroekonomi
6. Rolling volatility — 21-day annualized
7. Stationarity testing — Augmented Dickey-Fuller
8. ARIMA(1,0,1) baseline modeling
9. GARCH(1,1) volatility modeling & persistence analysis
10. Volatility forecasting — 5-day ahead
11. Dollar bars — de Prado alternative data structure
12. Triple barrier method — proper ML labeling
13. Feature engineering — momentum, volatility, volume, mean reversion
14. Random Forest — TimeSeriesSplit 5-fold cross validation
15. Feature importance analysis
16. SMOTE — handling class imbalance
17. GARCH integration sebagai fitur ML

## Model Performance Summary

| Model | Keterangan | Hasil |
|-------|------------|-------|
| ARIMA(1,0,1) | Baseline time series | Tidak signifikan (p>0.05) |
| GARCH(1,1) | Volatility modeling | Beta signifikan, persistence 0.9277 |
| Random Forest | ML classifier | ~42%, tidak konsisten |
| Random Forest + SMOTE | Handle imbalance | ~42%, minimal improvement |
| Random Forest + GARCH | Hybrid approach | 41.22%, tidak membantu |

## Key Insight
Seluruh pipeline mengkonfirmasi satu temuan yang konsisten:
**BBCA adalah pasar yang cukup efisien.** Fitur teknikal sederhana 
tidak cukup untuk menghasilkan alpha yang konsisten.

ARIMA tidak signifikan, Random Forest akurasi rendah, dan GARCH 
vol tidak meningkatkan prediksi arah — semuanya konsisten dengan 
Efficient Market Hypothesis pada saham blue chip yang heavily traded.

Untuk alpha yang nyata dibutuhkan:
- Data intraday tick level (bukan daily)
- Fitur sophisticated: NLP sentiment, macro indicators, order flow
- Lebih banyak data historis
- Model yang lebih powerful: XGBoost, LSTM

Ini adalah temuan yang valid secara ilmiah — bukan kegagalan 
pipeline, tapi konfirmasi empiris bahwa pasar BBCA tidak bisa 
dikalahkan dengan pendekatan sederhana.

## Tech Stack
- Python 3
- pandas, numpy, matplotlib
- statsmodels, arch
- scikit-learn, imbalanced-learn
- yfinance

## Next Steps
- [ ] XGBoost sebagai model alternatif
- [ ] NLP sentiment dari berita keuangan sebagai fitur
- [ ] Integrasi data makroekonomi (suku bunga, kurs)
- [ ] Data intraday untuk dollar bars yang lebih akurat
- [ ] Ekspansi ke multi-saham BEI
- [ ] Backtesting & Sharpe Ratio evaluation

## Author
Aril Satrio Saputro
Data Science Student | Universitas Airlangga
GitHub: github.com/satrio-lakers
