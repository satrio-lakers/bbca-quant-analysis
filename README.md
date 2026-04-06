# BBCA Quant Analysis: Full ML Pipeline with Macro Integration

## Overview
Analisis kuantitatif komprehensif saham BBCA (Bank Central Asia) 
periode Januari 2022 – Januari 2024. Mengimplementasikan pipeline 
quant lengkap dari preprocessing hingga macro-integrated ML strategy, 
mengikuti pendekatan de Prado (2018) dalam Advances in Financial 
Machine Learning.

## Key Findings

### Data
- **Total bars final:** 336 (dollar bars, threshold 500M IDR)
- **Periode:** Feb 2022 – Des 2023
- **Distribusi label:** Buy 163 (48.5%) / Sell 135 (40.2%) / Hold 38 (11.3%)

### Statistical Properties
- **Volatilitas tahunan:** 19.95% rata-rata
- **Skewness:** -0.145 — sedikit miring kiri
- **Kurtosis:** 2.09 — platykurtic, kejadian ekstrem relatif jarang
- **ADF Test:** Harga Close tidak stasioner (p=0.518), 
  Log Returns stasioner (p=0.000) ✓
- **GARCH persistence (α+β):** 0.9277 — volatilitas sangat persisten
- **Volatility forecast 5 hari:** 1.17% per hari (di bawah baseline 1.28%)

### Outlier & Macro Events
- **Mei 2022:** -6.68% H+1 — Fed hike 50bps + krisis energi Eropa
- **Desember 2023:** +3.94% H+1 — Fed pivot signal + dovish language
- **3 outlier terdeteksi** via z-score, semua berkorelasi dengan FOMC

### Fed Impact Analysis (15 FOMC meetings)
- **Rata-rata H+5 setelah HIKE:** +0.24%
- **Rata-rata H+5 setelah HOLD:** +2.25%
- **Gap 2.01%** — hold regime konsisten lebih bullish untuk BBCA
- **Very hawkish (75bps) tidak selalu negatif** — pasar sering 
  sudah pricing in, BBCA diuntungkan dari NIM expansion

### ML Model Performance

| Model | Accuracy | Keterangan |
|-------|----------|------------|
| ARIMA(1,0,1) | — | Tidak signifikan (p>0.05) |
| GARCH(1,1) | — | Beta signifikan, persistence 0.9277 |
| Random Forest baseline | 40.71% | Technical features only |
| Random Forest + SMOTE | ~42% | Minimal improvement |
| Random Forest + GARCH vol | 41.22% | Tidak membantu |
| Random Forest + Fed features | 41.11% | +0.40% marginal gain |

### Feature Importance (Final Model)
1. ma_dist 0.198 — mean reversion signal terkuat
2. mom_10 0.159 — momentum 10 bar
3. vol_ratio 0.150 — volume relatif
4. vol_10 0.125 — volatilitas 10 bar
5. mom_5 0.117 — momentum 5 bar
6. fed_change_bps 0.106 — besaran Fed hike
7. log_return 0.091 — return kemarin
8. fed_sentiment 0.055 — redundant dengan change_bps

## Pipeline
1. Data acquisition — yfinance API (BBCA.JK)
2. Preprocessing — missing values, forward fill, duplicate check
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
14. Clean pipeline — reproducible data state (db_final)
15. Random Forest — TimeSeriesSplit 5-fold cross validation
16. Feature importance analysis
17. SMOTE — handling class imbalance experiment
18. GARCH integration sebagai fitur ML experiment
19. Fed events timeline — 15 FOMC meetings 2022-2023
20. Fed impact analysis — H+1, H+3, H+5 return setelah FOMC
21. Fed macro features integration ke ML model

## Key Insights

### 1. BBCA adalah pasar yang cukup efisien
Seluruh pipeline mengkonfirmasi satu temuan konsisten: fitur 
teknikal sederhana tidak menghasilkan alpha yang reliable. 
ARIMA tidak signifikan, Random Forest akurasi 40-42%, 
GARCH vol dan Fed features hanya marginal improvement.

### 2. Information surprise lebih penting dari keputusan aktual
Fed hike 75bps di Juni 2022 — BBCA justru naik karena pasar 
sudah pricing in. Fed hike 25bps di Maret 2022 — BBCA turun 
karena unexpected. Pasar bergerak karena selisih antara 
ekspektasi dan realita, bukan keputusannya sendiri.

### 3. Volatility persistence adalah temuan paling robust
GARCH persistence 0.9277 adalah satu-satunya temuan yang 
konsisten dan signifikan secara statistik. Sekali BBCA 
bergejolak, efeknya bertahan berbulan-bulan — terbukti dari 
elevated volatility Jun–Nov 2022 pasca very hawkish cycle.

### 4. Data quality adalah bottleneck utama
Hold tidak terprediksi di semua fold bukan karena model buruk, 
tapi karena 38 sampel terlalu sedikit untuk 5-fold CV. 
Root cause: daily data + 2 tahun = 336 bars, tidak cukup 
untuk ML yang robust di 3-class classification.

## Limitations & Next Steps

### Keterbatasan
- Data harian — tidak cukup granular untuk dollar bars yang akurat
- 2 tahun data — terlalu sedikit untuk ML yang robust
- Fed features low-frequency — 15 events di-forward fill ke 324 bars
- Scraping Kontan/berita blocked di Kaggle environment

### Next Steps
- [ ] Data intraday tick level untuk dollar bars yang akurat
- [ ] Extend periode data minimal 5-10 tahun
- [ ] NLP sentiment dari berita keuangan (IndoBERT)
- [ ] Fed Funds Futures sebagai market expectation proxy
- [ ] Yield curve shape (10Y-2Y spread) sebagai regime indicator
- [ ] XGBoost & LSTM sebagai model alternatif
- [ ] Markowitz portfolio optimization multi-saham BEI
- [ ] Backtesting dengan transaction costs & Sharpe Ratio

## Tech Stack
- Python 3
- pandas, numpy, matplotlib
- statsmodels, arch
- scikit-learn, imbalanced-learn
- yfinance

## References
- de Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley.
- Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity.
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*.

## Author
Aril Satrio Saputro
Data Science Student | Universitas Airlangga
GitHub: github.com/satrio-lakers
