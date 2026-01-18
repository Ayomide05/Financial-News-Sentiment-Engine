# 📊 Gold Market Sentiment Analyzer

A quantitative analysis pipeline that scrapes gold market news, performs sentiment analysis using both rule-based and deep learning (FinBERT) approaches, and builds ML models to predict gold price movements.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)
![FinBERT](https://img.shields.io/badge/FinBERT-NLP-purple.svg)

---

##  Project Overview

**Research Question:** *Can news sentiment predict gold price movements?*

This project builds an end-to-end data pipeline to answer this question using:
- **3,376 news articles** scraped from Kitco.com
- **466 trading days** of gold price data
- **22 months** of historical coverage (Feb 2024 - Jan 2026)
- **Two sentiment approaches** compared: Rule-based vs FinBERT
- **5 ML models** trained and optimized

---

##  Key Findings

### Executive Summary

| Finding | Result |
|---------|--------|
| Best ML Model | XGBoost Tuned (50.0% accuracy) |
| Sentiment Contribution | 45.2% of feature importance |
| High Confidence Win Rate | 75.0% |
| Best Sharpe Ratio | 5.28 (ML signals) |
| Rule-based vs FinBERT | Rule-based won for ML (+6.8% vs -2.3%) |

### Core Insight

> **News sentiment is REACTIVE, not PREDICTIVE.** Journalists write bullish articles AFTER gold rises, not before. However, combining sentiment with technical indicators and using high-confidence ML signals achieves 75% win rate.

---

##  Methodology: Rule-Based vs Deep Learning Sentiment

I compared two sentiment analysis approaches:

### Correlation with Next-Day Returns

| Method | Correlation | P-Value |
|--------|-------------|---------|
| Rule-based | 0.0015 | 0.974 |
| FinBERT | 0.0382 | 0.411 |

### ML Model Performance

| Feature Set | Accuracy | Change vs Baseline |
|-------------|----------|-------------------|
| Technical Only | 47.3% | Baseline |
| **Technical + Rule-based** | **50.5%** | **+6.8%** ✅ |
| Technical + FinBERT | 46.2% | -2.3% |
| Technical + Both | 48.4% | +2.3% |

### Win Rate Comparison

| Signal Type | Rule-Based | FinBERT |
|-------------|------------|---------|
| Bullish/Positive | 59.3% | 62.9% |
| Bearish/Negative | 36.8% | 41.6% |

### Key Insight

**Rule-based sentiment outperformed FinBERT for ML prediction** because domain-specific features (gold market terminology) provided better signal than generic financial NLP. This demonstrates that sophisticated models don't always win - understanding your domain matters more.

---

##  Machine Learning Models

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 41.3% | 62.9% | 27.9% | 0.386 | 0.505 |
| Random Forest | 43.5% | 62.2% | 37.7% | 0.469 | 0.492 |
| Gradient Boosting | 43.5% | 60.5% | 42.6% | 0.500 | 0.440 |
| XGBoost | 42.4% | 59.5% | 41.0% | 0.485 | 0.443 |
| **XGBoost Tuned** | **50.0%** | **64.7%** | **54.1%** | **0.589** | 0.471 |

### Hyperparameter Tuning Results

**+17.9% accuracy improvement** after RandomizedSearchCV optimization:
```
Best Parameters:
├── n_estimators: 50
├── max_depth: 5
├── learning_rate: 0.2
├── min_child_weight: 5
├── subsample: 0.8
├── colsample_bytree: 1.0
└── gamma: 0.2
```

### Feature Importance

| Category | Importance | Top Features |
|----------|------------|--------------|
| **Technical** | 54.8% | atr_change, bb_position, return_lag1 |
| **Sentiment** | 45.2% | bullish_ratio_ma3, sentiment_ma5, bullish_ratio |

---

##  Trading Strategy Performance

| Strategy | Return | Sharpe | Win Rate | Trades |
|----------|--------|--------|----------|--------|
| Buy & Hold | +32.1% | 3.52 | 65.2% | 92 |
| ML All Signals | +18.8% | **5.28** | 70.4% | 27 |
| ML High Conf (>60%) | +16.1% | 4.73 | **75.0%** | 20 |
| ML Very High Conf (>70%) | +7.4% | 3.39 | 72.7% | 11 |

### Key Insight

**Trade less, win more.** High-confidence ML signals achieve 75% win rate with significantly better risk-adjusted returns (Sharpe 5.28 vs 3.52).

---

##  Visualizations

### Summary Dashboard
![Summary Dashboard](kitco_scraper/visualizations/00_summary_dashboard.png)

### Correlation Heatmap
![Correlation Heatmap](kitco_scraper/visualizations/01_correlation_heatmap.png)

### Feature Importance
![Feature Importance](kitco_scraper/ml_results/02_feature_importance.png)

### Trading Simulation
![Trading Simulation](kitco_scraper/ml_results/03_trading_simulation.png)

---

##  Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Scrapy     │     │  PostgreSQL  │     │   Analysis   │    │
│  │   Spiders    │────▶│   Database   │────▶│   & ML       │    │
│  │              │     │              │     │              │    │
│  │ • Headlines  │     │ • articles   │     │ • Correlation│    │
│  │ • Articles   │     │ • sentiment  │     │ • XGBoost    │    │
│  │ • Selenium   │     │ • prices     │     │ • FinBERT    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   yfinance   │     │  Technical   │     │  Sentiment   │    │
│  │  Gold Prices │────▶│  Indicators  │     │  Analysis    │    │
│  │              │     │              │     │              │    │
│  │ • GC=F       │     │ • RSI, MACD  │     │ • Rule-based │    │
│  │ • GLD, SLV   │     │ • Bollinger  │     │ • FinBERT    │    │
│  │ • DX-Y.NYB   │     │ • ATR, SMA   │     │ • Comparison │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

##  Tech Stack

| Category | Technologies |
|----------|-------------|
| **Scraping** | Scrapy, Selenium, BeautifulSoup |
| **Database** | PostgreSQL |
| **NLP** | Custom lexicon, FinBERT (Transformers) |
| **ML** | Scikit-learn, XGBoost |
| **Analysis** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Price Data** | yfinance |

---

##  Project Structure
```
Gold-Sentiment-Analyzer/
├── kitco_scraper/
│   ├── kitco_scraper/
│   │   ├── spiders/
│   │   │   ├── kitcospider.py           # Headlines scraper
│   │   │   └── kitco_article_spider.py  # Full article scraper
│   │   ├── pipelines.py                 # Data processing (9 pipelines)
│   │   ├── items.py                     # Data models
│   │   └── settings.py                  # Scrapy settings
│   ├── visualizations/                  # Generated charts
│   └── ml_results/                      # ML outputs
│
├── src/
│   ├── config.py                        # Database configuration
│   ├── gold_prices.py                   # Price data fetcher
│   ├── correlation_analysis.py          # Statistical analysis
│   ├── correlation_finbert.py           # FinBERT comparison
│   ├── ml_models.py                     # ML pipeline + tuning
│   ├── ml_models_finbert.py             # Feature set comparison
│   ├── finbert_sentiment.py             # Deep learning sentiment
│   └── visualizations.py                # Chart generation
│
├── .env                                 # Environment variables
├── requirements.txt                     # Dependencies
└── README.md                            # This file
```

---

##  Getting Started

### Prerequisites
- Python 3.9+
- PostgreSQL 15+
- Chrome (for Selenium)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gold-sentiment-analyzer.git
cd gold-sentiment-analyzer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
DB_NAME=gold_analysis
DB_USER=postgres
DB_PASS=your_password
DB_HOST=localhost
DB_PORT=5432
```

5. **Run the scrapers**
```bash
cd kitco_scraper
scrapy crawl kitcospider        # Scrape headlines
scrapy crawl kitco_articles     # Scrape full articles
```

6. **Fetch gold prices**
```bash
cd ../src
python gold_prices.py
```

7. **Run sentiment analysis**
```bash
python finbert_sentiment.py     # FinBERT (takes ~30 min)
```

8. **Run ML models**
```bash
python ml_models.py             # Includes hyperparameter tuning
python ml_models_finbert.py     # Compare feature sets
```

9. **Generate visualizations**
```bash
python visualizations.py
```

---

##  Database Schema
```sql
-- Core tables
articles                -- 3,376 news articles
sentiment_analysis      -- Rule-based sentiment scores
finbert_sentiment       -- FinBERT sentiment scores
market_prices          -- Daily OHLCV data
technical_indicators   -- RSI, MACD, Bollinger, etc.

-- Aggregated tables
daily_sentiment        -- Rule-based daily metrics
daily_sentiment_finbert -- FinBERT daily metrics
```

---

##  Methodology

### Sentiment Analysis

**Rule-based approach:**
- Custom gold-specific lexicon with weighted terms
- Bullish terms: surge, rally, breakout, haven (+2 to +3)
- Bearish terms: crash, plunge, selloff (-2 to -3)
- Context modifiers: dollar strength, yields, inflation

**FinBERT approach:**
- Pre-trained BERT model fine-tuned on financial text
- Outputs: Positive, Negative, Neutral with probabilities
- Combined score: 40% headline + 60% content

### Technical Indicators
- **RSI (14):** Overbought/oversold detection
- **MACD:** Momentum and trend direction
- **Bollinger Bands:** Volatility measurement
- **ATR (14):** Average True Range
- **SMA/EMA:** Trend following (5, 10, 20, 50 periods)

### ML Pipeline
1. Feature engineering (38 features)
2. Time-series train/test split (80/20)
3. Train 5 models (Logistic, RF, GB, XGBoost, XGBoost Tuned)
4. Hyperparameter tuning (RandomizedSearchCV, 50 iterations)
5. Evaluate on test set

---

##  Results Interpretation

### Why Prediction is Hard

Financial markets are inherently difficult to predict. Our models achieved ~50% accuracy, which may seem low but:

1. **Markets are efficient** - news is quickly priced in
2. **50% with edge > 50% random** - small edges compound
3. **75% win rate on high-confidence** - selectivity matters
4. **Sharpe of 5.28** - excellent risk-adjusted returns

### Practical Takeaways

1. **Sentiment reflects, doesn't predict** - same-day correlation (0.26), next-day (~0)
2. **Domain-specific > generic models** - rule-based beat FinBERT
3. **Combine signals** - sentiment + technical works better
4. **Be selective** - high-confidence trades have better outcomes

---

##  Skills Demonstrated

- **Data Engineering:** Web scraping, ETL pipelines, database design
- **NLP:** Rule-based sentiment, FinBERT deep learning, comparison analysis
- **Machine Learning:** Classification, hyperparameter tuning, feature importance
- **Statistical Analysis:** Correlation, hypothesis testing, significance testing
- **Financial Knowledge:** Technical indicators, trading simulation, Sharpe ratio
- **Python:** Scrapy, Pandas, Scikit-learn, XGBoost, Transformers, PostgreSQL

---

##  Future Enhancements

- [ ] Real-time data pipeline with Airflow
- [ ] Streamlit interactive dashboard
- [ ] LSTM/Transformer models for sequence prediction
- [ ] Extended backtesting with transaction costs
- [ ] Multi-asset expansion (silver, commodities)

---

##  Author

**Gabriel Justina Ayomide**
- Email: gabrieljustina4@gmail.com

---

## Acknowledgments

- [Kitco.com](https://www.kitco.com) for gold market news
- [Yahoo Finance](https://finance.yahoo.com) for price data
- [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) for financial NLP model
- [Anthropic Claude](https://www.anthropic.com) for development assistance