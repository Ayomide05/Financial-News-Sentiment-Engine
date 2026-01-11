# 📊 Gold Market Sentiment Analyzer

A quantitative analysis pipeline that scrapes gold market news, performs sentiment analysis, and correlates findings with gold price movements to identify predictive signals.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Project Overview

**Research Question:** *Can news sentiment predict gold price movements?*

This project builds an end-to-end data pipeline to answer this question using:
- **3,376 news articles** scraped from Kitco.com
- **466 trading days** of gold price data
- **22 months** of historical coverage (Feb 2024 - Jan 2026)

---

## 📈 Key Findings

### Summary Dashboard
![Summary Dashboard](kitco_scraper/visualizations/00_summary_dashboard.png)

### Main Results

| Finding | Value | Interpretation |
|---------|-------|----------------|
| Same-day correlation | **0.26*** | Sentiment reflects current price action |
| Next-day correlation | **0.00** | Sentiment does NOT predict future prices |
| Urgency → Volatility | **0.21*** | High urgency news predicts price swings |
| Bullish win rate | **59.3%** | Better than random (50%) |
| Strong bullish win rate | **61.7%** | High conviction signals work better |

### Core Insight
> **News sentiment is REACTIVE, not PREDICTIVE.** Journalists write bullish articles AFTER gold rises, not before. However, combining sentiment with technical indicators (RSI oversold) shows promising results.

---

## 📊 Visualizations

### Correlation Heatmap
![Correlation Heatmap](kitco_scraper/visualizations/01_correlation_heatmap.png)

### Win Rate Analysis
![Win Rates](kitco_scraper/visualizations/02_win_rates.png)

### Sentiment vs Returns
![Sentiment vs Returns](kitco_scraper/visualizations/03_sentiment_vs_returns.png)

### Strategy Performance
![Strategy Performance](kitco_scraper/visualizations/05_strategy_performance.png)

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Scrapy     │     │  PostgreSQL  │     │   Analysis   │    │
│  │   Spiders    │────▶│   Database   │────▶│   & Charts   │    │
│  │              │     │              │     │              │    │
│  │ • Headlines  │     │ • articles   │     │ • Correlation│    │
│  │ • Articles   │     │ • sentiment  │     │ • ML Models  │    │
│  │ • Selenium   │     │ • prices     │     │ • Visualize  │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   yfinance   │     │  Technical   │                         │
│  │  Gold Prices │────▶│  Indicators  │                         │
│  │              │     │              │                         │
│  │ • GC=F       │     │ • RSI, MACD  │                         │
│  │ • GLD, SLV   │     │ • Bollinger  │                         │
│  │ • DX-Y.NYB   │     │ • ATR, SMA   │                         │
│  └──────────────┘     └──────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Scraping** | Scrapy, Selenium, BeautifulSoup |
| **Database** | PostgreSQL |
| **Analysis** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Price Data** | yfinance |
| **NLP** | Custom sentiment lexicon |

---

## 📁 Project Structure
```
Gold-Sentiment-Analyzer/
├── kitco_scraper/
│   ├── kitco_scraper/
│   │   ├── spiders/
│   │   │   ├── kitcospider.py       # Headlines scraper
│   │   │   └── kitco_article_spider.py  # Full article scraper
│   │   ├── pipelines.py             # Data processing pipelines
│   │   ├── items.py                 # Data models
│   │   └── settings.py              # Scrapy settings
│   └── visualizations/              # Generated charts
│
├── src/
│   ├── config.py                    # Database configuration
│   ├── gold_prices.py               # Price data fetcher
│   ├── correlation_analysis.py      # Statistical analysis
│   └── visualizations.py            # Chart generation
│
├── .env                             # Environment variables
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## 🚀 Getting Started

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

7. **Run analysis**
```bash
python correlation_analysis.py
python visualizations.py
```

---

## 📊 Database Schema
```sql
-- Main tables
articles            -- 3,376 news articles
sentiment_analysis  -- Sentiment scores per article
market_prices       -- Daily OHLCV data
technical_indicators -- RSI, MACD, Bollinger, etc.
daily_sentiment     -- Aggregated daily metrics
```

---

## 🔬 Methodology

### Sentiment Analysis
- Custom gold-specific lexicon with weighted terms
- Bullish terms: surge, rally, breakout, haven (+2 to +3)
- Bearish terms: crash, plunge, selloff (-2 to -3)
- Context modifiers: dollar strength, yields, inflation
- Negation handling for accurate scoring

### Technical Indicators
- **RSI (14):** Overbought/oversold detection
- **MACD:** Momentum and trend direction
- **Bollinger Bands:** Volatility measurement
- **ATR (14):** Average True Range
- **SMA/EMA:** Trend following (5, 10, 20, 50 periods)

### Statistical Tests
- Pearson correlation with p-values
- T-tests for group comparisons
- Win rate analysis
- Sharpe ratio calculation

---

## 📈 Trading Signals Discovered

| Strategy | Days | Win Rate | Avg Return | Sharpe |
|----------|------|----------|------------|--------|
| High Bullish (>70%) | 183 | 61.7% | +0.227% | 3.20 |
| Bullish + Oversold RSI | 32 | 59.4% | +0.414% | 8.05 |

*Note: These results are from historical analysis and may not persist in live trading.*

---

## 🎓 Skills Demonstrated

- **Data Engineering:** Web scraping, ETL pipelines, database design
- **NLP:** Custom sentiment analysis with domain-specific lexicon
- **Statistical Analysis:** Correlation, hypothesis testing, significance
- **Financial Knowledge:** Technical indicators, market microstructure
- **Python:** Scrapy, Pandas, PostgreSQL, Matplotlib
- **Quantitative Research:** Signal discovery, backtesting methodology

---

## 📚 Future Enhancements

- [ ] FinBERT deep learning sentiment (compare with rule-based)
- [ ] ML models (XGBoost, Random Forest) for prediction
- [ ] Streamlit interactive dashboard
- [ ] Real-time data pipeline with Airflow
- [ ] Extended backtesting with transaction costs

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Gabriel Justina Ayomide**


---

