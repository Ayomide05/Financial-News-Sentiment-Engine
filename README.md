#  Financial News Sentiment Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scrapy](https://img.shields.io/badge/Framework-Scrapy-green)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue)

## Project Overview
This project is a **Quantitative Data Pipeline** designed to harvest, process, and analyze unstructured financial news in real-time.

Unlike standard web scrapers, this engine treats news as a financial signal. It utilizes a **Decoupled Producer-Consumer Architecture** to handle high-latency pagination (via Selenium) and high-concurrency extraction (via Scrapy), feeding a structured PostgreSQL warehouse used for sentiment analysis and price target extraction.

##  System Architecture

The system operates in two distinct stages to maximize efficiency and fault tolerance:

### **Stage 1: Discovery ( The Producer)**
* **Tool:** Selenium WebDriver
* **Function:** Navigates dynamic, React-based pagination on Kitco.com.
* **Logic:** Robustly identifies new article URLs using partial selector matching to avoid brittle hash-based failures. Pushes unique URLs to the database.

### **Stage 2: Enrichment (The Consumer)**
* **Tool:** Scrapy (Asynchronous)
* **Function:** Fetches full article text, timestamps, and metadata.
* **Performance:** Optimized to process pending articles in batches using `psycopg2` connection pooling.
* **Idempotency:** Designed to run continuously without creating duplicate entries.

##  NLP & Signal Processing
The pipeline includes a custom `pipelines.py` layer that performs real-time feature engineering:

* **Context-Aware Sentiment:** Scores news not just on polarity, but on market context (e.g., *'Strong Dollar'* is interpreted as Bearish for Gold).
* **Price Target Extraction:** Uses RegEx patterns to extract Support/Resistance levels and analyst price targets (e.g., "$2,400").
* **Urgency Classification:** Categorizes news as "Breaking," "High," or "Low" urgency based on keyword velocity.

## Tech Stack

| Component | Technology | Key Features |
| :--- | :--- | :--- |
| **Acquisition** | Python, Selenium, Scrapy | Hybrid architecture for speed vs. complexity balance. |
| **Storage** | PostgreSQL | Connection pooling, Upserts, Materialized Views. |
| **Processing** | Pandas, NLTK (Planned) | Vectorized text analysis. |
| **Environment** | Dotenv | Secure credential management. |

##  Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Financial-News-Sentiment-Engine.git](https://github.com/YOUR_USERNAME/Financial-News-Sentiment-Engine.git)

##  Author
   Gabriel Justina Ayomide
