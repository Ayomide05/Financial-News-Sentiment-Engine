import yfinance as yf
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import DB_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GoldPriceFetcher:
    """This fetches and stores gold/silver price data"""
    SYMBOLS = {
        'GC=F': 'Gold Futures',
        'SI=F': 'Silver Futures',
        'GLD': 'Gold ETF (SPDR)',
        'SLV': 'Silver ETF (iShares)',
        'DX-Y.NYB': 'US Dollar Index',
        '^TNX': '10-Year Treasury Yield'
    }
    def __init__(self):
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def create_tables(self):
        """This creates the necessary tables"""
        cur = self.conn.cursor()

        #Market prices table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS market_prices(
                id SERIAL,
                date DATE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                name VARCHAR(100),
                open DECIMAL(12, 4),
                high DECIMAL(12, 4),
                low DECIMAL(12, 4),
                close DECIMAL(12, 4),
                adj_close DECIMAL(12, 4),
                volume BIGINT,
                daily_return DECIMAL(10, 6),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, symbol)
            )
        ''')

        # Technical indicators table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                date DATE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                sma_5 DECIMAL(12, 4),
                sma_10 DECIMAL(12, 4),
                sma_20 DECIMAL(12, 4),
                sma_50 DECIMAL(12, 4),
                ema_12 DECIMAL(12, 4),
                ema_26 DECIMAL(12, 4),
                macd DECIMAL(12, 4),
                macd_signal DECIMAL(12, 4),
                macd_histogram DECIMAL(12, 4),
                rsi_14 DECIMAL(8, 4),
                bollinger_upper DECIMAL(12,4),
                bollinger_middle DECIMAL(12,4),
                bollinger_lower DECIMAL(12,4),
                atr_14 DECIMAL(12,4),
                volatility_20d DECIMAL(10,6),
                price_vs_sma20 DECIMAL(10,6),
                trend_direction VARCHAR(10),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, symbol)
            )
        ''')

        # Daily sentiment aggregare table (for correlation)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS daily_sentiment (
                date DATE PRIMARY KEY,
                article_count INTEGER,
                avg_sentiment_score DECIMAL(10, 4),
                median_sentiment_score DECIMAL(10, 4),
                min_sentiment_score DECIMAL(10, 4),
                max_sentiment_score DECIMAL(10, 4),
                bullish_count INTEGER,
                bearish_count INTEGER,
                neutral_count INTEGER,
                bullish_ratio DECIMAL(10, 4),
                bearish_ratio DECIMAL(10, 4),
                avg_urgency_score DECIMAL(10, 4),
                breaking_news_count INTEGER,
                high_urgency_count INTEGER,
                dominant_category VARCHAR(50),
                sentiment_std DECIMAL(10, 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP                
            )
        ''')

        # Create indexes
        cur.execute('CREATE INDEX IF NOT EXISTS idx_market_prices_symbol ON market_prices(symbol)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_market_prices_date ON market_prices(date DESC)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_technical_symbol ON technical_indicators(symbol)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_technical_date ON technical_indicators(date DESC)')
        

        self.conn.commit()
        logger.info("Tables created successfully")
    def get_news_date_range(self):
        """Get the date range of scraped news articles"""
        cur = self.conn.cursor()

        # Using published_date
        cur.execute('''
            SELECT
                MIN(DATE(published_date)) as min_date,
                MAX(DATE(published_date)) as max_date,
                COUNT(*) as total_articles,
                COUNT(DISTINCT DATE(published_date)) as unique_days
            FROM articles
            WHERE published_date IS NOT NULL
        ''')

        result = cur.fetchone()
        min_date, max_date, total, unique_days = result

        print("SCRAPED NEWS DATA SUMMARY")
        print("="*60)
        print(f"  Total Articles: {total}")
        print(f"  Unique Days: {unique_days}")
        print(f"  Date Range: {min_date} to {max_date}")
        print("="*60 + "\n")
        
        return min_date, max_date, total
    
    def fetch_for_news_range(self, buffer_days=60):
        """
        We use our get_news_date_range function to fetch for gold proces dates where we have new
        buffer_days: Extra days before first new date (for calculating indicators)
        """
        min_date, max_date, total_articles = self.get_news_date_range()

        if min_date is None:
            logger.error("No articles with dates found! Run the scraper first.")
            return 0
        
        # Add buffer for technical indicator calculation
        start = (min_date - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        end = (max_date + timedelta(days=2)).strftime('%Y-%m-%d')

        logger.info(f"Fetching prices from {start} to {end} (includes {buffer_days}-day buffer for indicators)")

        rows = self.fetch_historical(start_date=start, end_date=end)

        # Calculate technical indicators
        self.calculate_technical_indicators()

        # Aggregate_daily_sentiment
        self.aggregate_daily_sentiment()

        return rows
    
    def fetch_historical(self, start_date, end_date=None):
        """Fetch historical price data for all symbols"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        total_rows = 0

        for symbol, name in self.SYMBOLS.items():
            try:
                logger.info(f"Fetching {name} ({symbol})...")
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )

                if data.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue

                rows = self._save_prices(data, symbol, name)
                total_rows += rows
                logger.info(f"Saved {rows} rows for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        self._calculate_returns()

        logger.info(f"Total price rows saved: {total_rows}")
        return total_rows
    def fetch_daily_update(self):
        """Fetch latest prices (for daily automation)"""
        start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        rows = self.fetch_historical(start_date=start)
        self.calculate_technical_indicators()
        self.aggregate_daily_sentiment()
        return rows
    def _save_prices(self, data, symbol, name):
        """Save price data to database"""
        cur = self.conn.cursor()
        rows_saved = 0
        
        # if the dataframe has multiple levels (Price, Ticker), extract just our symbol.
        df_to_process = data
        if isinstance(data.columns, pd.MultiIndex):
            try:
                df_to_process = data.xs(symbol, axis=1, level=1, drop_level=True)
            except KeyError:
                try:
                    df_to_process = data.xs(symbol, axis=1, level=0, drop_level=True)
                except KeyError:
                    logger.warning(f"could not find data for {symbol} in MultiIndex.")
                    return 0

        for date, row in df_to_process.iterrows():
            try:
                open_val = row.get('Open')
                high_val = row.get('High')
                low_val = row.get('Low')
                close_val = row.get('Close')
                adj_close_val = row.get('Adj Close')
                volume_val = row.get('Volume')
              
                cur.execute('''
                    INSERT INTO market_prices (date, symbol, name, open, high, low, close, adj_close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, symbol) DO UPDATE SET
                       open = EXCLUDED.open,
                       high = EXCLUDED.high,
                       low = EXCLUDED.low,
                       close = EXCLUDED.close,
                       adj_close = EXCLUDED.adj_close,
                       volume = EXCLUDED.volume 
                ''', (
                    date.date(),
                    symbol,
                    name,
                    float(open_val) if pd.notna(open_val) else None,
                    float(high_val) if pd.notna(high_val) else None,
                    float(low_val) if pd.notna(low_val) else None,
                    float(close_val) if pd.notna(close_val) else None,
                    float(adj_close_val) if pd.notna(adj_close_val) else None,
                    int(volume_val) if pd.notna(volume_val) else None
                ))
                rows_saved += 1
            except Exception as e:
                logger.error(f"Error saving row for {date}: {e}")
        self.conn.commit()
        return rows_saved
    
    def _calculate_returns(self):
        """Calculate daily returns"""
        cur = self.conn.cursor()

        cur.execute('''
            UPDATE market_prices m1
            SET daily_return = (m1.close -m2.close) / NULLIF(m2.close, 0)
            FROM market_prices m2
            WHERE m1.symbol = m2.symbol
            AND m2.date = (
                SELECT MAX(date)
                FROM market_prices m3
                WHERE m3.symbol = m1.symbol
                AND m3.date < m1.date
            )
            AND m1.daily_return IS NULL
        ''')
        self.conn.commit()
        logger.info("Daily returns calculated")

    def calculate_technical_indicators(self):
        """Calculate technical indicators for all symbols"""
        logger.info("Calculating technical indicators...")
        
        for symbol in self.SYMBOLS.keys():
            self._calculate_indicators_for_symbol(symbol)

        logger.info("Technical indicators calculated for all symbols")

    def _calculate_indicators_for_symbol(self, symbol):
        """Calculate technical indicators for a single symbol"""
        cur = self.conn.cursor()

        # Get price data
        cur.execute('''
            SELECT date, open, high, low, close, volume, daily_return
            FROM market_prices
            WHERE symbol = %s
            ORDER BY date                     
        ''', (symbol,))
        rows = cur.fetchall()
        if len(rows) < 26:
            logger.warning(f"Not enough data for {symbol} (need 26 rows, have {len(rows)})")
            return
        df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'daily_return'])

        # Convert to float
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        # Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        #RSI (14-period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bollinger_middle'] = df['sma_20']
        rolling_std = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['sma_20'] + (rolling_std * 2)
        df['bollinger_lower'] = df['sma_20'] - (rolling_std * 2)

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()

        # 20-day Volatility (standard deviation of returns)
        df['volatility_20d'] = df['daily_return'].astype(float).rolling(window=20).std()

        # Price vs SMA20 (percentage)
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']

        # Trend Direction
        df['trend_direction'] = 'neutral'
        df.loc[df['close'] > df['sma_20'], 'trend_direction'] = 'bullish'
        df.loc[df['close'] < df['sma_20'], 'trend_direction'] = 'bearish'

        # save to database
        for _, row in df.iterrows():
            if pd.isna(row['sma_50']) or pd.isna(row['sma_20']):    # Skip rows without enough data
                continue

            try:
                cur.execute('''
                    INSERT INTO technical_indicators 
                    (date, symbol, sma_5, sma_10, sma_20, sma_50, ema_12, ema_26, macd, 
                    macd_signal, macd_histogram, rsi_14, bollinger_upper, bollinger_middle,
                    bollinger_lower, atr_14, volatility_20d, price_vs_sma20, trend_direction)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, symbol) DO UPDATE SET
                        sma_5 = EXCLUDED.sma_5,
                        sma_10 = EXCLUDED.sma_10,
                        sma_20 = EXCLUDED.sma_20,
                        sma_50 = EXCLUDED.sma_50,
                        ema_12 = EXCLUDED.ema_12,
                        ema_26 = EXCLUDED.ema_26,
                        macd = EXCLUDED.macd,
                        macd_signal = EXCLUDED.macd_signal,
                        macd_histogram = EXCLUDED.macd_histogram,
                        rsi_14 = EXCLUDED.rsi_14,
                        bollinger_upper = EXCLUDED.bollinger_upper,
                        bollinger_middle = EXCLUDED.bollinger_middle,
                        bollinger_lower = EXCLUDED.bollinger_lower,
                        atr_14 = EXCLUDED.atr_14,
                        volatility_20d = EXCLUDED.volatility_20d,
                        price_vs_sma20 = EXCLUDED.price_vs_sma20,
                        trend_direction = EXCLUDED.trend_direction
                ''',(
                row['date'],
                symbol,
                float(row['sma_5']),
                float(row['sma_10']),   
                float(row['sma_20']), 
                float(row['sma_50']),
                float(row['ema_12']), 
                float(row['ema_26']), 
                float(row['macd']), 
                float(row['macd_signal']), 
                float(row['macd_histogram']),
                float(row['rsi_14']), 
                float(row['bollinger_upper']), 
                float(row['bollinger_middle']), 
                float(row['bollinger_lower']),
                float(row['atr_14']), 
                float(row['volatility_20d']), 
                float(row['price_vs_sma20']), 
                row['trend_direction']
                ))
            except Exception as e:
                logger.error(f"Error saving indicators for {row['date']}: {e}")
        self.conn.commit()
        logger.info(f"Indicators calculated for {symbol}")

    def aggregate_daily_sentiment(self):
        """Aggregate article sentiment by day for correlation analysis"""
        logger.info("Aggregating daily sentiment...")

        cur = self.conn.cursor()

        # Using published_date
        cur.execute('''
            INSERT INTO daily_sentiment (
                date, article_count, avg_sentiment_score, min_sentiment_score, max_sentiment_score,
                bullish_count, bearish_count, neutral_count, bullish_ratio, bearish_ratio,
                avg_urgency_score, breaking_news_count, high_urgency_count, dominant_category, sentiment_std
            )
            SELECT 
                DATE(a.published_date) as date,
                COUNT(*) as article_count,
                AVG(s.sentiment_score) as avg_sentiment_score,
                MIN(s.sentiment_score) as min_sentiment_score,
                MAX(s.sentiment_score) as max_sentiment_score,
                SUM(CASE WHEN s.sentiment ILIKE '%%bullish%%' THEN 1 ELSE 0 END) as bullish_count,
                SUM(CASE WHEN s.sentiment ILIKE '%%bearish%%' THEN 1 ELSE 0 END) as bearish_count,
                SUM(CASE WHEN s.sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                CAST(SUM(CASE WHEN s.sentiment ILIKE '%%bullish%%' THEN 1 ELSE 0 END) AS DECIMAL) / NULLIF(COUNT(*), 0) as bullish_ratio,
                CAST(SUM(CASE WHEN s.sentiment ILIKE '%%bearish%%' THEN 1 ELSE 0 END) AS DECIMAL) / NULLIF(COUNT(*), 0) as bearish_ratio,
                AVG(m.urgency_score) as avg_urgency_score,
                SUM(CASE WHEN m.is_breaking_news = true THEN 1 ELSE 0 END) as breaking_news_count,
                SUM(CASE WHEN m.urgency_category IN ('high', 'critical') THEN 1 ELSE 0 END) as high_urgency_count,
                MODE () WITHIN GROUP (ORDER BY s.primary_category) as dominant_category,
                STDDEV(s.sentiment_score) as sentiment_std
            FROM articles a
            JOIN sentiment_analysis s ON a.id = s.article_id
            LEFT JOIN market_urgency m ON a.id = m.article_id
            WHERE a.published_date IS NOT NULL
            GROUP BY DATE(a.published_date)
            ON CONFLICT (date) DO UPDATE SET
                article_count = EXCLUDED.article_count,
                avg_sentiment_score = EXCLUDED.avg_sentiment_score,
                min_sentiment_score = EXCLUDED.min_sentiment_score,
                max_sentiment_score = EXCLUDED.max_sentiment_score,
                bullish_count = EXCLUDED.bullish_count,
                bearish_count = EXCLUDED.bearish_count,
                neutral_count = EXCLUDED.neutral_count,
                bullish_ratio = EXCLUDED.bullish_ratio,
                bearish_ratio = EXCLUDED.bearish_ratio,
                avg_urgency_score = EXCLUDED.avg_urgency_score,
                breaking_news_count = EXCLUDED.breaking_news_count,
                high_urgency_count = EXCLUDED.high_urgency_count,
                dominant_category = EXCLUDED.dominant_category,
                sentiment_std = EXCLUDED.sentiment_std
        ''')

        self.conn.commit()

        # Log summary
        cur.execute('SELECT COUNT(*), MIN(date), MAX(date) FROM daily_sentiment')
        count, min_date, max_date = cur.fetchone()
        logger.info(f"Daily sentiment aggregated: {count} days from {min_date} to {max_date}")

    def get_summary(self):
        """Show summary of all data"""
        cur = self.conn.cursor()
        
        print("MARKET PRICES SUMMARY")
               
        cur.execute('''
            SELECT 
                symbol, name, COUNT(*) as rows,
                MIN(date) as first_date, MAX(date) as last_date,
                MIN(close) as min_price, MAX(close) as max_price
            FROM market_prices
            GROUP BY symbol, name
            ORDER BY symbol
        ''')
        
        for row in cur.fetchall():
            symbol, name, count, first, last, min_p, max_p = row
            print(f"\n{symbol} - {name}")
            print(f"  Rows: {count} | {first} to {last}")
            if min_p and max_p:
                print(f"  Price Range: ${float(min_p):.2f} - ${float(max_p):.2f}")
        
        print("TECHNICAL INDICATORS SUMMARY")
        cur.execute('''
            SELECT symbol, COUNT(*), MIN(date), MAX(date)
            FROM technical_indicators
            GROUP BY symbol
        ''')
        
        for row in cur.fetchall():
            symbol, count, first, last = row
            print(f"  {symbol}: {count} rows ({first} to {last})")
        
        print("DAILY SENTIMENT SUMMARY")
               
        cur.execute('''
            SELECT 
                COUNT(*) as days,
                MIN(date) as first_date,
                MAX(date) as last_date,
                AVG(article_count) as avg_articles_per_day,
                AVG(avg_sentiment_score) as overall_avg_sentiment,
                AVG(bullish_ratio) as avg_bullish_ratio
            FROM daily_sentiment
        ''')
        
        result = cur.fetchone()
        if result[0]:
            days, first, last, avg_articles, avg_sent, avg_bull = result
            print(f"  Days with data: {days}")
            print(f"  Date range: {first} to {last}")
            print(f"  Avg articles/day: {float(avg_articles):.1f}")
            if avg_sent:
                print(f"  Overall avg sentiment: {float(avg_sent):.2f}")
            if avg_bull:
                print(f"  Avg bullish ratio: {float(avg_bull)*100:.1f}%")
        
        print("DATA OVERLAP (for correlation)")
                
        cur.execute('''
            SELECT COUNT(*) 
            FROM daily_sentiment ds
            JOIN market_prices mp ON ds.date = mp.date AND mp.symbol = 'GC=F'
        ''')
        matching = cur.fetchone()[0]
        print(f"  Days with BOTH sentiment AND gold prices: {matching}")
        
        cur.execute('''
            SELECT COUNT(*) 
            FROM daily_sentiment ds
            JOIN market_prices mp ON ds.date = mp.date AND mp.symbol = 'GC=F'
            JOIN technical_indicators ti ON ds.date = ti.date AND ti.symbol = 'GC=F'
        ''')
        full_match = cur.fetchone()[0]
        print(f"  Days with sentiment + prices + indicators: {full_match}")
        
        print("="*70)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Main entry point"""
    fetcher = GoldPriceFetcher()
    
    try:
        # Check news data range first
        min_date, max_date, total = fetcher.get_news_date_range()
        
        if total == 0:
            print("\nNo articles found! Run the scraper first:")
            print("    cd kitco_scraper")
            print("    scrapy crawl kitcospider")
            print("    scrapy crawl kitco_articles")
            return
        
        # Fetch gold prices matching news dates
        print("Fetching gold prices for your news date range...")
        print("   (Including 60-day buffer for technical indicators)\n")
        fetcher.fetch_for_news_range(buffer_days=60)
        
        # Show summary
        fetcher.get_summary()
        
        print("\n All data loaded successfully!")
           
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()







            