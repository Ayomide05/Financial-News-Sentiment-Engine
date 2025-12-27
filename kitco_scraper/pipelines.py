# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from .items import KitcoItem, KitcoArticleItem
from datetime import datetime, timedelta
import re
import os
import logging
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from scrapy.utils.project import get_project_settings
from psycopg2.extras import RealDictCursor, Json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import hashlib

load_dotenv()
class DuplicatesPipeline:
    def __init__(self):
        self.scraped_urls = set()

    def process_item(self, item, spider):
        url = item.get('url')
        if url in self.scraped_urls:
            raise DropItem(f"Duplicate item found: {url}")
        else:
            self.scraped_urls.add(url)    
        return item
class ValidationPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)

        required_fields = []
        if isinstance(item, KitcoItem):
            required_fields = ['headline', 'timestamp', 'url']
        elif isinstance(item, KitcoArticleItem):
            required_fields = ['full_text', 'published_date', 'updated_date', 'author', 'tags']
        for field in required_fields:
            if not adapter.get(field):
                raise DropItem(f"Missing required field {field} in {type(item).__name__}")
        return item
    
class TimestampPipeline:
    def process_item(self, item, spider):
        if spider.name == "kitcospider":
            timestamp_text = item.get('timestamp')
            if timestamp_text:
                parsed = self.parse_date(timestamp_text, spider)  
                if parsed:
                    item['datetime'] = parsed.strftime("%Y-%m-%d %H:%M:%S")
                    item['date'] = parsed.strftime("%Y-%m-%d")
                    item['time'] = parsed.strftime("%H:%M:%S")    
        elif spider.name == "kitco_articles":
            if item.get('published_date'):
                parsed = self.parse_date(item['published_date'], spider)
                if parsed:
                    item['published_datetime'] = parsed.strftime("%Y-%m-%d %H:%M:%S") 
                    item['published_date_only'] = parsed.strftime("%Y-%m-%d")
                    item['published_time_only'] = parsed.strftime("%H:%M:%S")  

            if item.get('updated_date'):
                parsed = self.parse_date(item['updated_date'], spider)
                if parsed:
                    item['updated_datetime'] = parsed.strftime("%Y-%m-%d %H:%M:%S") 
                    item['updated_date_only'] = parsed.strftime("%Y-%m-%d")
                    item['updated_time_only'] = parsed.strftime("%H:%M:%S") 
        return item 
    def parse_date(self, date_str, spider):
        formats = [
            "%b %d, %Y %I:%M%p",      # Nov 25, 2024 3:30PM
            "%B %d, %Y %I:%M%p",      # November 25, 2024 3:30PM
            "%b %d, %Y %I:%M %p",     # Nov 25, 2024 3:30 PM
            "%b %d, %Y",              # Nov 25, 2024 (no time)
            "%b %d, %Y - %I:%M %p",  # Matches "Apr 01, 2025 - 5:34 PM"
            "%Y-%m-%d %H:%M:%S",      # Already formatted
        ] 
     
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        spider.logger.warning(f"Could not parse date: {date_str}")    
        return None
       

class NewsCategoryPipeline:
    def process_item(self, item, spider):
        if spider.name == 'kitco_articles' and item.get('full_text'):
            text_to_analyze = (item.get('headline', '') + ' ' + item.get('full_text', '')).lower()
        else:
            text_to_analyze = item.get('headline', '').lower()   

        categories = {
            'fed_policy': ['fed', 'fomc', 'powell', 'rate', 'monetary policy', 'federal reserve', 'hawkish','dovish'],
            'technical_analysis': ['resistance', 'support', 'chart','analysis', 'fibonacci', 'moving average', 'breakout', 'pattern'],
            'market_news': ['rally', 'surge', 'drop', 'fall', 'gain', 'loss', 'volatile', 'swing'],
            'geopolitical': ['war', 'conflict', 'tension', 'crisis', 'sanction', 'military', 'political'], 
            'china': ['china', 'chinese', 'beijing', 'yuan', 'shangbai', 'pboc'],
            'inflation': ['inflation', 'cpi', 'pce', 'consumer price', 'deflation', 'stagflation'],
            'dollar': ['dollar', 'usd', 'greenback', 'currency', 'forex', 'dxy'],
            'central_banks': ['ecb', 'boe', 'boj', 'central bank', 'christine lagarde', 'andrew bailey'],
            'mining': ['mining', 'production', 'mine', 'gold mine', 'barrick', 'newmont'],
            'etf': ['etf', 'gld', 'iau', 'gold fund', 'holdings'],
            'india': ['india', 'indian', 'rupee', 'delhi', 'mumbai', 'diwali', 'wedding season'],
            'crypto': ['bitcoin', 'crypto', 'digital gold', 'blockchain'],
        }   

        item['categories'] = []
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(text_to_analyze.count(keyword) for keyword in keywords)
            if score > 0:
                item['categories'].append(category)
                category_scores[category] = score

        if category_scores:
            item['primary_category'] = max(category_scores, key=category_scores.get)
        else:
            item['categories'] = ['general']
            item['primary_category'] = 'general' 
        item['category_confidence'] = len(item['categories'])  

        return item     
class SentimentPipeline:
    def __init__(self):
        # Gold-specific sentiment dictionaries with weights
        self.sentiment_words = {
            'strong_positive': {
                'surge': 3, 'soar': 3, 'rocket': 3, 'moon': 3,
                'skyrocket': 3, 'explode': 3, 'breakthrough': 3
            },
            'positive': {
                'rally': 2, 'gain': 2, 'rise': 2, 'jump': 2, 
                'bullish': 2, 'strong': 2, 'breakout': 2, 'climb': 2,
                'haven': 2, 'safe-haven': 2, 'inflation hedge': 2,
                'accumulation': 2, 'buying': 2
            },
            'mild_positive': {
                'up': 1, 'higher': 1, 'positive': 1, 'support': 1,
                'steady': 1, 'firm': 1, 'recover': 1, 'rebound': 1
            },
            'strong_negative': {
                'crash': -3, 'plunge': -3, 'collapse': -3, 'plummet': -3,
                'tank': -3, 'freefall': -3, 'capitulation': -3
            },
            'negative': {
                'drop': -2, 'fall': -2, 'decline': -2, 'weak': -2, 
                'bearish': -2, 'sink': -2, 'selloff': -2, 'slip': -2,
                'pressure': -2, 'breakdown': -2, 'selling': -2
            },
            'mild_negative': {
                'down': -1, 'lower': -1, 'negative': -1, 'resistance': -1,
                'dip': -1, 'ease': -1, 'soften': -1, 'pullback': -1
            }
        }
        
        # Context modifiers for gold
        self.context_modifiers = {
            'dollar_strong': -0.5,  # Strong dollar typically bad for gold
            'dollar_weak': 0.5,     # Weak dollar typically good for gold
            'yields_rise': -0.5,    # Rising yields bad for gold
            'yields_fall': 0.5,     # Falling yields good for gold
            'inflation': 0.5,       # Inflation good for gold
            'deflation': -0.5,      # Deflation bad for gold
            'uncertainty': 0.5,     # Uncertainty drives safe-haven demand
            'risk-on': -0.5,        # Risk-on sentiment bad for gold
            'risk-off': 0.5         # Risk-off sentiment good for gold
        }
        
        # Negation words
        self.negation_words = ['not', 'no', 'never', 'neither', 'nor', 'barely','hardly', 'scarcely', 'seldom', 'despite']
    
    def analyze_sentiment(self, text):
        text_lower = text.lower()
        
        # Calculate base sentiment score
        score = 0
        word_counts = {}
        
        # Check for sentiment words
        for category, words in self.sentiment_words.items():
            for word, weight in words.items():
                count = text_lower.count(word)
                if count > 0:
                    # Check for negation
                    if self.is_negated(word, text_lower):
                        weight = -weight  # Flip the sentiment
                    score += count * weight
                    word_counts[word] = count
        
        # Apply context modifiers
        context_adjustment = 0
        for context, modifier in self.context_modifiers.items():
            if context.replace('_', ' ') in text_lower or context.replace('-', ' ') in text_lower:
                context_adjustment += modifier
        
        # Adjust score with context
        final_score = score + (score * context_adjustment if score != 0 else context_adjustment)
        
        # Determine sentiment with more granular categories
        if final_score >= 5:
            sentiment = 'strong_bullish'
        elif final_score >= 2:
            sentiment = 'bullish'
        elif final_score > 0:
            sentiment = 'mild_bullish'
        elif final_score == 0:
            sentiment = 'neutral'
        elif final_score > -2:
            sentiment = 'mild_bearish'
        elif final_score > -5:
            sentiment = 'bearish'
        else:
            sentiment = 'strong_bearish'
        
        return {
            'sentiment': sentiment, 
            'score': final_score,
            'word_counts': word_counts,
            'context_factors': context_adjustment
        }
    
    def is_negated(self, word, text):
        """Check if a word is negated in the text"""
        # Simple negation detection - looks for negation words within 3 words before
        words = text.split()
        for i, w in enumerate(words):
            if word in w:
                # Check previous 3 words for negation
                start = max(0, i-3)
                for j in range(start, i):
                    if words[j] in self.negation_words:
                        return True
        return False
    
    def process_item(self, item, spider):
        if spider.name == "kitco_articles" and item.get("full_text"):
            headline_sentiment = self.analyze_sentiment(item.get('headline', ''))
            content_sentiment = self.analyze_sentiment(item.get('full_text', ''))
            
            # Store detailed analysis
            item['headline_sentiment'] = headline_sentiment['sentiment']
            item['headline_sentiment_score'] = headline_sentiment['score']
            item['content_sentiment'] = content_sentiment['sentiment']
            item['content_sentiment_score'] = content_sentiment['score']
            
            # Calculate overall sentiment with weighted average
            # Headline gets 40% weight, content gets 60%
            weighted_score = (headline_sentiment['score'] * 0.4) + (content_sentiment['score'] * 0.6)
            
            # Determine if there's conflict
            headline_bullish = 'bullish' in headline_sentiment['sentiment']
            content_bullish = 'bullish' in content_sentiment['sentiment']
            headline_bearish = 'bearish' in headline_sentiment['sentiment']
            content_bearish = 'bearish' in content_sentiment['sentiment']
            
            if (headline_bullish and content_bearish) or (headline_bearish and content_bullish):
                item['sentiment'] = 'mixed'
                item['sentiment_conflict'] = True
            else:
                # Re-categorize based on weighted score
                if weighted_score >= 5:
                    item['sentiment'] = 'strong_bullish'
                elif weighted_score >= 2:
                    item['sentiment'] = 'bullish'
                elif weighted_score > 0:
                    item['sentiment'] = 'mild_bullish'
                elif weighted_score == 0:
                    item['sentiment'] = 'neutral'
                elif weighted_score > -2:
                    item['sentiment'] = 'mild_bearish'
                elif weighted_score > -5:
                    item['sentiment'] = 'bearish'
                else:
                    item['sentiment'] = 'strong_bearish'
                item['sentiment_conflict'] = False
            
            item['sentiment_score'] = weighted_score
            item['sentiment_details'] = {
                'headline_words': headline_sentiment.get('word_counts', {}),
                'content_context': content_sentiment.get('context_factors', 0)
            }
        else:
            result = self.analyze_sentiment(item.get('headline', ''))
            item['sentiment'] = result['sentiment']
            item['sentiment_score'] = result['score']
            item['sentiment_conflict'] = False
        
        return item

class PriceTargetExtractorPipeline:
    def process_item(self, item, spider):
        if spider.name != 'kitco_articles' or not item.get('full_text'):
            return item

        text = item.get('full_text', '') + ' ' + item.get('headline', '')

            #Extract price mentions
        price_pattern = r'\$\s*(\d{1,2},?\d{3,4}(?:\.\d{2})?)'
        prices = re.findall(price_pattern, text)
        item['price_mentions'] = [float(p.replace(',', '')) for p in prices]

            #Extract targets with context
        target_patterns = [
            r'target(?:s|ed|ing)?\s*(?:of|at|to|near)?\s*\$?\s*(\d{1,2},?\d{3,4})',
            r'(?:sees?|forecast|predict|expect)\s*(?:gold)?\s*(?:at|to reach|hitting)?\s*\$?\s*(\d{1,2},?\d{3,4})',
            r'resistance\s*(?:at|near|around)?\s*\$?\s*(\d{1,2},?\d{3,4})',
            r'support\s*(?:at|near|around)?\s*\$?\s*(\d{1,2},?\d{3,4})',
        ]  

        targets = []
        support_levels =[]
        resistance_levels = []

        for pattern in target_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                price = float(match.replace(',', ''))
                if 'resistance' in pattern:
                    resistance_levels.append(price)
                elif 'support' in pattern:
                    support_levels.append(price)
                else:
                    targets.append(price)

        item['price_targets'] = list(set(targets))
        item['support_levels'] = list(set(support_levels))
        item['resistance_levels'] = list(set(resistance_levels))  

        #Average target if multiple
        if item['price_targets']:
            item['avg_price_target'] = sum(item['price_targets']) / len(item['price_targets'])
        return item

class MarketUrgencyPipeline:
    """Detect urgency and time-sensitive information with weighted scoring"""
    
    def __init__(self):
        # Weighted urgency words for financial markets
        self.urgency_weights = {
            # Critical market events (8-10)
            'crash': 10,
            'plunge': 9,
            'surge': 9,
            'breaking': 8,
            'emergency': 8,
            
            # High urgency (5-7)
            'flash': 7,
            'alert': 6,
            'urgent': 6,
            'just in': 6,
            'spike': 5,
            'tumble': 5,
            
            # Medium urgency (3-4)
            'immediate': 4,
            'now': 3,
            'today': 3,
            'sharp': 3,
            
            # Market-specific urgency terms
            'halt': 8,  # trading halt
            'circuit breaker': 9,
            'margin call': 7,
            'limit down': 8,
            'limit up': 8
        }
        
        # Context amplifiers
        self.amplifiers = {
            'extremely': 1.5,
            'very': 1.3,
            'highly': 1.3,
            'most': 1.2,
            'significantly': 1.2
        }
    
    def calculate_urgency_score(self, text):
        """Calculate weighted urgency score with context amplification"""
        score = 0
        text_lower = text.lower()
        
        # Check for urgency words and their weights
        for word, weight in self.urgency_weights.items():
            if word in text_lower:
                # Check for amplifiers before the urgent word
                amplifier_bonus = 1.0
                for amp, multiplier in self.amplifiers.items():
                    if f"{amp} {word}" in text_lower:
                        amplifier_bonus = multiplier
                        break
                
                score += weight * amplifier_bonus
        
        return score
    
    def categorize_urgency(self, score):
        """Categorize urgency based on score thresholds"""
        if score >= 15:
            return 'critical'
        elif score >= 10:
            return 'high'
        elif score >= 5:
            return 'medium'
        elif score > 0:
            return 'low'
        return 'none'
    
    def process_item(self, item, spider):
        text = item.get('headline', '')
        if spider.name == 'kitco_articles' and item.get('full_text'):
            text += ' ' + item.get('full_text', '')
        
        # Calculate weighted urgency score
        item['urgency_score'] = self.calculate_urgency_score(text)
        item['urgency_category'] = self.categorize_urgency(item['urgency_score'])
        item['is_breaking_news'] = item['urgency_score'] >= 10
        
        # Enhanced time references with market context
        time_words = {
            'immediate': ['now', 'immediately', 'today', 'right now', 'at open', 'at close'],
            'short_term': ['tomorrow', 'next week', 'coming days', 'soon', 'this week'],
            'medium_term': ['next month', 'quarter', 'q1', 'q2', 'q3', 'q4'],
            'long_term': ['next year', 'annual', 'yearly', '2025', '2026']
        }
        
        item['time_horizon'] = []
        text_lower = text.lower()
        for horizon, words in time_words.items():
            if any(word in text_lower for word in words):
                item['time_horizon'].append(horizon)
        
        if not item['time_horizon']:
            item['time_horizon'] = ['unspecified']
        
        # Add market-specific urgency flags
        item['market_flags'] = {
            'volatility_indicator': any(word in text_lower for word in 
                ['volatile', 'volatility', 'swing', 'whipsaw']),
            'price_movement': any(word in text_lower for word in 
                ['rally', 'selloff', 'correction', 'breakout']),
            'volume_indicator': any(word in text_lower for word in 
                ['heavy volume', 'thin trading', 'high volume'])
        }
        
        return item        

class ExpertSourceExtractorPipeline:
    def process_item(self, item, spider):
        if spider.name != 'kitco_articles' or not item.get('full_text'):
            return item

        text = item.get('full_text', '')
        
        institutions = {
            'goldman_sachs': ['goldman', 'goldman sachs'],
            'jp_morgan': ['jpmorgan', 'jp mprgan', 'jpm'],
            'bank_of_america': ['bank of america', 'bofa', 'merrill'],
            'citi': ['citi', 'citigroup', 'citibank'],
            'ubs': ['ubs'],
            'wells_fargo': ['wells fargo'],
            'morgan_stanley': ['morgan stanley'],
            'deutsche_bank': ['deutsche bank'],
            'barclays': ['barclays'],
            'credit_suisse': ['credit suisse']
        }    

        item['expert_sources'] = []
        for inst, keywords in institutions.items():
            if any(keyword in text.lower() for keyword in keywords):
                item['expert_sources'].append(inst)   

        # Credibility score based on number of expert sources
        item['source_credibility'] = len(item['expert_sources'])

        quote_pattern = r'"([^"]{20,200})"'
        quotes = re.findall(quote_pattern, text)
        item['expert_quotes'] = quotes[:5]

        return item                            

class PostgreSQLPipeline:
    def __init__(self):
        settings = get_project_settings()
        self.db_config = {
            "host": settings.get("DB_HOST"),
            "user": settings.get("DB_USER"),
            "password": settings.get("DB_PASS"),
            "port": settings.get("DB_PORT"),
            "dbname": settings.get("DB_NAME"),
        }
        
        self.connection_pool = None
        self.logger = logging.getLogger(__name__)

        self.stats = {
            'inserted': 0,        # New articles added to database
            'updated': 0,         # Existing articles that got updated
            'failed': 0           #Articles that failed to save
        }

    def open_spider(self,spider):
        """Initialize connection pool and create tables when spider opens"""
        try:
            self.create_database_if_not_exists()
            self.connection_pool = ThreadedConnectionPool(
                2, 10, **self.db_config    
            )
            self.create_tables()
            self.create_indexes()
            self.create_analysis_views()
            spider.logger.info("PostgreSQL pipeline initialized successfully")
        except Exception as e:
            spider.logger.error(f"Failed to Initialize PostgreSQL: {e}")
            raise
     

    def create_database_if_not_exists(self):
        """Create database if it does not exist"""
        conn = psycopg2.connect(
            host=self.db_config['host'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            port=self.db_config['port'],
            database='postgres'
        ) 
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check is database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (self.db_config['dbname'],)
        )
          
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {self.db_config['dbname']}")
            self.logger.info(f"Created database: {self.db_config['dbname']}")

        cursor.close()
        conn.close()     
    
    def create_tables(self):
        """Create all necessary tables with proper schema"""
        conn = self.connection_pool.getconn()
        cursor = conn.cursor()
        
        try:
            #Main articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                   id SERIAL PRIMARY KEY,
                   url TEXT UNIQUE NOT NULL,
                   url_hash VARCHAR(64) UNIQUE NOT NULL,
                   headline TEXT ,
                   full_text TEXT,
                   author VARCHAR(255),    

                   -- Timestamps
                   published_date TIMESTAMPTZ,
                   updated_date TIMESTAMPTZ,
                   scraped_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

                   -- Metadata
                   tags TEXT[],
                   created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                   updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP                                                                                           
                );
            """)
            #Sentiment analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                   id SERIAL PRIMARY KEY,
                   article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                           
                   -- Overall sentiment
                   sentiment VARCHAR(50),
                   sentiment_score DECIMAL(10,4),

                   -- Detailed Sentiment
                   headline_sentiment VARCHAR(50),
                   headline_sentiment_score DECIMAL(10,4),
                   content_sentiment VARCHAR(50),
                   content_sentiment_score DECIMAL(10,4),
                   sentiment_conflict BOOLEAN DEFAULT FALSE,
                   sentiment_details JSONB, 

                   -- Categories
                   categories TEXT[],
                   primary_category VARCHAR(100),
                   category_confidence DECIMAL(5, 2),
                                                                                                                                  
                   analyzed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                   UNIQUE(article_id)                            
                );
            """)
            # Price analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_analysis (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,

                    -- Price mentions and targets
                    price_mentions DECIMAL(10, 2)[],
                    price_targets DECIMAL(10, 2)[],
                    support_levels DECIMAL(10, 2)[],
                    resistance_levels DECIMAL(10, 2)[],
                    avg_price_target DECIMAL(10, 2),

                    -- Analysis metadata
                    analysis_confidence DECIMAL(5, 2),
                    analyzed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(article_id)                                                                              
                );
            """)
            # Market urgency table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_urgency (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,

                    urgency_score DECIMAL(5, 2),
                    urgency_category VARCHAR(50),
                    is_breaking_news BOOLEAN DEFAULT FALSE,
                    time_horizon TEXT[],
                    market_flags JSONB,

                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(article_id)                                                               
                );
            """)
            # Expert sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expert_mentions (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,

                    expert_sources TEXT[],
                    source_credibility DECIMAL(5,2),
                    expert_quotes TEXT[],

                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(article_id)                                
                );
            """)
            # Scraping Metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scraping_runs (
                    id SERIAL PRIMARY KEY,
                    run_id UUID DEFAULT gen_random_uuid(),
                    spider_name VARCHAR(100),
                    start_time TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMPTZ,
                    articles_scraped INTEGER DEFAULT 0,
                    articles_updated INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    status VARCHAR(50) DEFAULT 'running'                                                 
                );
            """)
            
            # Performance tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS article_performance (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,

                    -- Track how well predictions performed
                    predicted_direction VARCHAR(20), -- 'bullish', 'bearish', 'neutral'
                    actual_price_change_1h DECIMAL(10, 4),
                    actual_price_change_4h DECIMAL(10, 4),
                    actual_price_change_1d DECIMAL(10, 4),                   
                    prediction_accuracy DECIMAL(5, 2),  

                    evaluated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP                                                                   
                );
            """)
            cursor.execute("""
                -- A function that updates the timestamp           
                CREATE OR REPLACE FUNCTION set_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;  
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                -- Attach trigger to my table
                DROP TRIGGER IF EXISTS update_articles_timestamp ON articles;
                                      
                CREATE TRIGGER update_articles_timestamp
                BEFORE UPDATE ON articles
                FOR EACH ROW
                EXECUTE FUNCTION set_updated_at();                                           
            """)
        

            conn.commit()
            self.logger.info("All tables created successfully")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error creating tables: {e}")
            raise
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)    
            
    def create_indexes(self):
        """Create indexes for optimal query performance"""
        conn = self.connection_pool.getconn()
        cursor = conn.cursor()

        try:
            indexes = [
                #Article Indexes
                "CREATE INDEX IF NOT EXISTS idx_articles_published_date ON articles(published_date DESC)",
                "CREATE INDEX IF NOT EXISTS idx_articles_scraped_at ON articles(scraped_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_articles_url_hash ON articles(url_hash)",

                #Sentiment Indexes
                "CREATE INDEX IF NOT EXISTS idx_sentiment_score ON sentiment_analysis(sentiment_score)",
                "CREATE INDEX IF NOT EXISTS idx_sentiment_category ON sentiment_analysis(primary_category)",
                "CREATE INDEX IF NOT EXISTS idx_sentiment_article ON sentiment_analysis(article_id)",

                #Performance Analysis Index
                "CREATE INDEX IF NOT EXISTS idx_price_targets ON price_analysis(avg_price_target)",
                "CREATE INDEX IF NOT EXISTS idx_price_articles ON price_analysis(article_id)",

                #Urgency indexes
                "CREATE INDEX IF NOT EXISTS idx_urgency_score ON market_urgency(urgency_score DESC)",
                "CREATE INDEX IF NOT EXISTS idx_breaking_news ON market_urgency(is_breaking_news) WHERE is_breaking_news = TRUE",

                # Full text search
                "CREATE INDEX IF NOT EXISTS idx_articles_fulltext ON articles USING gin(to_tsvector('english', headline || ' ' || COALESCE(full_text, '')))"
            ]

            for index in indexes:
                cursor.execute(index)

            conn.commit()
            self.logger.info("Indexes created successfully")

        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error creating indexes: {e}")
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

    def create_analysis_views(self):
        """Create views for easier data analysis"""
        conn = self.connection_pool.getconn()
        cursor = conn.cursor()

        try:
            #Comprehensive article view
            cursor.execute("""
                CREATE OR REPLACE VIEW article_analysis_view AS
                SELECT
                    a.id,
                    a.url,
                    a.headline,
                    a.published_date,
                    a.author,
                    s.sentiment,
                    s.sentiment_score,
                    s.primary_category,
                    p.avg_price_target,
                    m.urgency_score,
                    m.is_breaking_news,
                    e.expert_sources
                FROM articles a
                LEFT JOIN sentiment_analysis s ON a.id = s.article_id
                LEFT JOIN price_analysis p ON a.id = p.article_id
                LEFT JOIN market_urgency m ON a.id = m.article_id
                LEFT JOIN expert_mentions e ON a.id = e.article_id
                ORDER BY a.published_date DESC;                                                                                                                             
            """)

            # Daily sentiment summary view
            cursor.execute("""
                CREATE OR REPLACE VIEW daily_sentiment_summary AS
                SELECT
                    DATE(a.published_date) as date,
                    COUNT(*) as article_count,
                    AVG(S.sentiment_score) as avg_sentiment,
                    STDDEV(s.sentiment_score) as sentiment_volatility,
                    COUNT(CASE WHEN s.sentiment = 'bullish' THEN 1 END) as bullish_count,
                    COUNT(CASE WHEN s.sentiment = 'bearish' THEN 1 END) as bearish_count,
                    COUNT(CASE WHEN m.is_breaking_news THEN 1 END) as breaking_news_count
                FROM articles a
                JOIN sentiment_analysis s ON a.id = s.article_id
                LEFT JOIN market_urgency m ON a.id = M.article_id
                WHERE a.published_date IS NOT NULL
                GROUP BY DATE(a.published_date)
                ORDER BY date DESC;                                                                                                                
            """)
             # Top expert mentions view
            cursor.execute("""
                CREATE OR REPLACE VIEW top_expert_mentions AS
                SELECT
                    unnest(expert_sources) as expert,
                    COUNT(*) as mention_count,
                    AVG(s.sentiment_score) as avg_sentiment_when_mentioned
                FROM expert_mentions e
                JOIN sentiment_analysis s ON e.article_id = s.article_id
                WHERE e.expert_sources IS NOT NULL
                GROUP BY expert
                ORDER BY mention_count DESC;                                                                                                                                                                                    
            """)
            conn.commit()
            self.logger.info("Analysis views created successfullt")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error creating views: {e}")
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)        

    def process_item(self, item, spider):
        """Process and save scraped item to PostgreSQL"""
        conn = self.connection_pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            #Generate url hash for faster lookups
            url_hash = hashlib.sha256(item['url'].encode()).hexdigest()

            cursor.execute("SELECT id FROM articles WHERE url_hash = %s", (url_hash,))
            existing_article = cursor.fetchone()

            
            cursor.execute("""
                INSERT INTO articles (url, url_hash, headline, full_text, author, published_date, updated_date, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url_hash)
                DO UPDATE SET      
                    full_text = EXCLUDED.full_text,
                    author = EXCLUDED.author,
                    updated_date = EXCLUDED.updated_date,
                    tags = EXCLUDED.tags,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id;
            """, (
                item.get('url'),
                url_hash,
                item.get('headline'),
                item.get('full_text'),
                item.get('author'),
                item.get('published_datetime'),
                item.get('updated_datetime'),
                item.get('tags', [])
            ))

            article_id = cursor.fetchone()['id']

            if existing_article:
                self.stats['updated'] += 1
                spider.logger.info(f"Updated existing article: {item.get('headline')[:50]}...")
            else:
                self.stats['inserted'] += 1
                spider.logger.info(f"Inserted new article: {item.get('headline')[:50]}...")    

            #Insert Sentiment Analysis
            if item.get('sentiment_score') is not None:
                cursor.execute("""
                    INSERT INTO sentiment_analysis (
                        article_id, sentiment, sentiment_score, headline_sentiment, headline_sentiment_score,
                        content_sentiment, content_sentiment_score, sentiment_conflict, sentiment_details,
                        categories, primary_category, category_confidence                     
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id)
                    DO UPDATE SET
                       sentiment = EXCLUDED.sentiment,
                       sentiment_score = EXCLUDED.sentiment_score,
                       sentiment_details = EXCLUDED.sentiment_details,
                       analyzed_at = CURRENT_TIMESTAMP;                                
                """, (
                    article_id,
                    item.get('sentiment'),
                    item.get('sentiment_score'),
                    item.get('headline_sentiment'),
                    item.get('headline_sentiment_score'),
                    item.get('content_sentiment'),
                    item.get('content_sentiment_score'),
                    item.get('sentiment_conflict', False),
                    Json(item.get('sentiment_details', {})),
                    item.get('categories', []),
                    item.get('primary_category'),
                    item.get('category_confidence')
                ))

                #Insert Price analysis
            if item.get('price_targets') or item.get('support_levels'):
                cursor.execute("""
                    INSERT INTO price_analysis (
                        article_id, price_mentions, price_targets, support_levels, resistance_levels, avg_price_target       
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id)
                    DO UPDATE SET
                        price_targets = EXCLUDED.price_targets,
                        avg_price_target = EXCLUDED.avg_price_target;                           
                """, (
                    article_id,
                    item.get('price_mentions', []),
                    item.get('price_targets', []),
                    item.get('support_levels', []),
                    item.get('resistance_levels', []),
                    item.get('avg_price_target')
                ))

            if item.get('urgency_score') is not None:
                cursor.execute("""
                    INSERT INTO market_urgency (
                        article_id, urgency_score, urgency_category, is_breaking_news, time_horizon, market_flags       
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id)
                    DO UPDATE SET
                        urgency_score = EXCLUDED.urgency_score,
                        market_flags = EXCLUDED.market_flags;                                    
                """, (
                    article_id,
                    item.get('urgency_score'),
                    item.get('urgency_category'),
                    item.get('is_breaking_news', False),
                    item.get('time_horizon', []),
                    Json(item.get('market_flags', {}))
                ))   

            if item.get('expert_sources'):
                cursor.execute("""
                    INSERT INTO expert_mentions (
                        article_id, expert_sources, source_credibility, expert_quotes      
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (article_id)
                    DO UPDATE SET
                        expert_sources = EXCLUDED.expert_sources;                      
                """, (
                    article_id,
                    item.get('expert_sources', []),
                    item.get('source_credibility'),
                    item.get('expert_quotes', [])
                ))
            conn.commit()
            spider.logger.info(f"Saved article: {item.get('headline')[:50]}")

            return item
        except Exception as e:
            conn.rollback()
            self.stats['failed'] += 1
            spider.logger.error(f"Error saving item: {e}")
            raise
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)
    
    def close_spider(self, spider):
        conn = None
        cursor = None

        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # FIX: Use subquery for PostgreSQL compatibility
            cursor.execute("""
                UPDATE scraping_runs 
                SET end_time = CURRENT_TIMESTAMP,
                    status = 'completed',
                    articles_scraped = %s,
                    articles_updated = %s,
                    errors_count = %s
                WHERE id = (
                    SELECT id FROM scraping_runs 
                    WHERE status = 'running' AND spider_name = %s
                    ORDER BY start_time DESC LIMIT 1
                );
            """, (
                self.stats.get('inserted', 0),
                self.stats.get('updated', 0),
                self.stats.get('failed', 0),
                spider.name
            ))
            
            conn.commit()
            
            # Log summary (only if stats exist)
            if hasattr(self, 'stats'):
                total = sum(self.stats.values())
                total = sum(self.stats.values())
                spider.logger.info(f"Stats: Inserted={self.stats.get('inserted')}, Updated={self.stats.get('updated')}, Total={total}")
            
        except Exception as e:
            if conn:
                conn.rollback()   # Undo any half-finished changes
            spider.logger.error(f"Error closing PostgreSQL pipeline: {e}")

        finally:
            #First, close the cursor if it exists
            if cursor:
                cursor.close()
            
            if conn and self.connection_pool:
                self.connection_pool.putconn(conn)

            # Third, close the entire pool
            if self.connection_pool:
                self.connection_pool.closeall()
                spider.logger.info("PostgreSQL pipeline closed successfully")
                






        