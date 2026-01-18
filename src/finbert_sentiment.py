"""FinBERT Sentiment Analysis - Deep Learning-based sentiment for financial text"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import psycopg2
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from config import DB_CONFIG

warnings.filterwarnings('ignore')

class FinBERTAnalyzer:
    """FinBERT-based sentiment analysis for financial text"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading FinBERT model (this may take a minute)...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.to(self.device)
        self.model.eval()
        print("FinBERT loaded successfully!")

        # Label mapping
        self.labels = ['negative', 'neutral', 'positive']

        # Database connection
        self.conn = psycopg2.connect(**DB_CONFIG)

    def analyze_text(self, text):
        """Analyze single text and return sentiment"""
        if not text or len(text.strip()) == 0:
            return {
                'sentiment': 'neutral',
                'positive_prob': 0.33,
                'negative_prob': 0.33,
                'neutral_prob': 0.34,
                'confidence': 0.34,
                'score': 0.0
            }
        # Truncate long text
        text = text[:512]
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        #inputs = {k: v.to(self.device) for k, v in inputs.items()}
        new_inputs = {}
        # Open the 'inputs' bag and look at every item inside
        for k, v in inputs.items():
            # Take the data (v) and move it to the GPU
            v_on_gpu = v.to(self.device)
            # Put it into the new bag with the same label (k)
            new_inputs[k] = v_on_gpu
        inputs = new_inputs            
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]

        # Extract results
        neg_prob, neu_prob, pos_prob = probs
        sentiment_idx = np.argmax(probs)
        sentiment = self.labels[sentiment_idx]
        confidence = probs[sentiment_idx]

        # Calculate continious score (-1 to +1)
        score = pos_prob - neg_prob

        return {
            'sentiment': sentiment,
            'positive_prob': float(pos_prob),
            'negative_prob': float(neg_prob),
            'neutral_prob': float(neu_prob),
            'confidence': float(confidence),
            'score': float(score)
        }

    def analyze_batch(self, texts, batch_size=16):
        """Analyze multiple texts efficiently"""
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing"):
            batch_texts = texts[i:i+batch_size]

            # Filter empty texts
            batch_texts = [t if t and len(str(t).strip()) > 0 else "neutral" for t in batch_texts]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors = "pt",
                truncation = True,
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs.cpu().numpy()
            
            # Process results
            for j, prob in enumerate(probs):
                neg_prob, neu_prob, pos_prob = prob
                sentiment_idx = np.argmax(prob)

                results.append({
                    'sentiment': self.labels[sentiment_idx],
                    'positive_prob': float(pos_prob),
                    'negative_prob': float(neg_prob),
                    'neutral_prob': float(neu_prob),
                    'confidence': float(prob[sentiment_idx]),
                    'score': float(pos_prob - neg_prob)
                })

        return results
    
    def create_finbert_table(self):
        """Create table to store FinBERT results"""
        cur = self.conn.cursor()

        cur.execute('''
            CREATE TABLE IF NOT EXISTS finbert_sentiment (
                id SERIAL PRIMARY KEY,
                article_id INTEGER REFERENCES articles(id),
                headline_sentiment VARCHAR(20),
                headline_score DECIMAL(6, 4),
                headline_positive DECIMAL(6, 4),
                headline_negative DECIMAL(6, 4),
                headline_neutral DECIMAL(6, 4),
                headline_confidence DECIMAL(6, 4),
                content_sentiment VARCHAR(20),
                content_score DECIMAL(6, 4),
                content_positive DECIMAL(6, 4),
                content_negative DECIMAL(6, 4),
                content_neutral DECIMAL(6, 4),
                content_confidence DECIMAL(6, 4),
                combined_score DECIMAL(6, 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(article_id)
            )
        ''')

        self.conn.commit()
        print(" finbert_sentiment table created")

    def analyze_all_articles(self):
        """Analyze all artucles with FinBERT"""
        print(" FINBERT SENTIMENT ANALYSIS")

        # Create table
        self.create_finbert_table()
        # Load articles
        cur = self.conn.cursor()
        cur.execute('''
            SELECT id, headline, full_text
            FROM articles
            WHERE headline IS NOT NULL
            ORDER BY id
        ''')
        articles = cur.fetchall()

        print(f"\n Analyzing {len(articles)} articles...")

        # Process in batches
        batch_size = 16
        results = []

        for i in tqdm(range(0, len(articles), batch_size), desc="Processing articles"):
            batch = articles[i:i+batch_size]

            for article_id, headline, content in batch:
                # Analyze headline
                headline_result = self.analyze_text(headline)

                # Analyze content (first 512 chars for speed)
                if content:
                    content_text = content[:1000]
                    content_result = self.analyze_text(content_text)
                else:
                    content_result = headline_result

                # Combined score (weighted: 40% headline, 60% content)
                combined_score = 0.4 * headline_result['score'] + 0.6 * content_result['score']

                results.append({
                    'article_id': article_id,
                    'headline_sentiment': headline_result['sentiment'],
                    'headline_score': headline_result['score'],
                    'headline_positive': headline_result['positive_prob'],
                    'headline_negative': headline_result['negative_prob'],
                    'headline_neutral': headline_result['neutral_prob'],
                    'headline_confidence': headline_result['confidence'],
                    'content_sentiment': content_result['sentiment'],
                    'content_score': content_result['score'],
                    'content_positive': content_result['positive_prob'],
                    'content_negative': content_result['negative_prob'],
                    'content_neutral': content_result['neutral_prob'],
                    'content_confidence': content_result['confidence'],
                    'combined_score': combined_score
                })
        
        # Save to database
        print("\n Saving results to database...")
        self._save_results(results)

        print("\n FinBERT analysis complete!")
        return results
    
    def _save_results(self, results):
        """Save results to database"""
        cur = self.conn.cursor()

        for r in tqdm(results, desc="Saving"):
            cur.execute('''
                INSERT INTO finbert_sentiment (
                    article_id, headline_sentiment, headline_score, headline_positive, headline_negative,
                    headline_neutral, headline_confidence, content_sentiment, content_score, content_positive,
                    content_negative, content_neutral, content_confidence, combined_score
                ) VALUES (%s, %s, %s, %s,%s, %s,%s, %s,%s, %s,%s, %s,%s, %s)
                ON CONFLICT (article_id) DO UPDATE SET
                    headline_sentiment = EXCLUDED.headline_sentiment,
                    headline_score = EXCLUDED.headline_score,
                    headline_positive = EXCLUDED.headline_positive,
                    headline_negative = EXCLUDED.headline_negative,
                    headline_neutral = EXCLUDED.headline_neutral,
                    headline_confidence = EXCLUDED.headline_confidence,
                    content_sentiment = EXCLUDED.content_sentiment,
                    content_score = EXCLUDED.content_score,
                    content_positive = EXCLUDED.content_positive,
                    content_negative = EXCLUDED.content_negative,
                    content_neutral = EXCLUDED.content_neutral,
                    content_confidence = EXCLUDED.content_confidence,
                    combined_score = EXCLUDED.combined_score
            ''', (
                r['article_id'], r['headline_sentiment'], r['headline_score'],
                r['headline_positive'], r['headline_negative'], r['headline_neutral'], r['headline_confidence'],
                r['content_sentiment'], r['content_score'],
                r['content_positive'], r['content_negative'], r['content_neutral'], r['content_confidence'],
                r['combined_score']
            ))
        self.conn.commit()
        print("All results saved!")

    def comprae_with_rulebased(self):
        """Compare FinBERT with rule-based sentiment"""
        print("COMPARING FINBERT vs RULE-BASED")

        query = """
            SELECT
                a.id,
                a.headline,
                s.sentiment as rulebased_sentiment,
                s.sentiment_score as rulebased_score,
                f.headline_sentiment as finbert_sentiment,
                f.combined_score as finbert_score
            FROM articles a
            JOIN sentiment_analysis s ON a.id = s.article_id
            JOIN finbert_sentiment f ON a.id = f.article_id
        """

        df = pd.read_sql(query, self.conn)

        # Compare sentiments
        def categorize_rulebased(score):
            if score > 5:
                return 'positive'
            elif score < -5:
                return 'negative'
            else: 
                return 'neutral'
            
        df['rulebased_category'] = df['rulebased_score'].apply(categorize_rulebased)
        df['agreement'] = df['rulebased_category'] == df['finbert_sentiment']

        agreement_rate = df['agreement'].mean() * 100

        print(f"\n Agreement Rate: {agreement_rate:.1f}%")
        print(f"   (How often both methods agree)")

        # Correlation
        correlation = df['rulebased_score'].corr(df['finbert_score'])
        print(f"\n Score Correlation: {correlation:.4f}")

        # Distribution comparison
        print("\n Sentiment Distribution:")
        print("\n   Rule-based:")
        rb_dist = df['rulebased_category'].value_counts(normalize=True) * 100
        for cat, pct in rb_dist.items():
            print(f"      {cat}: {pct:.1f}%")
        
        print("\n   FinBERT:")
        fb_dist = df['finbert_sentiment'].value_counts(normalize=True) * 100
        for cat, pct in fb_dist.items():
            print(f"      {cat}: {pct:.1f}%")
    
        # Examples of disagreement
        print("\n Examples where methods DISAGREE:")
        print("-"*70)
        
        disagreements = df[~df['agreement']].head(5)
        for _, row in disagreements.iterrows():
            print(f"\nHeadline: {row['headline'][:80]}...")
            print(f"   Rule-based: {row['rulebased_category']} (score: {row['rulebased_score']:.1f})")
            print(f"   FinBERT:    {row['finbert_sentiment']} (score: {row['finbert_score']:.3f})")
        
        return df
    
    def update_daily_sentiment(self):
        """Create updated daily sentiment using FinBERT scores"""
        print(" UPDATING DAILY SENTIMENT WITH FINBERT")
                
        cur = self.conn.cursor()
        
        # Create new table for FinBERT daily sentiment
        cur.execute('''
            CREATE TABLE IF NOT EXISTS daily_sentiment_finbert (
                date DATE PRIMARY KEY,
                article_count INTEGER,
                avg_finbert_score DECIMAL(6,4),
                avg_headline_score DECIMAL(6,4),
                avg_content_score DECIMAL(6,4),
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                positive_ratio DECIMAL(5,4),
                negative_ratio DECIMAL(5,4),
                avg_confidence DECIMAL(6,4),
                score_std DECIMAL(6,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Aggregate by day
        cur.execute('''
            INSERT INTO daily_sentiment_finbert (
                date, article_count, avg_finbert_score, avg_headline_score,
                avg_content_score, positive_count, negative_count, neutral_count,
                positive_ratio, negative_ratio, avg_confidence, score_std
            )
            SELECT 
                DATE(a.published_date) as date,
                COUNT(*) as article_count,
                AVG(f.combined_score) as avg_finbert_score,
                AVG(f.headline_score) as avg_headline_score,
                AVG(f.content_score) as avg_content_score,
                SUM(CASE WHEN f.headline_sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN f.headline_sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
                SUM(CASE WHEN f.headline_sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                CAST(SUM(CASE WHEN f.headline_sentiment = 'positive' THEN 1 ELSE 0 END) AS DECIMAL) / 
                    NULLIF(COUNT(*), 0) as positive_ratio,
                CAST(SUM(CASE WHEN f.headline_sentiment = 'negative' THEN 1 ELSE 0 END) AS DECIMAL) / 
                    NULLIF(COUNT(*), 0) as negative_ratio,
                AVG(f.headline_confidence) as avg_confidence,
                STDDEV(f.combined_score) as score_std
            FROM articles a
            JOIN finbert_sentiment f ON a.id = f.article_id
            WHERE a.published_date IS NOT NULL
            GROUP BY DATE(a.published_date)
            ON CONFLICT (date) DO UPDATE SET
                article_count = EXCLUDED.article_count,
                avg_finbert_score = EXCLUDED.avg_finbert_score,
                avg_headline_score = EXCLUDED.avg_headline_score,
                avg_content_score = EXCLUDED.avg_content_score,
                positive_count = EXCLUDED.positive_count,
                negative_count = EXCLUDED.negative_count,
                neutral_count = EXCLUDED.neutral_count,
                positive_ratio = EXCLUDED.positive_ratio,
                negative_ratio = EXCLUDED.negative_ratio,
                avg_confidence = EXCLUDED.avg_confidence,
                score_std = EXCLUDED.score_std
        ''')
        
        self.conn.commit()
        
        # Show summary
        cur.execute('SELECT COUNT(*), MIN(date), MAX(date) FROM daily_sentiment_finbert')
        count, min_date, max_date = cur.fetchone()
        
        print(f"\n Daily FinBERT sentiment aggregated:")
        print(f"   Days: {count}")
        print(f"   Range: {min_date} to {max_date}")

    def close(self):
        """Close connections"""
        if self.conn:
            self.conn.close()

def main():
    analyzer = FinBERTAnalyzer()

    try:
        # Analyze all articles
        analyzer.analyze_all_articles()
        # Compare with rule-based
        analyzer.comprae_with_rulebased()
        #Update daily sentiment
        analyzer.update_daily_sentiment()
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()