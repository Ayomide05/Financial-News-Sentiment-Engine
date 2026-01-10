"""Gold Sentiment Correlation Analysis: This Analyzes relationship between news sentiment and gold price movements"""
import psycopg2
import pandas as pd
import numpy as np
from scipy import stats
from datetime import timedelta
import warnings
from config import DB_CONFIG
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    """Analyzes correlation between sentiment and gold prices"""

    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.data = None

    def load_data(self):
        """Load merged sentiment and price data"""
        query = """
            SELECT
                ds.date, ds.article_count, ds.avg_sentiment_score, ds.bullish_count, ds.bearish_count,
                ds.neutral_count, ds.bullish_ratio, ds.bearish_ratio, ds.avg_urgency_score, ds.breaking_news_count,
                ds.high_urgency_count, ds.sentiment_std, mp.open, mp.high, mp.low, mp.close, mp.volume, mp.daily_return,
                ti.rsi_14, ti.macd, ti.macd_signal, ti.sma_20, ti.sma_50, ti.bollinger_upper, ti.bollinger_lower,
                ti.atr_14, ti.volatility_20d, ti.trend_direction
            FROM daily_sentiment ds
            JOIN market_prices mp ON ds.date = mp.date AND mp.symbol = 'GC=F'
            JOIN technical_indicators ti ON ds.date = ti.date AND ti.symbol = 'GC=F'
            ORDER BY ds.date
        """
        self.data = pd.read_sql(query, self.conn)
        self.data['date'] = pd.to_datetime(self.data['date'])

        # Calculate next-day return (what we want to predict)
        self.data['next_day_return'] = self.data['daily_return'].shift(-1)

        # Calculate 2-day and 5-day forward returns
        self.data['return_2d'] = self.data['close'].pct_change(2).shift(-2)
        self.data['return_5d'] = self.data['close'].pct_change(5).shift(-5)
        # Price direction (binary)
        self.data['price_up'] = (self.data['next_day_return'] > 0).astype(int)
        #Sentiment direction(binary)
        self.data['sentiment_bullish'] = (self.data['avg_sentiment_score'] > 0).astype(int)

        # High Sentiment days
        self.data['high_bullish'] = (self.data['bullish_ratio'] > 0.7).astype(int)
        self.data['high_bearish'] = (self.data['bearish_ratio'] > 0.5).astype(int)

        # Volatility metrics
        self.data['price_range'] = (self.data['high'] - self.data['low']) / self.data['open']
        self.data['abs_return'] = self.data['daily_return'].abs()

        print(f"Loaded {len(self.data)} days of data")
        print(f"Date range: {self.data['date'].min().date()} to {self.data['date'].max().date()}")
        
        return self.data
    
    def basic_correlations(self):
        """Calculate basic correlations between sentiment and returns"""

        correlations = [
            ('avg_sentiment_score', 'next_day_return', 'Sentiment Score -> Next Day Return'),
            ('avg_sentiment_score', 'daily_return', 'Sentiment Score → Same Day Return'),
            ('bullish_ratio', 'next_day_return', 'Bullish Ratio → Next Day Return'),
            ('bearish_ratio', 'next_day_return', 'Bearish Ratio → Next Day Return'),
            ('article_count', 'abs_return', 'Article Count → Absolute Return'),
            ('avg_urgency_score', 'price_range', 'Urgency Score → Price Range'),
            ('breaking_news_count', 'abs_return', 'Breaking News → Absolute Return'),
            ('sentiment_std', 'volatility_20d', 'Sentiment Disagreement → Volatility'),
        ]
        results = []

        for x_col, y_col, description in correlations:
            # Drop NaN values for this pair
            valid_data = self.data[[x_col, y_col]].dropna()
            
            if len(valid_data) < 30:
                continue
            corr, p_value = stats.pearsonr(valid_data[x_col], valid_data[y_col])

            # Significance indicator
            if p_value < 0.01:
                sig = "***"
            elif p_value < 0.05:
                sig = "**"
            elif p_value < 0.10:
                sig = "*"
            else:
                sig = ""
            results.append({
                'description': description,
                'correlation': corr,
                'p_value': p_value,
                'significance': sig,
                'n': len(valid_data)
            })
            print(f"\n{description}")
            print(f"  Correlation: {corr:.4f} {sig}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Sample size: {len(valid_data)}")

        print("\n" + "-"*50)
        print("Significance: *** p<0.01, ** p<0.05, * p<0.10")

        return pd.DataFrame(results)
    
    def predictive_analysis(self):
        """Analyze if sentiment predicts price direction"""

        valid_data = self.data.dropna(subset=['next_day_return', 'avg_sentiment_score'])

        # When sentiment is bullish, does price go up
        bullish_days = valid_data[valid_data['sentiment_bullish'] == 1]
        bearish_days = valid_data[valid_data['sentiment_bullish'] == 0]

        bullish_win_rate = bullish_days['price_up'].mean() * 100
        bearish_win_rate = (1 - bearish_days['price_up'].mean()) * 100

        print("\n SENTIMENT DIRECTION ACCURACY")
        print("-"*50)
        print(f"When sentiment is BULLISH ({len(bullish_days)} days):")
        print(f"  → Price went UP: {bullish_win_rate:.1f}% of the time")
        print(f"  → Avg next-day return: {bullish_days['next_day_return'].mean()*100:.3f}%")
        
        print(f"\nWhen sentiment is BEARISH ({len(bearish_days)} days):")
        print(f"  → Price went DOWN: {bearish_win_rate:.1f}% of the time")
        print(f"  → Avg next-day return: {bearish_days['next_day_return'].mean()*100:.3f}%")

        # High Conviction signals
        high_bullish = valid_data[valid_data['bullish_ratio'] > 0.7]
        high_bearish = valid_data[valid_data['bearish_ratio'] > 0.5]

        if len(high_bullish) > 10:
            print(f"\nStrong Bullish Days (>70% bullish articles): {len(high_bullish)} days")
            print(f"  → Avg next-day return: {high_bullish['next_day_return'].mean()*100:.3f}%")
            print(f"  → Win rate: {(high_bullish['price_up'].mean()*100):.1f}%")

        if len(high_bearish) > 10:
           print(f"\nStrong Bearish Days (>50% bearish articles): {len(high_bearish)} days")
           print(f"  → Avg next-day return: {high_bearish['next_day_return'].mean()*100:.3f}%")
           print(f"  → Win rate (price down): {((1-high_bearish['price_up'].mean())*100):.1f}%")

        # Statistical significance test
        # T-test: Are returns different on bullish vs bearish days

        t_stat, t_pvalue = stats.ttest_ind(
            bullish_days['next_day_return'].dropna(),
            bearish_days['next_day_return'].dropna()
        )
        print(f"T-test (bullish vs bearish days):")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {t_pvalue:.4f}")
        
        if t_pvalue < 0.05:
            print("  SIGNIFICANT: Returns ARE different on bullish vs bearish days!")
        else:
            print("  NOT significant at 95% confidence level")

    def regime_analysis(self):
        """Analyze sentiment effectiveness in different market regimes"""
        
        valid_data = self.data.dropna(subset=['next_day_return', 'rsi_14', 'trend_direction'])

        # RSI Regimes
        oversold = valid_data[valid_data['rsi_14'] < 30]
        neutral_rsi = valid_data[(valid_data['rsi_14'] >= 30) & (valid_data['rsi_14'] <= 70)]
        overbought = valid_data[valid_data['rsi_14'] > 70]

        for name, regime_data in [('Oversold (RSI<30)', oversold),
                                  ('Neutral (30-70)', neutral_rsi),
                                  ('Overbought (RSI>70)', overbought)]:
            if len(regime_data) < 10:
                continue

            bullish_in_regime = regime_data[regime_data['sentiment_bullish'] == 1]
            if len(bullish_in_regime) > 5:
                corr = regime_data['avg_sentiment_score'].corr(regime_data['next_day_return'])
                win_rate = bullish_in_regime['price_up'].mean() * 100
                print(f"\n{name} ({len(regime_data)} days):")
                print(f"  Sentiment-Return Correlation: {corr:.4f}")
                print(f"  Bullish sentiment win rate: {win_rate:.1f}%")

        for trend in ['bullish', 'bearish', 'neutral']:
            trend_data = valid_data[valid_data['trend_direction'] == trend]
            if len(trend_data) < 20:
                continue

            corr = trend_data['avg_sentiment_score'].corr(trend_data['next_day_return'])
            bullish_in_trend = trend_data[trend_data['sentiment_bullish'] == 1]
            win_rate = bullish_in_trend['price_up'].mean() * 100 if len(bullish_in_trend) > 0 else 0

            print(f"\n{trend.upper()} Trend ({len(trend_data)} days):")
            print(f"  Sentiment-Return Correlation: {corr:.4f}")
            print(f"  Bullish sentiment win rate: {win_rate:.1f}%")   

    def urgency_analysis(self):
        """Analyze if urgency/breaking news predictd volatility"""

        valid_data = self.data.dropna(subset=['avg_urgency_score', 'abs_return'])

        # Breaking news impact
        breaking_days = valid_data[valid_data['breaking_news_count'] > 0]
        normal_days = valid_data[valid_data['breaking_news_count'] == 0]

        print(f"Days with breaking news: {len(breaking_days)}")
        print(f"Days without breaking news: {len(normal_days)}")

        if len(breaking_days) > 10:
            print(f"\nAvg absolute return on breaking news days: {breaking_days['abs_return'].mean()*100:.3f}%")
            print(f"Avg absolute return on normal days: {normal_days['abs_return'].mean()*100:.3f}%")

            # T-test for volatility difference
            t_stat, p_val = stats.ttest_ind(
                breaking_days['abs_return'].dropna(),
                normal_days['abs_return'].dropna()
            )
            print(f"\nT-test p-value: {p_val:.4f}")
            if p_val < 0.05:
                print("Breaking news DOES predict higher volatility!")
            else:
                print("No significant difference in volatility")
        # High urgency analysis
        high_urgency = valid_data[valid_data['high_urgency_count'] > 0]
        low_urgency = valid_data[valid_data['high_urgency_count'] == 0]

        if len(high_urgency) > 10:
            print(f"High urgency days: {len(high_urgency)}")
            print(f"Avg price range: {high_urgency['price_range'].mean()*100:.3f}%")
            print(f"Avg price range (normal): {low_urgency['price_range'].mean()*100:.3f}%")
    
    def generate_trading_signals(self):

        valid_data = self.data.dropna(subset=['next_day_return', 'bullish_ratio', 'rsi_14'])

        # Strategy 1: High bullish ratio
        signal_days = valid_data[valid_data['bullish_ratio'] > 0.7]
        if len(signal_days) > 10:
            avg_return = signal_days['next_day_return'].mean() * 100
            win_rate = (signal_days['next_day_return'] > 0).mean() * 100
            sharpe = signal_days['next_day_return'].mean() / signal_days['next_day_return'].std() * np.sqrt(252)

            print(f"Signal days: {len(signal_days)}")
            print(f"Avg next-day return: {avg_return:.3f}%")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Annualized Sharpe: {sharpe:.2f}")
        
        # Strategy 2: Bullish sentiment + Oversold RSI
        signal_days = valid_data[(valid_data['sentiment_bullish'] == 1) & (valid_data['rsi_14'] < 40)]
        if len(signal_days) > 10:
            avg_return = signal_days['next_day_return'].mean() * 100
            win_rate = (signal_days['next_day_return'] > 0).mean() * 100
            sharpe = signal_days['next_day_return'].mean() / signal_days['next_day_return'].std() * np.sqrt(252)

            print(f"Signal days: {len(signal_days)}")
            print(f"Avg next-day return: {avg_return:.3f}%")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"Annualized Sharpe: {sharpe:.2f}")

        # Contrarian - Bearish sentiment but oversold
        signal_days = valid_data[(valid_data['sentiment_bullish'] == 0) & (valid_data['rsi_14'] < 30)]
        if len(signal_days) > 5:
            avg_return = signal_days['next_day_return'].mean() * 100
            win_rate = (signal_days['next_day_return'] > 0).mean() * 100

            print(f"Signal days: {len(signal_days)}")
            print(f"Avg next-day return: {avg_return:.3f}%")
            print(f"Win rate: {win_rate:.1f}%")
        else:
            print("Not enough data points for this strategy")
    def summary_report(self):
        """Generate executive summary of findings"""
        
        print("\n" + "="*70)
        print(" EXECUTIVE SUMMARY")
        print("="*70)
        
        valid_data = self.data.dropna(subset=['next_day_return', 'avg_sentiment_score'])
        
        # Key metrics
        overall_corr = valid_data['avg_sentiment_score'].corr(valid_data['next_day_return'])
        bullish_days = valid_data[valid_data['sentiment_bullish'] == 1]
        bearish_days = valid_data[valid_data['sentiment_bullish'] == 0]
        
        bullish_return = bullish_days['next_day_return'].mean() * 100
        bearish_return = bearish_days['next_day_return'].mean() * 100
        spread = bullish_return - bearish_return
        
        print(f"""
            Dataset: {len(valid_data)} trading days analyzed
            Period: {valid_data['date'].min().date()} to {valid_data['date'].max().date()}                
                                                                  
            SENTIMENT → PRICE CORRELATION                                   
            ─────────────────────────────                                   
            Overall correlation: {overall_corr:.4f}                                   
                                                                  
            RETURN BY SENTIMENT                                             
            ───────────────────                                             
            Bullish days ({len(bullish_days)}): {bullish_return:+.3f}% avg next-day return        
            Bearish days ({len(bearish_days)}): {bearish_return:+.3f}% avg next-day return        
            Spread: {spread:+.3f}%                                                
                                                                  
            SIGNAL STRENGTH                                                 
            ───────────────                                                 
            Bullish sentiment win rate: {bullish_days['price_up'].mean()*100:.1f}%                      
            Overall market win rate: {valid_data['price_up'].mean()*100:.1f}%                         
        """)
        
        # Interpretation
        print("\n INTERPRETATION")
        print("-"*50)
        
        if overall_corr > 0.1:
            print("POSITIVE correlation: Bullish sentiment tends to precede price increases")
        elif overall_corr < -0.1:
            print("NEGATIVE correlation: Sentiment may be a contrarian indicator")
        else:
            print("WEAK correlation: Sentiment alone is not strongly predictive")
        
        if spread > 0.05:
            print(f"ACTIONABLE: {spread:.3f}% return spread between bullish/bearish days")
        else:
            print("Small return spread - signal may not be tradeable after costs")
        
        win_rate = bullish_days['price_up'].mean() * 100
        if win_rate > 55:
            print(f"WIN RATE: {win_rate:.1f}% is above random chance (50%)")
        else:
            print(f"WIN RATE: {win_rate:.1f}% is close to random chance")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("   GOLD SENTIMENT CORRELATION ANALYSIS")
                
        # Load data
        self.load_data()
        
        # Run all analyses
        self.basic_correlations()
        self.predictive_analysis()
        self.regime_analysis()
        self.urgency_analysis()
        self.generate_trading_signals()
        self.summary_report()

        print("\n" + "="*70)
        print(" ANALYSIS COMPLETE!")
           
        return self.data
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
def main():
    analyzer = CorrelationAnalyzer()
    
    try:
        data = analyzer.run_full_analysis()
        
        # Save data for visualization
        data.to_csv('correlation_data.csv', index=False)
        print("\n Data saved to correlation_data.csv")
        
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
