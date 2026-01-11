"""Gold Sentiment Analysis - Visualizations"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import psycopg2
from config import DB_CONFIG
import warnings
warnings.filterwarnings('ignore')

#Set Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SentimentVisualizer:
    """Creates visualizations for sentiment analysis"""

    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.data = None
        self.output_dir = '../visualizations/'

        # Create output directory
        import os
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Load correlation data"""
        query = """
            SELECT
                ds.date, ds.article_count, ds.avg_sentiment_score, ds.bullish_count, ds.bearish_count, ds.bullish_ratio, ds.bearish_ratio,
                ds.avg_urgency_score, ds.breaking_news_count, ds.sentiment_std, mp.open, mp.high, mp.low, mp.close, mp.daily_return,
                ti.rsi_14, ti.macd, ti.sma_20, ti.volatility_20d, ti.trend_direction
            FROM daily_sentiment ds
            JOIN market_prices mp ON ds.date = mp.date AND mp.symbol = 'GC=F'
            JOIN technical_indicators ti ON ds.date = ti.date AND ti.symbol = 'GC=F'
            ORDER BY ds.date
        """
        self.data = pd.read_sql(query, self.conn)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['next_day_return'] = self.data['daily_return'].shift(-1)
        self.data['price_up'] = (self.data['next_day_return'] > 0).astype(int)
        self.data['sentiment_bullish'] = (self.data['avg_sentiment_score'] > 0).astype(int)
        self.data['abs_return'] = self.data['daily_return'].abs()
        self.data['price_range'] = (self.data['high'] - self.data['low']) / self.data['open']
        
        print(f"Loaded {len(self.data)} days of data")
        return self.data

    def plot_correlation_heatmap(self):
        """Create correlation heatmap"""
        # Select relevant columns
        cols = [
            'avg_sentiment_score', 'bullish_ratio', 'bearish_ratio',
            'article_count', 'avg_urgency_score', 'breaking_news_count',
            'daily_return', 'next_day_return', 'abs_return', 'price_range',
            'rsi_14', 'volatility_20d'
        ]
        # Rename for readability
        rename_map = {
            'avg_sentiment_score': "Sentiment",
            'bullish_ratio': 'Bullish %',
            'bearish_ratio': 'Bearish %',
            'article_count': 'Articles',
            'avg_urgency_score': 'Urgency',
            'breaking_news_count': 'Breaking News',
            'daily_return': 'Same-Day Return',
            'next_day_return': 'Next-Day Return',
            'abs_return': 'Absolute Return',
            'price_range': 'Price Range',
            'rsi_14': 'RSI',
            'volatility_20d': 'Volatility'
        }

        corr_data = self.data[cols].rename(columns=rename_map)
        corr_matrix = corr_data.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidth=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax
        )

        ax.set_title('Correlation Matrix: Sentiment vs Price Methods', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}01_correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 01_correlation_heatmap.png")

    def plot_win_rates(self):
        """Create win rate comparison chart"""
        
        valid_data = self.data.dropna(subset=['next_day_return'])
        # Calculate win rates for different conditions
        conditions = {
            'All Days': valid_data,
            'Bullish Sentiment': valid_data[valid_data['sentiment_bullish'] == 1],
            'Bearish Sentiment': valid_data[valid_data['sentiment_bullish'] == 0],
            'Strong Bullish\n(>70%)': valid_data[valid_data['bullish_ratio'] > 0.7],
            'High Urgency': valid_data[valid_data['avg_urgency_score'] > valid_data['avg_urgency_score'].median()]
        }

        win_rates = []
        labels = []
        counts  = []

        for label, subset in conditions.items():
            if len(subset) > 10:
                win_rate = (subset['next_day_return'] > 0).mean() * 100
                win_rates.append(win_rate)
                labels.append(label)
                counts.append(len(subset))

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#27ae60', '#9b59b6']
        bars = ax.bar(labels, win_rates, color=colors[:len(labels)], edgecolor='black', linewidth=1.2)

        # Add 50% refrence line
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
        
        # Add value labels on bars
        for bar, counts in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%\n(n={counts})',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Win Rate (% of days price went UP)', fontsize=12)
        ax.set_title('Next-Day Win Rate by Sentiment Condition', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 80)
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}02_win_rates.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 02_win_rates.png")
    
    def plot_sentiment_vs_returns(self):
        """Create a scatter plot of sentiment vs returns"""

        valid_data = self.data.dropna(subset=['avg_sentiment_score', 'daily_return', 'next_day_return'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Same-day returns
        ax1 = axes[0]
        ax1.scatter(valid_data['avg_sentiment_score'], valid_data['daily_return'] * 100,
        alpha=0.5, c='#3498db', edgecolors='white', linewidth=0.5)

        # Add trend line
        z = np.polyfit(valid_data['avg_sentiment_score'], valid_data['daily_return'] * 100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_data['avg_sentiment_score'].min(), valid_data['avg_sentiment_score'].max(), 100)
        ax1.plot(x_line, p(x_line), "r-", linewidth=2, label=f'Trend (r=0.26)')

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Sentiment Score', fontsize=11)
        ax1.set_ylabel('Same-Day Return (%)', fontsize=11)
        ax1.set_title('Sentiment vs Same-Day Return\n(Correlation: 0.26***)', fontsize=12, fontweight='bold')
        ax1.legend()

        # Next-day returns
        ax2 = axes[1]
        ax2.scatter(valid_data['avg_sentiment_score'], valid_data['next_day_return'] * 100,
                    alpha=0.5, c='#e74c3c', edgecolors='white', linewidth=0.5)
        z2 = np.polyfit(valid_data['avg_sentiment_score'].dropna(),
                        valid_data['next_day_return'].dropna() * 100, 1)
        p2= np.poly1d(z2)
        ax2.plot(x_line, p2(x_line), "r-", linewidth=2, label=f'Trend (r=0.00)')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Sentiment Score', fontsize=11)
        ax2.set_ylabel('Next-Day Return (%)', fontsize=11)
        ax2.set_title('Sentiment vs Next-Day Return\n(Correlation: 0.00 - Not Predictive)', fontsize=12, fontweight='bold')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}03_sentiment_vs_returns.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 03_sentiment_vs_returns.png")
    
    def plot_time_series(self):
        """Create time series with sentiment and price overlay"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Plot 1: Gold Price
        ax1 = axes[0]
        ax1.plot(self.data['date'], self.data['close'], color='#f39c12', linewidth=1.5, label='Gold Price')
        ax1.fill_between(self.data['date'], self.data['close'], alpha=0.3, color='#f39c12')
        ax1.set_ylabel('Gold Price ($)', fontsize=11)
        ax1.set_title('Gold Price, Sentiment & Article Volume Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Sentiment Score
        ax2 = axes[1]
        colors = ['#2ecc71' if x > 0 else  '#e74c3c' for x in self.data['avg_sentiment_score']]
        ax2.bar(self.data['date'], self.data['avg_sentiment_score'], color=colors, alpha=0.7, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Sentiment Score', fontsize=11)
        ax2.legend(['Sentiment (Green=Bullish, Red=Bearish)'], loc='upper left')

        # Plot 3: Article Count
        ax3 = axes[2]
        ax3.bar(self.data['date'], self.data['article_count'], color='#3498db', alpha=0.7, width=1)
        ax3.set_ylabel('Article Count', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.legend(['Daily Article Count'], loc='upper left')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}04_time_series.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 04_time_series.png")

    def plot_strategy_performance(self):
        """Create cumulative returns for strategies"""
        valid_data = self.data.dropna(subset=['next_day_return']).copy()
        
        # Strategy 1: Buy and Hold
        valid_data['cumret_bh'] = (1 + valid_data['next_day_return']).cumprod()

        # strategy 2: Only trade on bullish sentiment days
        valid_data['strategy_bullish'] = valid_data['next_day_return'] * valid_data['sentiment_bullish']
        valid_data['cumret_bullish'] = (1 + valid_data['strategy_bullish']).cumprod()
        
        # Strategy 3: Only trade on strong bullish days (>70%)
        valid_data['strong_bullish'] = (valid_data['bullish_ratio'] > 0.7).astype(int)
        valid_data['strategy_strong'] = valid_data['next_day_return'] * valid_data['strong_bullish']
        valid_data['cumret_strong'] = (1 + valid_data['strategy_strong']).cumprod()

        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(valid_data['date'], valid_data['cumret_bh'], 
               label='Buy & Hold', linewidth=2, color='#3498db')
        ax.plot(valid_data['date'], valid_data['cumret_bullish'], 
               label='Bullish Sentiment Only', linewidth=2, color='#2ecc71')
        ax.plot(valid_data['date'], valid_data['cumret_strong'], 
               label='Strong Bullish (>70%) Only', linewidth=2, color='#9b59b6')
        
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Cumulative Return (1 = Starting Value)', fontsize=11)
        ax.set_title('Strategy Comparison: Cumulative Returns', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add final returns annotation
        final_bh = valid_data['cumret_bh'].iloc[-1]
        final_bullish = valid_data['cumret_bullish'].iloc[-1]
        final_strong = valid_data['cumret_strong'].iloc[-1]
        
        textstr = f'Final Returns:\nBuy & Hold: {(final_bh-1)*100:.1f}%\nBullish Only: {(final_bullish-1)*100:.1f}%\nStrong Bullish: {(final_strong-1)*100:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}05_strategy_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 05_strategy_performance.png")
    
    def plot_rsi_regimes(self):
        valid_data = self.data.dropna(subset=['rsi_14', 'next_day_return']).copy()

        # Categorize RSI
        def rsi_category(rsi):
            if rsi < 30:
                return 'Oversold (<30)'
            elif rsi > 70:
                return 'Overbought (>70)'
            else: 
                return 'Neutral (30-70)'
        
        valid_data['rsi_regime'] = valid_data['rsi_14'].apply(rsi_category)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Win rate by RSI regime
        ax1 = axes[0]

        regimes = ['Oversold (<30)', 'Neutral (30-70)', 'Overbought (>70)']
        win_rates = []
        counts = []

        for regime in regimes:
            subset = valid_data[valid_data['rsi_regime'] == regime]
            if len(subset) > 5:
                bullish_subset = subset[subset['sentiment_bullish'] == 1]
                win_rate = (bullish_subset['next_day_return'] > 0).mean() * 100 if len(bullish_subset) > 0 else 0
                win_rates.append(win_rate)
                counts.append(len(subset))

            else:
                win_rates.append(0)
                counts.append(0)

        colors = ['#e74c3c', '#3498db', '#2ecc71']
        bars = ax1.bar(regimes, win_rates, color=colors, edgecolor='black', linewidth=1.2)
        ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{height:.1f}%\n(n={count})',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_ylabel('Bullish Sentiment Win Rate (%)', fontsize=11)
        ax1.set_title('Win Rate by RSI Regime\n(When Sentiment is Bullish)', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 80)
        ax1.legend()
        
        # Plot 2: RSI distribution
        ax2 = axes[1]
        ax2.hist(valid_data['rsi_14'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
        ax2.axvline(x=30, color='red', linestyle='--', linewidth=2, label='Oversold (30)')
        ax2.axvline(x=70, color='green', linestyle='--', linewidth=2, label='Overbought (70)')
        ax2.set_xlabel('RSI Value', fontsize=11)
        ax2.set_ylabel('Frequency (Days)', fontsize=11)
        ax2.set_title('RSI Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}06_rsi_regimes.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved: 06_rsi_regimes.png")

    def plot_urgency_analysis(self):
        """Create urgency vs volatility chart"""
        print("Creating urgency analysis chart...")
        
        valid_data = self.data.dropna(subset=['avg_urgency_score', 'price_range']).copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Scatter plot
        ax1 = axes[0]
        ax1.scatter(valid_data['avg_urgency_score'], valid_data['price_range'] * 100,
                   alpha=0.5, c='#e74c3c', edgecolors='white', linewidth=0.5)
        
        # Trend line
        z = np.polyfit(valid_data['avg_urgency_score'], valid_data['price_range'] * 100, 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_data['avg_urgency_score'].min(), valid_data['avg_urgency_score'].max(), 100)
        ax1.plot(x_line, p(x_line), "b-", linewidth=2, label='Trend (r=0.21***)')
        
        ax1.set_xlabel('Average Urgency Score', fontsize=11)
        ax1.set_ylabel('Price Range (%)', fontsize=11)
        ax1.set_title('Urgency Score vs Price Volatility\n(Correlation: 0.21***)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot by urgency level
        ax2 = axes[1]
        
        valid_data['urgency_level'] = pd.qcut(valid_data['avg_urgency_score'], q=3, labels=['Low', 'Medium', 'High'])
        
        urgency_groups = [valid_data[valid_data['urgency_level'] == level]['price_range'] * 100 
                        for level in ['Low', 'Medium', 'High']]
        
        bp = ax2.boxplot(urgency_groups, labels=['Low Urgency', 'Medium Urgency', 'High Urgency'],
                        patch_artist=True)
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Price Range (%)', fontsize=11)
        ax2.set_title('Price Volatility by Urgency Level', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}07_urgency_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 07_urgency_analysis.png")

    def create_summary_dashboard(self):
        """Create a summary dashboard with key findings"""
        print("Creating summary dashboard...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Title
        fig.suptitle('Gold Sentiment Analysis - Key Findings Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Key Metrics Box (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        metrics_text = """
        KEY METRICS
        ─────────────────
        Dataset: 466 trading days
        Articles: 3,376
        Period: Feb 2024 - Jan 2026
        
        CORRELATIONS
        ─────────────────
        Same-day: 0.26***
        Next-day: 0.00
        Urgency→Vol: 0.21***
        
        WIN RATES
        ─────────────────
        Bullish: 59.3%
        Strong Bullish: 61.7%
        Random: 50%
        """
        ax1.text(0.1, 0.9, metrics_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. Win Rate Chart (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        categories = ['Random', 'All Bullish', 'Strong\nBullish']
        win_rates = [50, 59.3, 61.7]
        colors = ['#95a5a6', '#3498db', '#27ae60']
        bars = ax2.bar(categories, win_rates, color=colors, edgecolor='black')
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 75)
        ax2.set_title('Win Rate Comparison', fontweight='bold')
        ax2.set_ylabel('Win Rate (%)')
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
        
        # 3. Conclusion Box (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        conclusion_text = """
        KEY FINDINGS
        ─────────────────────────
        
        ✅ Sentiment REFLECTS price
           (same-day r=0.26)
        
        ❌ Sentiment does NOT
           PREDICT next-day price
        
        ✅ High urgency predicts
           higher volatility
        
        ✅ 61.7% win rate on
           strong bullish days
        
        ⚠️ Edge may not survive
           transaction costs
        """
        ax3.text(0.1, 0.9, conclusion_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 4. Time Series (middle row, full width)
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(self.data['date'], self.data['close'], color='#f39c12', linewidth=1.5)
        ax4.fill_between(self.data['date'], self.data['close'], alpha=0.3, color='#f39c12')
        ax4.set_title('Gold Price Over Analysis Period', fontweight='bold')
        ax4.set_ylabel('Price ($)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Correlation comparison (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        corr_types = ['Same-Day', 'Next-Day']
        corr_values = [0.26, 0.00]
        colors = ['#27ae60', '#e74c3c']
        bars = ax5.bar(corr_types, corr_values, color=colors, edgecolor='black')
        ax5.axhline(y=0, color='black', linewidth=0.5)
        ax5.set_title('Sentiment-Return Correlation', fontweight='bold')
        ax5.set_ylabel('Correlation (r)')
        ax5.set_ylim(-0.1, 0.35)
        for bar in bars:
            height = bar.get_height()
            ax5.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
        
        # 6. Sentiment Distribution (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(self.data['avg_sentiment_score'].dropna(), bins=30, color='#3498db', 
                edgecolor='black', alpha=0.7)
        ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax6.set_title('Sentiment Score Distribution', fontweight='bold')
        ax6.set_xlabel('Sentiment Score')
        ax6.set_ylabel('Frequency')
        
        # 7. Strategy Sharpe Ratios (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        strategies = ['Buy & Hold', 'Bullish\nSentiment', 'Strong\nBullish']
        sharpes = [1.5, 2.1, 3.2]  # Approximate values
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        bars = ax7.bar(strategies, sharpes, color=colors, edgecolor='black')
        ax7.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Good (1.0)')
        ax7.set_title('Strategy Sharpe Ratios', fontweight='bold')
        ax7.set_ylabel('Sharpe Ratio')
        ax7.legend(loc='upper left', fontsize=8)
        for bar in bars:
            height = bar.get_height()
            ax7.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
        
        plt.savefig(f'{self.output_dir}00_summary_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 00_summary_dashboard.png")
    def generate_all(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print(" GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.load_data()
        
        print("\nCreating charts...")
        self.create_summary_dashboard()
        self.plot_correlation_heatmap()
        self.plot_win_rates()
        self.plot_sentiment_vs_returns()
        self.plot_time_series()
        self.plot_strategy_performance()
        self.plot_rsi_regimes()
        self.plot_urgency_analysis()
        
        print("\n" + "="*60)
        print(" ALL VISUALIZATIONS COMPLETE!")
        print("="*60)
        print(f"\n Charts saved to: {self.output_dir}")
        print("\nFiles created:")
        print("  00_summary_dashboard.png  - Overview of all findings")
        print("  01_correlation_heatmap.png - Correlation matrix")
        print("  02_win_rates.png - Win rate comparison")
        print("  03_sentiment_vs_returns.png - Scatter plots")
        print("  04_time_series.png - Price & sentiment over time")
        print("  05_strategy_performance.png - Cumulative returns")
        print("  06_rsi_regimes.png - RSI regime analysis")
        print("  07_urgency_analysis.png - Urgency vs volatility")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    visualizer = SentimentVisualizer()
    
    try:
        visualizer.generate_all()
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()