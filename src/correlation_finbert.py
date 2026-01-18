"""
Correlation Analysis - Comparing Rule-Based vs FinBERT
"""

import psycopg2
import pandas as pd
import numpy as np
from scipy import stats
from config import DB_CONFIG
import warnings

warnings.filterwarnings('ignore')

def load_combined_data():
    """Load data with both sentiment methods"""
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
        SELECT
            dsf.date,
            dsf.article_count,
            dsf.avg_finbert_score,
            dsf.positive_ratio as finbert_positive_ratio,
            dsf.negative_ratio as finbert_negative_ratio,
            dsf.avg_confidence as finbert_confidence,
            ds.avg_sentiment_score as rulebased_score,
            ds.bullish_ratio as rulebased_bullish_ratio,
            ds.bearish_ratio as rulebased_bearish_ratio,
            ds.avg_urgency_score,
            ds.breaking_news_count,
            mp.close,
            mp.daily_return,
            ti.rsi_14,
            ti.macd,
            ti.volatility_20d,
            ti.trend_direction
        FROM daily_sentiment_finbert dsf
        JOIN daily_sentiment ds ON dsf.date = ds.date
        JOIN market_prices mp ON dsf.date = mp.date AND mp.symbol = 'GC=F'
        JOIN technical_indicators ti ON dsf.date = ti.date AND ti.symbol = 'GC=F'
        ORDER BY dsf.date
    """

    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df['next_day_return'] = df['daily_return'].shift(-1)
    df['price_up']= (df['next_day_return'] > 0).astype(int)

    conn.close()
    return df
def compare_correlations(df):
    """Compare correlations between methods"""
    print("CORRELATION COMPARISON: RULE-BASED vs FINBERT")

    valid_df = df.dropna(subset=['next_day_return'])

    # same-day correlations
    rb_same = valid_df['rulebased_score'].corr(valid_df['daily_return'])
    fb_same = valid_df['avg_finbert_score'].corr(valid_df['daily_return'])

    print(f"Rule-based:  {rb_same:.4f}")
    print(f"FinBERT:     {fb_same:.4f}")
    print(f"Winner:      {'FinBERT' if abs(fb_same) > abs(rb_same) else 'Rule-based'}")

    # Next-day correlations
    rb_next = valid_df['rulebased_score'].corr(valid_df['next_day_return'])
    fb_next = valid_df['avg_finbert_score'].corr(valid_df['next_day_return'])

    # Statistical Significance
    rb_corr, rb_pval = stats.pearsonr(valid_df['rulebased_score'].dropna(), valid_df['next_day_return'].dropna())
    fb_corr, fb_pval = stats.pearsonr(valid_df['avg_finbert_score'].dropna(), valid_df['next_day_return'].dropna())

    rb_sig = "***" if rb_pval < 0.01 else "**" if rb_pval < 0.05 else "*" if rb_pval < 0.1 else ""
    fb_sig = "***" if fb_pval < 0.01 else "**" if fb_pval < 0.05 else "*" if fb_pval < 0.1 else ""

    print(f"Rule-based:  {rb_next:.4f} {rb_sig} (p={rb_pval:.4f})")
    print(f"FinBERT:     {fb_next:.4f} {fb_sig} (p={fb_pval:.4f})")
    print(f"Winner:      {'FinBERT' if abs(fb_next) > abs(rb_next) else 'Rule-based'} ")
    
    return {
        'rulebased_same': rb_same,
        'finbert_same': fb_same,
        'rulebased_next': rb_next,
        'finbert_next': fb_next,
        'rulebased_pval': rb_pval,
        'finbert_pval': fb_pval
    }

def compare_win_rates(df):
    """Compare win rates between methods"""
    print(" WIN RATE COMPARISON")

    valid_df = df.dropna(subset=['next_day_return'])

    # Rule-based bullish
    rb_bullish = valid_df[valid_df['rulebased_score'] > 0]
    rb_win_rate = (rb_bullish['next_day_return'] > 0).mean() * 100

    # FinBERT positive
    fb_positive = valid_df[valid_df['avg_finbert_score'] > 0]
    fb_win_rate = (fb_positive['next_day_return'] > 0).mean() * 100

    print("\n🔹 BULLISH/POSITIVE SIGNAL WIN RATE")
    print("-"*50)
    print(f"Rule-based bullish days: {len(rb_bullish)}, Win rate: {rb_win_rate:.1f}%")
    print(f"FinBERT positive days:   {len(fb_positive)}, Win rate: {fb_win_rate:.1f}%")
    print(f"Winner: {'FinBERT' if fb_win_rate > rb_win_rate else 'Rule-based'} ")

    # Rule-based strong bullish (>70% bullish ratio)
    rb_strong = valid_df[valid_df['rulebased_bullish_ratio'] > 0.7]
    rb_strong_win = (rb_strong['next_day_return'] > 0).mean() * 100 if len(rb_strong) > 10 else 0
    
    # FinBERT strong positive (>50% positive ratio)
    fb_strong = valid_df[valid_df['finbert_positive_ratio'] > 0.5]
    fb_strong_win = (fb_strong['next_day_return'] > 0).mean() * 100 if len(fb_strong) > 10 else 0

    print(f"Rule-based strong bullish (>70%): {len(rb_strong)} days, Win rate: {rb_strong_win:.1f}%")
    print(f"FinBERT strong positive (>50%):   {len(fb_strong)} days, Win rate: {fb_strong_win:.1f}%")
    
    # Bearish/Negative signals
    print("\n🔹 BEARISH/NEGATIVE SIGNAL WIN RATE (Price goes DOWN)")
    print("-"*50)
    
    rb_bearish = valid_df[valid_df['rulebased_score'] < 0]
    rb_bearish_win = (rb_bearish['next_day_return'] < 0).mean() * 100 if len(rb_bearish) > 10 else 0
    
    fb_negative = valid_df[valid_df['avg_finbert_score'] < 0]
    fb_negative_win = (fb_negative['next_day_return'] < 0).mean() * 100 if len(fb_negative) > 10 else 0
    
    print(f"Rule-based bearish days: {len(rb_bearish)}, Win rate: {rb_bearish_win:.1f}%")
    print(f"FinBERT negative days:   {len(fb_negative)}, Win rate: {fb_negative_win:.1f}%")
    print(f"Winner: {'FinBERT' if fb_negative_win > rb_bearish_win else 'Rule-based'} ✅")
    
    return {
        'rb_bullish_win': rb_win_rate,
        'fb_positive_win': fb_win_rate,
        'rb_strong_win': rb_strong_win,
        'fb_strong_win': fb_strong_win,
        'rb_bearish_win': rb_bearish_win,
        'fb_negative_win': fb_negative_win
    }
def calculate_returns(df):
   """Calculate strategy returns"""
   print(" STRATEGY RETURNS COMPARISON")

   valid_df = df.dropna(subset=['next_day_return']).copy()

   # Buy and hold
   bh_return = (1 + valid_df['next_day_return']).prod() - 1

   # Rule-based strategy (buy when bullish)
   valid_df['rb_signal'] = (valid_df['rulebased_score'] > 0).astype(int)
   valid_df['rb_return'] = valid_df['next_day_return'] * valid_df['rb_signal']
   rb_strategy_return = (1 + valid_df['rb_return']).prod() - 1

   # FinBERT strategy (buy when positive)
   valid_df['fb_signal'] = (valid_df['avg_finbert_score'] > 0).astype(int)
   valid_df['fb_return'] = valid_df['next_day_return'] * valid_df['fb_signal']
   fb_strategy_return = (1 + valid_df['fb_return']).prod() - 1
    
   # Calculate Sharpe ratios
   rb_sharpe = valid_df['rb_return'].mean() / valid_df['rb_return'].std() * np.sqrt(252) if valid_df['rb_return'].std() > 0 else 0
   fb_sharpe = valid_df['fb_return'].mean() / valid_df['fb_return'].std() * np.sqrt(252) if valid_df['fb_return'].std() > 0 else 0
   bh_sharpe = valid_df['next_day_return'].mean() / valid_df['next_day_return'].std() * np.sqrt(252)
    
   print("\n TOTAL RETURNS")
   print("-"*50)
   print(f"Buy & Hold:        {bh_return*100:+.2f}% (Sharpe: {bh_sharpe:.2f})")
   print(f"Rule-based:        {rb_strategy_return*100:+.2f}% (Sharpe: {rb_sharpe:.2f})")
   print(f"FinBERT:           {fb_strategy_return*100:+.2f}% (Sharpe: {fb_sharpe:.2f})")

   best = max([('Buy & Hold', bh_return), ('Rule-based', rb_strategy_return), ('FinBERT', fb_strategy_return)], key=lambda x: x[1])
   print(f"\nBest Strategy: {best[0]} ") 

   return {
        'bh_return': bh_return,
        'rb_return': rb_strategy_return,
        'fb_return': fb_strategy_return,
        'bh_sharpe': bh_sharpe,
        'rb_sharpe': rb_sharpe,
        'fb_sharpe': fb_sharpe
   }

def  summary_report(correlations, win_rates, returns):
    """Generate summary report"""
    print("\n" + "="*70)
    print(" EXECUTIVE SUMMARY: RULE-BASED vs FINBERT")
    print("="*70)
    
    print(f"""
       CORRELATION WITH NEXT-DAY RETURNS                                  
        ─────────────────────────────────                                  
       Rule-based: {correlations['rulebased_next']:>8.4f} (p={correlations['rulebased_pval']:.4f})                      
       FinBERT:    {correlations['finbert_next']:>8.4f} (p={correlations['finbert_pval']:.4f}) 

       WIN RATES                                                          
       ─────────                                                          
       Bullish/Positive signal:                                           
       Rule-based: {win_rates['rb_bullish_win']:>5.1f}%                                              
       FinBERT:    {win_rates['fb_positive_win']:>5.1f}%  

       STRATEGY RETURNS                                                   
       ────────────────                                                   
       Rule-based: {returns['rb_return']*100:>+6.2f}% (Sharpe: {returns['rb_sharpe']:.2f})                         
       FinBERT:    {returns['fb_return']*100:>+6.2f}% (Sharpe: {returns['fb_sharpe']:.2f})                         
                                                                     
       CONCLUSION                                                         
       ──────────     
    """)
    # Determine winner
    fb_wins = 0
    rb_wins = 0
    
    if abs(correlations['finbert_next']) > abs(correlations['rulebased_next']):
        fb_wins += 1
    else:
        rb_wins += 1
        
    if win_rates['fb_positive_win'] > win_rates['rb_bullish_win']:
        fb_wins += 1
    else:
        rb_wins += 1
        
    if returns['fb_return'] > returns['rb_return']:
        fb_wins += 1
    else:
        rb_wins += 1
    
    if fb_wins > rb_wins:
        print(f"│  FinBERT wins {fb_wins}/3 metrics - BETTER for this dataset           │")
    else:
        print(f"│  Rule-based wins {rb_wins}/3 metrics - BETTER for this dataset        │")

def main():
    print("   SENTIMENT METHOD COMPARISON")
        
    # Load data
    print("\n Loading combined data...")
    df = load_combined_data()
    print(f"   Loaded {len(df)} days of data")
    
    # Run comparisons
    correlations = compare_correlations(df)
    win_rates = compare_win_rates(df)
    returns = calculate_returns(df)
    
    # Summary
    summary_report(correlations, win_rates, returns)
    
    print("\n Comparison complete!")
    

if __name__ == "__main__":
    main()
