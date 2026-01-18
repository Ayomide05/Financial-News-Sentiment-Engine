"""ML Models with FinBERT Features
Compare ML performance: Ruke-based vs FinBERT vs Combined"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import warnings
from config import DB_CONFIG

warnings.filterwarnings('ignore')

class MLComparison:
    """Comapre ML models with different sentiment features"""
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
    def load_data(self):
        """Load combined data with both sentiment methods"""
        query = """
            SELECT
                dsf.date,
                dsf.article_count,
                -- FinBERT features
                dsf.avg_finbert_score,
                dsf.avg_headline_score,
                dsf.avg_content_score,
                dsf.positive_ratio as finbert_positive_ratio,
                dsf.negative_ratio as finbert_negative_ratio,
                dsf.avg_confidence as finbert_confidence,
                dsf.score_std as finbert_std,
                -- Rule-based features
                ds.avg_sentiment_score as rulebased_score,
                ds.bullish_ratio,
                ds.bearish_ratio,
                ds.avg_urgency_score,
                ds.breaking_news_count,
                ds.sentiment_std as rulebased_std,
                -- Technical features
                mp.daily_return,
                ti.rsi_14,
                ti.macd,
                ti.macd_signal,
                ti.volatility_20d,
                ti.atr_14,
                ti.price_vs_sma20,
                ti.trend_direction
            FROM daily_sentiment_finbert dsf
            JOIN daily_sentiment ds ON dsf.date = ds.date
            JOIN market_prices mp ON dsf.date = mp.date AND mp.symbol = 'GC=F'
            JOIN technical_indicators ti ON dsf.date = ti.date AND ti.symbol = 'GC=F'
            ORDER BY dsf.date
        """

        df = pd.read_sql(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])

        # Target: Will price go up tomorrow?
        df['next_day_return'] = df['daily_return'].shift(-1)
        df['target'] = (df['next_day_return'] > 0).astype(int)

        # Create lagged features
        for col in ['avg_finbert_score', 'rulebased_score', 'rsi_14', 'daily_return']:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_lag2'] = df[col].shift(2)  
        # Rolling Features
        df['finbert_ma3'] = df['avg_finbert_score'].rolling(3).mean()
        df['rulebased_ma3'] = df['rulebased_score'].rolling(3).mean()

        # Trend features
        df['trend_bullish'] = (df['trend_direction'] == 'bullish').astype(int)
        df['trend_bearish'] = (df['trend_direction'] == 'bearish').astype(int)
        
        # RSI categories
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        
        print(f" Loaded {len(df)} days of data")
        
        return df
    def prepare_feature_sets(self, df):
        """Prepare three feature sets for comparison"""
        # Drop NaN
        df_clean = df.dropna()
        
        # Technical features (baseline)
        technical_features = [
            'rsi_14', 'rsi_14_lag1', 'rsi_oversold', 'rsi_overbought',
            'macd', 'macd_signal', 'volatility_20d', 'atr_14',
            'price_vs_sma20', 'trend_bullish', 'trend_bearish',
            'daily_return', 'daily_return_lag1', 'daily_return_lag2'
        ]
        
        # Rule-based sentiment features
        rulebased_features = [
            'rulebased_score', 'rulebased_score_lag1', 'rulebased_score_lag2',
            'rulebased_ma3', 'bullish_ratio', 'bearish_ratio',
            'avg_urgency_score', 'breaking_news_count', 'rulebased_std'
        ]
        
        # FinBERT sentiment features
        finbert_features = [
            'avg_finbert_score', 'avg_finbert_score_lag1', 'avg_finbert_score_lag2',
            'finbert_ma3', 'avg_headline_score', 'avg_content_score',
            'finbert_positive_ratio', 'finbert_negative_ratio',
            'finbert_confidence', 'finbert_std'
        ]

        # Feature sets to compare
        feature_sets = {
            'Technical Only': technical_features,
            'Technical + Rule-based': technical_features + rulebased_features,
            'Technical + FinBERT': technical_features + finbert_features,
            'Technical + Both': technical_features + rulebased_features + finbert_features
        }

        # Verify all features exist
        for name, features in feature_sets.items():
            missing = [f for f in features if f not in df_clean.columns]
            if missing:
                print(f"⚠️ Missing features in {name}: {missing}")
                feature_sets[name] = [f for f in features if f in df_clean.columns]
        
        return df_clean, feature_sets
    
    def train_and_evaluate(self, df, feature_sets):
        """Train models on each feature set"""
        
        print(" ML MODEL COMPARISON")
        
        results = {}
        
        for set_name, features in feature_sets.items():
            print(f" Feature Set: {set_name} ({len(features)} features)")

        # Prepare data
            X = df[features]
            y = df['target']

            # Time-based split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        # Train XGBoost (best model from before)
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)  

            # Evaluate
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            # Cross-validation
            cv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

            # Win rate on positive predictions
            positive_preds = y_pred == 1
            if positive_preds.sum() > 0:
                win_rate = y_test[positive_preds].mean() * 100
            else:
                win_rate = 0
            
            results[set_name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'win_rate': win_rate,
                'n_features': len(features)
            }
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   ROC-AUC:   {roc_auc:.4f}")
            print(f"   CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"   Win Rate:  {win_rate:.1f}%")

            # Feature importance for this set
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n   Top 5 Features:")
            for i, row in importance.head(5).iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")
        
        return results
    
    def compare_results(self, results):
        """Compare and visualize results"""
        
        print(" FINAL COMPARISON")
                
        # Create comparison table
        print("\n" + "-"*80)
        print(f"{'Feature Set':<25} {'Accuracy':<12} {'ROC-AUC':<12} {'CV Score':<12} {'Win Rate':<12}")
        print("-"*80)
        
        for name, r in results.items():
            print(f"{name:<25} {r['accuracy']:<12.4f} {r['roc_auc']:<12.4f} "
                f"{r['cv_mean']:<12.4f} {r['win_rate']:<11.1f}%")
        
        print("-"*80)
        
        # Find best
        best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_auc = max(results.items(), key=lambda x: x[1]['roc_auc'])
        best_winrate = max(results.items(), key=lambda x: x[1]['win_rate'])
        
        print(f"\n WINNERS:")
        print(f"   Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
        print(f"   Best ROC-AUC:  {best_auc[0]} ({best_auc[1]['roc_auc']:.4f})")
        print(f"   Best Win Rate: {best_winrate[0]} ({best_winrate[1]['win_rate']:.1f}%)")
        
        # Summary
        print("\n" + "="*70)
        print(" EXECUTIVE SUMMARY")
        print("="*70)
        
        tech_only = results['Technical Only']
        tech_rb = results['Technical + Rule-based']
        tech_fb = results['Technical + FinBERT']
        tech_both = results['Technical + Both']
        
        rb_improvement = (tech_rb['accuracy'] - tech_only['accuracy']) / tech_only['accuracy'] * 100
        fb_improvement = (tech_fb['accuracy'] - tech_only['accuracy']) / tech_only['accuracy'] * 100
        both_improvement = (tech_both['accuracy'] - tech_only['accuracy']) / tech_only['accuracy'] * 100
        
        print(f"""
            SENTIMENT CONTRIBUTION TO ML                        
            _____________________________
                                                                     
            Baseline (Technical Only):     {tech_only['accuracy']:.4f} accuracy               
                                                                    
            Adding Rule-based sentiment:   {tech_rb['accuracy']:.4f} ({rb_improvement:+.1f}% change)          
            Adding FinBERT sentiment:      {tech_fb['accuracy']:.4f} ({fb_improvement:+.1f}% change)          
            Adding Both:                   {tech_both['accuracy']:.4f} ({both_improvement:+.1f}% change)          
                                                                     
            CONCLUSION:                                                        
            {'FinBERT' if fb_improvement > rb_improvement else 'Rule-based'} sentiment provides better ML improvement.          
        """)
        
        return results 
    def close(self):
        if self.conn:
            self.conn.close()    

def main():
    print("   ML MODELS: RULE-BASED vs FINBERT")
        
    comparison = MLComparison()
    
    try:
        # Load data
        df = comparison.load_data()
        
        # Prepare feature sets
        df_clean, feature_sets = comparison.prepare_feature_sets(df)
        print(f" Clean data: {len(df_clean)} samples")
        
        # Train and evaluate
        results = comparison.train_and_evaluate(df_clean, feature_sets)
        
        # Compare
        comparison.compare_results(results)
        
        print("\n ML Comparison complete!")
    finally:
        comparison.close()


if __name__ == "__main__":
    main()

