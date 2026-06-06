# run_analysis.py
"""
Project #032: Master Verification Script
Orchestrates strategy logs extraction and fires the factor analytics suite.
"""

import numpy as np
import pandas as pd
from data_fetcher import FamaFrenchFetcher
from factor_engine import FactorAnalysisEngine

def generate_mock_strategy_returns(factor_df: pd.DataFrame) -> pd.Series:
    """
    Generates realistic daily tracking returns on top of the fetched historical timeline
    to test the analytical engine pipeline framework securely.
    """
    # Grab matching datetime indices directly from the source factor frame to avoid data gaps
    timeline = factor_df.index[-500:] # Grab recent 500 trading days
    
    # Seed reproducible random walk variations
    np.random.seed(42)
    
    # Simulate a strategy that slightly outperforms the market with positive loading sizes
    # We add a structural daily return boost to simulate genuine edge alpha
    mock_returns = (0.65 * factor_df['Mkt_RF'].loc[timeline]) + \
                   (0.20 * factor_df['SMB'].loc[timeline]) + \
                   (0.02 / 252) + \
                   np.random.normal(0, 0.005, size=len(timeline))
                   
    return pd.Series(mock_returns, index=timeline, name="Strategy_Daily_Returns")

def main():
    print("\n" + "#"*70)
    print("INITIALIZING STRATEGY PORTFOLIO MATRIX DECOMPOSITION: PROJECT #032")
    print("#"*70 + "\n")
    
    # Step 1: Initialize the Factor Library Download
    fetcher = FamaFrenchFetcher()
    try:
        factor_data = fetcher.fetch_daily_factors()
    except Exception as e:
        print(f"[CRITICAL INGESTION EXCEPTION] Pipeline broken: {str(e)}")
        return
        
    # Step 2: Grab Strategy Returns Vector Data (Simulated or loaded from Project 31 transaction logs)
    strategy_returns = generate_mock_strategy_returns(factor_data)
    
    # Step 3: Instantiate and Run the Structural Regression Framework
    analyzer = FactorAnalysisEngine(portfolio_returns=strategy_returns, factor_data=factor_data)
    analyzer.execute_regression()
    
    # Step 4: Output the complete analytical report frame
    analyzer.print_performance_report()

if __name__ == "__main__":
    main()