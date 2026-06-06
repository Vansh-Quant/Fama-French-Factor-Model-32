# factor_engine.py
"""
Project #032: Analytics & Core Regression Engine
Executes Ordinary Least Squares regression to decompose strategies into risk metrics.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

class FactorAnalysisEngine:
    def __init__(self, portfolio_returns: pd.Series, factor_data: pd.DataFrame):
        """
        Initializes regression pipeline matrix alignment.
        portfolio_returns: Daily strategy return series indexed by DateTime.
        factor_data: Fetched Fama-French baseline factor dataframe.
        """
        # Clean merge datasets strictly matching mutual datetime index fingerprints
        self.analysis_frame = pd.concat([portfolio_returns, factor_data], axis=1, join='inner').dropna()
        self.analysis_frame.columns = ['Strategy_Returns', 'Mkt_RF', 'SMB', 'HML', 'RF']
        
        # Compute Strategy Excess Returns over Risk-Free Rate
        self.analysis_frame['Excess_Returns'] = self.analysis_frame['Strategy_Returns'] - self.analysis_frame['RF']
        self.model_results = None

    def execute_regression(self):
        """Fits the Ordinary Least Squares (OLS) model against your alpha returns."""
        if len(self.analysis_frame) < 5:
            raise ValueError(f"Insufficient overlapping time data points found. Count: {len(self.analysis_frame)}")
            
        # Dependent Variable (Y Matrix): Excess returns of your algorithm
        Y = self.analysis_frame['Excess_Returns']
        
        # Independent Variables (X Matrix): Market Premium, Size Factor, Value Factor
        X = self.analysis_frame[['Mkt_RF', 'SMB', 'HML']]
        
        # Add a constant row to calculate the absolute Intercept (Alpha)
        X = sm.add_constant(X)
        
        # Initialize and Fit Ordinary Least Squares
        model = sm.OLS(Y, X)
        self.model_results = model.fit()

    def print_performance_report(self):
        """Renders a high-density matrix output breaking down your true factor alphas."""
        if self.model_results is None:
            print("[ERROR] Regression model has not been initialized or executed yet.")
            return
            
        print("\n" + "="*70)
        print("          FAMA-FRENCH THREE-FACTOR DECOMPOSITION MATRIX RESULT          ")
        print("="*70)
        
        # Extract foundational analytics parameters
        alpha = self.model_results.params['const']
        beta_mkt = self.model_results.params['Mkt_RF']
        beta_smb = self.model_results.params['SMB']
        beta_hml = self.model_results.params['HML']
        
        p_alpha = self.model_results.pvalues['const']
        r_squared = self.model_results.rsquared_adj
        
        # Standardize alpha serialization output up to annualized terms
        annualized_alpha = alpha * 252 * 100 # Transformed into clean annual percentage yield representation
        
        print(f"Annualized System Alpha (Unexplained Edge) : {annualized_alpha:.2f}%  (p-value: {p_alpha:.4f})")
        print(f"Adjusted Model R-Squared (Explanatory Power): {r_squared:.4f}")
        print("-" * 70)
        print("FACTOR LOADING COEFFICIENTS (BETAS):")
        print(f" * Market Systematic Sensitivity (Beta Mkt)   : {beta_mkt:.4f} (p-value: {self.model_results.pvalues['Mkt_RF']:.4f})")
        print(f" * Size Stratification Exposure (Beta SMB)    : {beta_smb:.4f} (p-value: {self.model_results.pvalues['SMB']:.4f})")
        print(f" * Structural Value Orientation (Beta HML)   : {beta_hml:.4f} (p-value: {self.model_results.pvalues['HML']:.4f})")
        print("="*70)
        
        # Print a quick raw structural analysis evaluation summary
        print("\n[QUANT INTERPRETATION BRIEF]:")
        if p_alpha < 0.05:
            print(" -> STATISTICAL VALIDITY CONFIRMED: Your strategy possesses a genuine, statistically")
            print("    significant alpha edge that cannot be replicated by basic market factor index exposure.")
        else:
            print(" -> SIGNIFCANCE WARNING: Strategy Alpha p-value is higher than baseline alpha limit (0.05).")
            print("    Your performance variance is heavily dictated by accidental risk factor exposures.")