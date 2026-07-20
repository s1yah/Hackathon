import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class PricingAnomalyDetector:
    def __init__(self):
        # category_baselines: { category_name: { 'mean': float, 'std': float, 'q1': float, 'q3': float, 'median': float } }
        self.category_baselines = {}
        
    def fit_from_dataframe(self, df: pd.DataFrame, category_col: str = "category", price_col: str = "price"):
        """Calculates statistical benchmarks per product category."""
        if category_col not in df.columns or price_col not in df.columns:
            logger.warning(f"Columns {category_col} or {price_col} missing in dataframe.")
            return

        grouped = df.groupby(category_col)
        for cat_name, group in grouped:
            prices = group[price_col].dropna().astype(float).values
            if len(prices) == 0:
                continue
            
            mean_p = float(np.mean(prices))
            std_p = float(np.std(prices))
            q1_p = float(np.percentile(prices, 25))
            q3_p = float(np.percentile(prices, 75))
            median_p = float(np.median(prices))

            self.category_baselines[str(cat_name).lower()] = {
                "mean": mean_p,
                "std": max(std_p, 1.0),
                "q1": q1_p,
                "q3": q3_p,
                "median": median_p,
                "count": len(prices)
            }
        logger.info(f"Fitted pricing baselines for {len(self.category_baselines)} categories.")

    def fit_from_csv(self, csv_path: str):
        if not os.path.exists(csv_path):
            logger.warning(f"CSV path {csv_path} does not exist.")
            return
        try:
            df = pd.read_csv(csv_path)
            # Normalize column names if needed
            cols = {c.lower(): c for c in df.columns}
            cat_col = cols.get("category", cols.get("product_category", "category"))
            price_col = cols.get("price", cols.get("discounted_price", cols.get("actual_price", "price")))
            
            # Clean price data if formatted as string (e.g. "$100" or "₹100")
            if df[price_col].dtype == object:
                df[price_col] = df[price_col].astype(str).str.replace(r"[^\d.]", "", regex=True)
                df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
                
            self.fit_from_dataframe(df, category_col=cat_col, price_col=price_col)
        except Exception as e:
            logger.error(f"Error loading price dataset CSV: {e}")

    def score(self, price: float, category: str) -> float:
        """
        Returns anomaly score between 0.0 (normal market price) and 1.0 (highly anomalous).
        """
        cat_key = str(category).lower().strip()
        if cat_key not in self.category_baselines:
            # Fallback for unknown categories: evaluate against overall global stats if available
            return 0.35

        baseline = self.category_baselines[cat_key]
        mean, std = baseline["mean"], baseline["std"]
        q1 = baseline["q1"]

        if std == 0:
            return 0.0

        # Z-score metric
        z_score = abs((price - mean) / std)
        
        # Base anomaly from Z-score (Z >= 3 yields near 1.0)
        anomaly_score = min(z_score / 3.0, 1.0)

        # Flag suspiciously deep discount / counterfeit cheap pricing (< 30% of Q1 or median)
        if price < (q1 * 0.3) or price < (baseline["median"] * 0.25):
            anomaly_score = min(anomaly_score + 0.35, 1.0)

        return round(min(max(anomaly_score, 0.0), 1.0), 3)

    def get_category_stats(self, category: str) -> dict:
        return self.category_baselines.get(str(category).lower().strip(), {})
