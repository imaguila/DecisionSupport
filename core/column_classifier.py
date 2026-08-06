## --------------------------------------------------------------------------------------
## column_classifier.py

import pandas as pd

class ColumnClassifier:
    """
    Handles dynamic column classification and exclusions based on problem configuration.
    Categorizes dataset attributes into Decision Variables, Base Metrics, and Derived Indicators.
    """
    
    def __init__(self, config: dict):
        self.metrics = set(config.get("metrics", []))
        self.var_prefix = config.get("var_prefix", "x_")
        self.user_excludes = set(config.get("exclude_cols", []))
        
        # Internal system-level columns generated dynamically by the framework
        self.system_excludes = {"highlight", "label", "highlight_label", "score", "cluster", "selected"}

    def get_decision_variables(self, df: pd.DataFrame) -> list:
        """Extracts decision variable columns (X) using the configured prefix."""
        return [col for col in df.columns if col.startswith(self.var_prefix)]

    def get_metrics(self, df: pd.DataFrame) -> list:
        """Extracts base optimization metrics (M) defined in the configuration."""
        return [col for col in df.columns if col in self.metrics]

    def get_derived_indicators(self, df: pd.DataFrame) -> list:
        """
        Extracts derived/enrichment indicators (I).
        Identifies numeric columns that are neither base metrics, decision variables, nor excluded attributes.
        """
        all_excluded = self.system_excludes | self.user_excludes | self.metrics
        
        indicators = []
        for col in df.columns:
            if col in all_excluded or col.startswith(self.var_prefix):
                continue
            # If the column is numeric and passed all exclusion filters, treat it as a derived lens/indicator
            if pd.api.types.is_numeric_dtype(df[col]):
                indicators.append(col)
                
        return indicators