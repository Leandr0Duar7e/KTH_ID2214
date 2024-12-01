import numpy as np
import pandas as pd

class DataPreprocessor:
    """
    Handles data preprocessing tasks like filtering, normalization, imputation, and encoding.
    """
    @staticmethod
    def create_column_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Creates a filtered dataframe by removing columns with only missing values or one unique value.
        Keeps CLASS and ID columns.
        """
        df_copy = df.copy()
        column_filter = [col for col in df.columns if col in ['CLASS', 'ID']]
        
        for col in df.columns:
            if col not in ['CLASS', 'ID']:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > 1:
                    column_filter.append(col)
        
        return df_copy[column_filter], column_filter

    @staticmethod
    def apply_column_filter(df: pd.DataFrame, column_filter: list[str]) -> pd.DataFrame:
        """Applies a column filter to keep only specified columns."""
        return df.copy()[column_filter]

    @staticmethod
    def create_imputation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Create and apply imputation values for missing data."""
        df_copy = df.copy()
        imputation = {}

        for col in df.columns:
            if col in ["CLASS", "ID"]:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
                if pd.isna(fill_value):
                    fill_value = 0
            elif df[col].dtype == 'category':
                fill_value = (
                    df[col].mode().iloc[0]
                    if not df[col].mode().empty
                    else df[col].cat.categories[0]
                )
            else:
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else ""

            df_copy[col] = df_copy[col].fillna(fill_value)
            imputation[col] = fill_value

        return df_copy, imputation

    @staticmethod
    def apply_imputation(df: pd.DataFrame, imputation: dict) -> pd.DataFrame:
        """Apply existing imputation values to missing values."""
        df_copy = df.copy()
        for col, value in imputation.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(value)
        return df_copy

class FeatureTransformer:
    """
    Handles feature transformation tasks like normalization and encoding.
    """
    @staticmethod
    def minmax_normalize(
            col: pd.Series, 
            min_val: float = None, 
            max_val: float = None
    ) -> tuple[pd.Series, tuple[float, float]]:
        """MinMax normalization of a column."""
        norm_col = col.copy()
        col_min = col.min() if min_val is None else min_val
        col_max = col.max() if max_val is None else max_val
        norm_col = (norm_col - col_min) / (col_max - col_min)
        return norm_col, (col_min, col_max)

    @staticmethod
    def zscore_normalize(
            col: pd.Series, 
            mean_val: float = None, 
            std_val: float = None
    ) -> tuple[pd.Series, tuple[float, float]]:
        """Z-score normalization of a column."""
        norm_col = col.copy()
        col_mean = col.mean() if mean_val is None else mean_val
        col_std = col.std() if std_val is None else std_val
        norm_col = (norm_col - col_mean) / col_std
        return norm_col, (col_mean, col_std)

    @staticmethod
    def create_one_hot(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Create one-hot encoding for categorical features."""
        df_copy = df.copy()
        one_hot = {}

        for col in df.columns:
            if col in ["CLASS", "ID"]:
                continue
            if df[col].dtype not in ["object", "category"]:
                continue

            categories = df[col].unique()
            one_hot[col] = categories

            for category in categories:
                new_col_name = f"{col}_{category}"
                df_copy[new_col_name] = (df[col] == category).astype(float)

            df_copy.drop(columns=[col], inplace=True)

        return df_copy, one_hot

    @staticmethod
    def apply_one_hot(df: pd.DataFrame, one_hot: dict) -> pd.DataFrame:
        """Apply existing one-hot encoding."""
        df_copy = df.copy()

        for col, categories in one_hot.items():
            if col not in df_copy.columns:
                continue

            for category in categories:
                new_col_name = f"{col}_{category}"
                df_copy[new_col_name] = (df[col] == category).astype(float)

            df_copy.drop(columns=[col], inplace=True)

        return df_copy

class ModelEvaluator:
    """
    Handles model evaluation metrics and scoring.
    """
    @staticmethod
    def accuracy(df: pd.DataFrame, correctlabels: list) -> float:
        """Calculate prediction accuracy."""
        predicted_labels = df.idxmax(axis=1)
        n_correct = sum(pred == correct for pred, correct in zip(predicted_labels, correctlabels))
        return n_correct / df.shape[0]

    @staticmethod
    def brier_score(df: pd.DataFrame, correctlabels: list) -> float:
        """Calculate Brier score for probabilistic predictions."""
        squared_errors = []
        for i, label in enumerate(correctlabels):
            true_vector = np.zeros(len(df.columns))
            true_vector[np.where(df.columns == label)[0][0]] = 1
            prediction = df.iloc[i].values
            squared_error = np.sum((prediction - true_vector) ** 2)
            squared_errors.append(squared_error)
        return np.mean(squared_errors)

    @staticmethod
    def auc(df: pd.DataFrame, correctlabels: list[int]) -> float:
        """Calculate area under ROC curve."""
        assert len(df) == len(correctlabels)
        correctlabels = np.asarray(correctlabels)
        classes, counts = np.unique(correctlabels, return_counts=True)
        counts = counts / len(correctlabels)
        class_freqs = {cls: cnt for cls, cnt in zip(classes, counts)}
        
        auc = 0
        for cls in df.columns:
            tps = (correctlabels == cls).astype(int)
            fps = (correctlabels != cls).astype(int)
            scores_performance = [(s, tp, fp) for s, tp, fp in zip(df[cls], tps, fps)]
            scores_performance.sort(key=lambda x: x[0], reverse=True)
            auc += class_freqs.get(cls, 0) * ModelEvaluator._feature_auc(
                scores_performance, sum(tps), sum(fps)
            )
        return auc

    @staticmethod
    def _feature_auc(
        scores_performance: list[tuple[float, int, int]], 
        tot_tp: int, 
        tot_fp: int
    ) -> float:
        """Calculate AUC for a specific feature."""
        auc_c = 0
        cov_tp = 0
        for s, tp_s, fp_s in scores_performance:
            if fp_s == 0:
                cov_tp += tp_s
            elif tp_s == 0:
                auc_c += (cov_tp / tot_tp) * (fp_s / tot_fp)
            else:
                auc_c += (cov_tp / tot_tp) * (fp_s / tot_fp) + (tp_s / tot_tp) * (
                    fp_s / tot_fp
                ) * 0.5
                cov_tp += tp_s
        return auc_c
