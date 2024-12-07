import os
import warnings
from typing import Tuple, Dict
from pathlib import Path

# Suppress warnings
os.environ['MKL_DISABLE_FAST_MM'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def load_preprocessed_data(feature_set: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed training and test data for a specific feature set
    """
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "processed_data"
    
    train_data = pd.read_parquet(data_dir / f"train_{feature_set}.parquet")
    test_data = pd.read_parquet(data_dir / f"test_{feature_set}.parquet")
    
    return train_data, test_data

def prepare_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[xgb.DMatrix, xgb.DMatrix, xgb.DMatrix, xgb.DMatrix]:
    """Split training data and convert to DMatrix"""
    # Split features and target
    X = train_data.drop(columns=["ACTIVE"])
    y = train_data["ACTIVE"]
    
    # Split train/val/test
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=(val_size/(test_size + val_size)),
        random_state=random_state,
        stratify=y_temp
    )
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dholdout = xgb.DMatrix(X_holdout, label=y_holdout)
    dtest = xgb.DMatrix(test_data)
    
    return dtrain, dval, dholdout, dtest

def train_baseline_model(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    random_state: int = 42
) -> Tuple[xgb.Booster, Dict]:
    """Train a baseline XGBoost model"""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'max_depth': 4,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state
    }
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    return model, params

def evaluate_model(
    model: xgb.Booster,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    dholdout: xgb.DMatrix
) -> Tuple[float, float, float]:
    """Evaluate model on all splits"""
    train_preds = model.predict(dtrain)
    val_preds = model.predict(dval)
    holdout_preds = model.predict(dholdout)
    
    train_auc = roc_auc_score(dtrain.get_label(), train_preds)
    val_auc = roc_auc_score(dval.get_label(), val_preds)
    holdout_auc = roc_auc_score(dholdout.get_label(), holdout_preds)
    
    return train_auc, val_auc, holdout_auc

def main():
    """Compare XGBoost performance on different feature sets"""
    # All available feature sets
    feature_sets = [
        'stat_nbest',
        'stat_FPR',
        'stat_FDR',
        'stat_FWER',
        'model_rf'
    ]
    
    results = []
    predictions = {}
    
    # Evaluate each feature set
    for feature_set in feature_sets:
        print(f"\nEvaluating feature set: {feature_set}")
        
        # Load and prepare data
        train_data, test_data = load_preprocessed_data(feature_set)
        dtrain, dval, dholdout, dtest = prepare_data(train_data, test_data)
        
        # Train model
        model, params = train_baseline_model(dtrain, dval)
        
        # Evaluate
        train_auc, val_auc, holdout_auc = evaluate_model(
            model, dtrain, dval, dholdout
        )
        
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Holdout AUC: {holdout_auc:.4f}")
        
        # Store results
        results.append({
            'feature_set': feature_set,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'holdout_auc': holdout_auc,
            'n_features': train_data.shape[1] - 1  # Exclude target
        })
        
        # Generate predictions for test set
        predictions[feature_set] = model.predict(dtest)
    
    # Save results
    current_dir = Path(__file__).parent
    output_dir = current_dir.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save summary
    pd.DataFrame(results).to_csv(
        output_dir / "feature_set_comparison.csv",
        index=False
    )
    
    # Save predictions for each feature set
    for feature_set, preds in predictions.items():
        pd.DataFrame({
            'prediction': preds
        }).to_csv(output_dir / f"predictions_{feature_set}.csv", index=False)
    
    # Print best performing feature set
    best_result = max(results, key=lambda x: x['holdout_auc'])
    print(f"\nBest feature set: {best_result['feature_set']}")
    print(f"Holdout AUC: {best_result['holdout_auc']:.4f}")

if __name__ == "__main__":
    main()