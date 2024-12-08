"""
Quick Start Guide:
-----------------
1. First run preprocessing.py to generate processed data files
2. Then run this script to train XGBoost model and generate predictions:
   python modeling_xgboost.py

Output:
- Saves predictions to predictions/5.txt
- First line contains validation AUC
- Remaining lines contain test set predictions
"""

# Standard imports for data processing, ML modeling, and utilities
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import logging
from pathlib import Path
import xgboost

print(f"XGBoost version: {xgboost.__version__}")

def load_data():
    """
    Loads preprocessed train and test datasets from parquet files.
    Returns DataFrames containing features and target variable 'ACTIVE'.
    """
    logging.info("Loading train and test datasets...")
    data_dir = Path(__file__).parent.parent / "processed_data"
    train = pd.read_parquet(data_dir / "processed_train.parquet")
    test = pd.read_parquet(data_dir / "processed_test.parquet")
    logging.info(f"Loaded train shape: {train.shape}, test shape: {test.shape}")
    return train, test

def split_data(train_df, test_size=0.1, random_state=42):
    """
    Performs a two-stage split of the training data:
    1. First split: Creates a holdout test set
    2. Second split: Divides remaining data into train and validation sets
    
    Maintains class balance through stratification on 'ACTIVE' target
    """
    logging.info("Splitting data into train/val/test sets...")
    # First split for test
    train_val, test = train_test_split(train_df, test_size=test_size, random_state=random_state, stratify=train_df['ACTIVE'])
    # Then split train_val into train and validation
    train, val = train_test_split(train_val, test_size=test_size/(1-test_size), random_state=random_state, stratify=train_val['ACTIVE'])
    
    logging.info(f"Split sizes - train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test

def get_initial_model(y_train):
    """
    Initializes XGBoost classifier with balanced class weights.
    
    Key parameters:
    - tree_method='hist': For faster training on large datasets
    - scale_pos_weight: Adjusts for class imbalance
    - eval_metric='auc': Optimizes for area under ROC curve
    """
    logging.info("Initializing XGBoost model...")
    # Calculate class weight scaling factor
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    logging.info(f"Class distribution - negative: {neg_count}, positive: {pos_count}")

    return XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        max_depth=7,
        learning_rate=0.1,
        n_estimators=150,
        min_child_weight=1
    )

def optimize_model(X_train, y_train, X_val, y_val):
    """
    Two-stage model optimization:
    1. Trains initial model with default hyperparameters
    2. Performs grid search around initial parameters to find optimal settings
    
    Grid search explores variations in:
    - max_depth: Controls tree complexity
    - learning_rate: Controls contribution of each tree
    - n_estimators: Number of boosting rounds
    """
    logging.info("Starting model optimization...")
    model = get_initial_model(y_train)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Refined parameter grid centered around current optimal values
    param_grid = {
        'max_depth': [6, 7, 8],
        'learning_rate': [0.05, 0.075, 0.1, 0.125],
        'n_estimators': [125, 150, 175, 200],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    logging.info("Starting grid search...")
    grid_search = GridSearchCV(
        estimator=XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist'
        ),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

def main():
    """
    Main pipeline orchestrating the entire modeling process:
    1. Data loading and splitting
    2. Feature preparation
    3. Model training and optimization
    4. Performance evaluation
    5. Final predictions and output generation
    """
    # Configure logging for tracking pipeline progress
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Starting XGBoost modeling pipeline...")
    
    train_df, test_df = load_data()
    train, val, test = split_data(train_df)
    
    # Extract features and target variable
    feature_cols = [col for col in train.columns if col != 'ACTIVE']
    X_train, y_train = train[feature_cols], train['ACTIVE']
    X_val, y_val = val[feature_cols], val['ACTIVE']
    X_test, y_test = test[feature_cols], test['ACTIVE']
    
    best_model = optimize_model(X_train, y_train, X_val, y_val)
    
    # Calculate AUC estimate (using validation AUC)
    val_pred = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    
    # Generate predictions for test set
    logging.info("Generating predictions for test set...")
    final_predictions = best_model.predict_proba(test_df)[:, 1]
    
    # Create output file with AUC estimate and predictions
    output_dir = Path(__file__).parent.parent / "predictions"
    output_dir.mkdir(exist_ok=True)
    
    # Combine AUC and predictions into single column
    output_data = np.concatenate(([val_auc], final_predictions))
    
    # Save without header, index as single column
    np.savetxt(output_dir / "5.txt", output_data, fmt='%.18e')
    
    logging.info(f"Saved predictions to {output_dir / '5.txt'}")
    logging.info("Pipeline completed successfully")
    
    # Verify output format
    predictions_df = pd.read_csv(output_dir / "5.txt", header=None)
    assert predictions_df.shape == (69647, 1)
    assert np.all((predictions_df.values >= 0) & (predictions_df.values <= 1))

if __name__ == "__main__":
    main()
