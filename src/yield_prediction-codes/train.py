import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
from itertools import product

# Configuration
CONFIG = {
    "data_path": r"data\yield_prediction-data",
    "model_path": r"models\yield_prediction-models",
    "seed": 42
}

# Create directories if they don't exist
os.makedirs(CONFIG['model_path'], exist_ok=True)

def load_data():
    """Load and verify datasets"""
    print("\nLoading data...")
    train = pd.read_csv(os.path.join(CONFIG['data_path'], 'train_data.csv'))
    val = pd.read_csv(os.path.join(CONFIG['data_path'], 'val_data.csv'))
    
    # Verify shapes
    assert train.shape[1] == val.shape[1], "Column mismatch"
    print(f"Data loaded - Train: {train.shape}, Val: {val.shape}")
    
    # Separate features and target
    X_train = train.drop('hg/ha_yield', axis=1)
    y_train = train['hg/ha_yield']
    X_val = val.drop('hg/ha_yield', axis=1)
    y_val = val['hg/ha_yield']
    
    return X_train, y_train, X_val, y_val

def get_model_configs():
    """Define hyperparameter grids for each model"""
    return {
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7, None],
            'min_samples_leaf': [3, 5, 7]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 25, None],
            'min_samples_leaf': [1, 3, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [4, 6, 8, None],
            'subsample': [0.6, 0.8, 1.0]
        },
        'DecisionTree': {
            'max_depth': [10, 25, None],
            'min_samples_leaf': [1, 3, 5]
        }
    }

def tune_and_train(X_train, y_train, X_val, y_val):
    """Tune and train models using validation set"""
    model_configs = get_model_configs()
    
    for model_name, param_grid in model_configs.items():
        print(f"\n=== Tuning {model_name} ===")
        
        # Generate all parameter combinations
        keys, values = zip(*param_grid.items())# *param_grid.items() -----> ([('n_estimators', [100, 200]), ('max_depth', [3, 5])...])
        # zip ----> [('n_estimators', 'max_depth'), ([100, 200], [3, 5])]
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        best_score = -np.inf
        best_params = None
        best_model = None
        
        # Grid search over parameter combinations
        for params in param_combinations:
            # Initialize model with current parameters
            if model_name == 'GradientBoosting':
                model = GradientBoostingRegressor(random_state=CONFIG['seed'], **params)
            elif model_name == 'RandomForest':
                model = RandomForestRegressor(random_state=CONFIG['seed'], n_jobs=-1, **params)
            elif model_name == 'XGBoost':
                model = XGBRegressor(random_state=CONFIG['seed'], n_jobs=-1, **params)
            elif model_name == 'DecisionTree':
                model = DecisionTreeRegressor(random_state=CONFIG['seed'], **params)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            score = r2_score(y_val, val_pred)
            
            # Track best parameters
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
        
        # Evaluate best model
        train_pred = best_model.predict(X_train)
        val_pred = best_model.predict(X_val)
        
        print("\nBest Parameters:")
        for k, v in best_params.items():
            print(f"{k}: {v}")
        
        print("\nTraining Performance:")
        print(f"R2 Score: {r2_score(y_train, train_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_train, train_pred):.2f}")
        
        print("\nValidation Performance:")
        print(f"R2 Score: {best_score:.4f}")
        print(f"MAE: {mean_absolute_error(y_val, val_pred):.2f}")
        
        # Save model
        model_file = f"{model_name}_best1.pkl"
        joblib.dump(best_model, os.path.join(CONFIG['model_path'], model_file))
        print(f"\nSaved best model as {model_file}")
        print("="*50)

def main():
    # Load data
    X_train, y_train, X_val, y_val = load_data()
    
    # Tune and train models
    tune_and_train(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()