import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Configuration
CONFIG = {
    "data_path": r"data\yield_prediction-data",
    "model_path": r"models\yield_prediction-models",
    "seed": 42
}

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_all_data():
    """Load and prepare all datasets"""
    print("\nLoading datasets...")
    train = pd.read_csv(os.path.join(CONFIG['data_path'], 'train_data.csv'))
    val = pd.read_csv(os.path.join(CONFIG['data_path'], 'val_data.csv'))
    test = pd.read_csv(os.path.join(CONFIG['data_path'], 'test_data.csv'))
    
    # Verify consistency
    assert 'hg/ha_yield' in train.columns, "Train data missing target column"
    assert 'hg/ha_yield' in val.columns, "Validation data missing target column"
    assert 'hg/ha_yield' in test.columns, "Test data missing target column"
    assert train.shape[1] == val.shape[1] == test.shape[1], "Column mismatch between datasets"
    
    print(f"Data loaded - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    
    def split_features_target(df):
        X = df.drop('hg/ha_yield', axis=1)
        y = df['hg/ha_yield']
        return X, y
    
    X_train, y_train = split_features_target(train)
    X_val, y_val = split_features_target(val)
    X_test, y_test = split_features_target(test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_saved_models():
    """Load all saved models from the models directory"""
    print("\nLoading saved models...")
    models = {}
    
    model_files = {
        'GradientBoosting': 'GradientBoosting_best.pkl',
        'RandomForest': 'RandomForest_best.pkl',
        'XGBoost': 'XGBoost_best.pkl',
        'DecisionTree': 'DecisionTree_best.pkl'
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(CONFIG['model_path'], filename)
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
            print(f"Loaded {model_name} from {filename}")
        else:
            print(f"Warning: Could not find {filename}")
    
    return models

def evaluate_model(model, X, y, dataset_name):
    """Evaluate a single model on a specific dataset"""
    y_pred = model.predict(X)
    
    return {
        'Dataset': dataset_name,
        'R2 Score': r2_score(y, y_pred),
        'MAE': mean_absolute_error(y, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred))
    }

def create_metric_plots(results_df):
    """Create clear visualizations for each evaluation metric"""
    plots_dir = os.path.join(CONFIG['model_path'], 'evaluation_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Custom color palette
    palette = {'Training': '#1f77b4', 'Validation': '#ff7f0e', 'Test': '#2ca02c'}
    
    # 1. R2 Score Plot with fixed 95-100% range
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=results_df, x='Model', y='R2 Score', hue='Dataset', palette=palette)
    ax.set_ylim(0.95, 1.0)  # Fixed range 95-100%
    
    # Add value labels with 3 decimal places
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 5), 
                   textcoords='offset points')
    
    plt.title('Model Comparison: R2 Scores')
    plt.ylabel('R2 Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'r2_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MAE Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=results_df, x='Model', y='MAE', hue='Dataset', palette=palette)
    
    # Add value labels with 2 decimal places
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 5), 
                   textcoords='offset points')
    
    plt.title('Model Comparison: Mean Absolute Error')
    plt.ylabel('MAE')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. RMSE Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=results_df, x='Model', y='RMSE', hue='Dataset', palette=palette)
    
    # Add value labels with 2 decimal places
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 5), 
                   textcoords='offset points')
    
    plt.title('Model Comparison: Root Mean Squared Error')
    plt.ylabel('RMSE')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rmse_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_all_models(models, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate all models on all datasets"""
    all_results = []
    
    print("\n=== Evaluating Models on All Datasets ===")
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Evaluate on all datasets
        train_results = evaluate_model(model, X_train, y_train, 'Training')
        val_results = evaluate_model(model, X_val, y_val, 'Validation')
        test_results = evaluate_model(model, X_test, y_test, 'Test')
        
        # Combine results
        for result in [train_results, val_results, test_results]:
            all_results.append({
                'Model': model_name,
                **result
            })
    
    results_df = pd.DataFrame(all_results)
    
    # Create visualizations
    create_metric_plots(results_df)
    
    return results_df

def main():
    # Load all datasets
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_data()
    
    # Load saved models
    models = load_saved_models()
    if not models:
        print("No models found to test!")
        return
    
    # Evaluate models on all datasets
    results_df = evaluate_all_models(models, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Print and save comprehensive results
    print("\n=== Comprehensive Model Evaluation ===")
    print(results_df.to_string(index=False))
    
    # Save detailed results
    results_file = os.path.join(CONFIG['model_path'], 'comprehensive_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved comprehensive results to {results_file}")
    print(f"Visualizations saved to {os.path.join(CONFIG['model_path'], 'evaluation_plots')}")

if __name__ == "__main__":
    main()