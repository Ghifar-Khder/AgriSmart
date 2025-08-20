import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
# Load dataset

yield_df = pd.read_csv(r"data\yield_prediction-data\yield_df.csv")

# Clean data
yield_df = yield_df.loc[:, ~yield_df.columns.str.contains('Unnamed')]
yield_df = yield_df.drop(['Year'], axis=1)

# Define features and target
X = yield_df.drop('hg/ha_yield', axis=1)
y = yield_df['hg/ha_yield']

# Identify categorical and numerical columns
cat_cols = ['Area', 'Item']
num_cols = [col for col in X.columns if col not in cat_cols]

# Initial split (train + temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Split temp into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

# Fit only on training data
preprocessor.fit(X_train)

# Get feature names after transformation
def get_feature_names(column_transformer):
    feature_names = []
    for name, transformer, features in column_transformer.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            feature_names.extend(transformer.get_feature_names_out())
    return feature_names

feature_names = get_feature_names(preprocessor)

# Transform all datasets
def preprocess_data(X_data):
    processed = preprocessor.transform(X_data)
    return pd.DataFrame(processed, columns=feature_names)

X_train_processed = preprocess_data(X_train)
X_val_processed = preprocess_data(X_val)
X_test_processed = preprocess_data(X_test)

# Add target back to create complete DataFrames
train_df = X_train_processed.copy()
train_df['hg/ha_yield'] = y_train.reset_index(drop=True)

val_df = X_val_processed.copy()
val_df['hg/ha_yield'] = y_val.reset_index(drop=True)

test_df = X_test_processed.copy()
test_df['hg/ha_yield'] = y_test.reset_index(drop=True)

# Save datasets
train_df.to_csv(r"data\yield_prediction-data\train_data.csv", index=False)
val_df.to_csv(r"data\yield_prediction-data\val_data.csv", index=False)
test_df.to_csv(r"data\yield_prediction-data\test_data.csv", index=False)

# Save preprocessor
joblib.dump(preprocessor, r"models\yield_prediction-models\preprocessor.pkl")

print("Data split and preprocessed successfully!")
print(f"Train: {len(train_df)} samples")
print(f"Validation: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")
print(f"Number of features: {len(feature_names)}")