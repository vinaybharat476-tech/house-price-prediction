import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============== OPTION 1: Use Built-in Dataset ==============
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target) * 100000  # Convert to actual prices

# ============== OPTION 2: Use Your Downloaded CSV ==============
# Uncomment this if you downloaded data from Kaggle
# X = pd.read_csv('house_data.csv')
# y = X['price']  # Target column
# X = X.drop('price', axis=1)

print("Dataset shape:", X.shape)
print("\nFirst few rows:")
print(X.head())

# ============== DATA PREPROCESSING ==============
# Handle missing values
X = X.fillna(X.mean())

# Remove outliers (optional)
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
y = y[(y >= Q1 - 1.5*IQR) & (y <= Q3 + 1.5*IQR)]
X = X.loc[y.index]

print(f"\nDataset after cleaning: {X.shape}")

# ============== TRAIN TEST SPLIT ==============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============== FEATURE SCALING ==============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============== TRAIN MODEL ==============
print("\n🚀 Training model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ============== EVALUATE ==============
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"\n✓ Model trained successfully!")
print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")

# ============== SAVE MODEL & SCALER ==============
pickle.dump(model, open('house_price_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(X.columns, open('feature_names.pkl', 'wb'))

print("\n✓ Model saved as 'house_price_model.pkl'")
print("✓ Scaler saved as 'scaler.pkl'")
print("✓ Feature names saved as 'feature_names.pkl'")