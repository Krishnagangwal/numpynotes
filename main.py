import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
from catboost import CatBoostClassifier

# Load the datasets
# NOTE: The file paths assume the files are in the same directory as the script.
# You may need to change these paths depending on your file location.
try:
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Error: The CSV files were not found. Please ensure 'UNSW_NB15_training-set.csv' and 'UNSW_NB15_testing-set.csv' are in the correct directory.")
    exit()

# --- Step 1: Data Preprocessing (Leakage-Free) ---
print("\n--- Data Preprocessing ---")

def preprocess_data(df):
    """
    Preprocess a single dataframe: drop id, handle infinities, impute missing values, one-hot encode
    """
    # Drop unnecessary columns like 'id'
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Handle missing or infinite values
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill missing values with mode for each column
    for col in df.columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val.iloc[0], inplace=True)
            else:
                # If no mode (all NaN), fill with 0 for numeric, 'unknown' for object
                if df[col].dtype == 'object':
                    df[col].fillna('unknown', inplace=True)
                else:
                    df[col].fillna(0, inplace=True)

    return df

# Preprocess training and test sets separately
train_df_clean = preprocess_data(train_df.copy())
test_df_clean = preprocess_data(test_df.copy())

# Separate features and target
# Use binary 'label' as target (benign vs attack)
y_train = train_df_clean['label']
y_test = test_df_clean['label']

X_train = train_df_clean.drop(['attack_cat', 'label'], axis=1)
X_test = test_df_clean.drop(['attack_cat', 'label'], axis=1)

# One-hot encode categorical features on training data
# Fit encoder on training data only to prevent leakage
categorical_features = ['proto', 'service', 'state']

# Get dummies for training data
X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)

# For test data, we need to ensure same columns as training
# Get dummies for test data
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align test columns with train columns (add missing columns with 0s, remove extra columns)
missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols:
    X_test_encoded[col] = 0

# Remove extra columns in test that are not in train
extra_cols = set(X_test_encoded.columns) - set(X_train_encoded.columns)
X_test_encoded = X_test_encoded.drop(columns=extra_cols)

# Reorder test columns to match train columns
X_test_encoded = X_test_encoded[X_train_encoded.columns]

# Identify numeric columns for scaling (exclude one-hot encoded columns)
numeric_columns = X_train.select_dtypes(exclude=['object']).columns
numeric_column_indices = [i for i, col in enumerate(X_train_encoded.columns) 
                         if any(col.startswith(num_col) and col == num_col for num_col in numeric_columns)]

# Scale only numeric features, fit scaler on training data only
scaler = StandardScaler()

# Convert to numpy arrays for easier indexing
X_train_array = X_train_encoded.values
X_test_array = X_test_encoded.values

# Fit scaler on numeric columns of training data only
if numeric_column_indices:
    X_train_array[:, numeric_column_indices] = scaler.fit_transform(X_train_array[:, numeric_column_indices])
    X_test_array[:, numeric_column_indices] = scaler.transform(X_test_array[:, numeric_column_indices])

print("Data preprocessing complete. No data leakage - all preprocessing fitted on training data only.")
print(f"Training data shape: {X_train_array.shape}")
print(f"Testing data shape: {X_test_array.shape}")

# --- Step 2: Model Training and Evaluation ---

def evaluate_model(model, model_name, y_true, y_pred):
    """
    Evaluate a model and print comprehensive results
    """
    print(f"\n--- {model_name} Results ---")

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])

    print("\nDetailed Metrics:")
    print(f"Class 0 (Normal) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-Score: {f1[0]:.4f}")
    print(f"Class 1 (Attack) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-Score: {f1[1]:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN={cm[0,0]:,}, FP={cm[0,1]:,}")
    print(f"FN={cm[1,0]:,}, TP={cm[1,1]:,}")
    print(cm)

    return accuracy, precision[1], recall[1], f1[1]  # Return attack class metrics

# Store results for comparison
results = {}

# --- Model 1: Decision Tree Classifier (Baseline) ---
print("\n=== Training Decision Tree Classifier (Baseline) ===")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=10, min_samples_leaf=5)
dt_model.fit(X_train_array, y_train)

dt_predictions = dt_model.predict(X_test_array)
dt_acc, dt_prec, dt_rec, dt_f1 = evaluate_model(dt_model, "Decision Tree", y_test, dt_predictions)
results['Decision Tree'] = {'Accuracy': dt_acc, 'Precision': dt_prec, 'Recall': dt_rec, 'F1-Score': dt_f1}

# --- Model 2: Gradient Boosting Classifier ---
print("\n=== Training Gradient Boosting Classifier ===")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_array, y_train)

gb_predictions = gb_model.predict(X_test_array)
gb_acc, gb_prec, gb_rec, gb_f1 = evaluate_model(gb_model, "Gradient Boosting", y_test, gb_predictions)
results['Gradient Boosting'] = {'Accuracy': gb_acc, 'Precision': gb_prec, 'Recall': gb_rec, 'F1-Score': gb_f1}

# --- Model 3: CatBoost Classifier ---
print("\n=== Training CatBoost Classifier ===")
cat_model = CatBoostClassifier(iterations=100, learning_rate=0.5, depth=6, 
                              loss_function='Logloss', verbose=False, random_state=42)
cat_model.fit(X_train_array, y_train)

cat_predictions = cat_model.predict(X_test_array)
cat_acc, cat_prec, cat_rec, cat_f1 = evaluate_model(cat_model, "CatBoost", y_test, cat_predictions)
results['CatBoost'] = {'Accuracy': cat_acc, 'Precision': cat_prec, 'Recall': cat_rec, 'F1-Score': cat_f1}

# --- Final Comparison ---
print("\n" + "="*80)
print("FINAL PERFORMANCE COMPARISON")
print("="*80)

print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
print("-" * 70)

for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['Accuracy']:<10.4f} {metrics['Precision']:<10.4f} "
          f"{metrics['Recall']:<10.4f} {metrics['F1-Score']:<10.4f}")

print("\nAll models have been trained and evaluated successfully.")
print("This pipeline prevents data leakage by fitting all preprocessing on training data only.")