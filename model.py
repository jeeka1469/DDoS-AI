# %%
import numpy as np
import pandas as pd
import os
import glob
import gc  # For garbage collection

dataset_dir = 'C:/Users/srujan/DDOS_AI/03-11/'

print("Available CSV files:")
csv_files = []
for filename in os.listdir(dataset_dir):
    if filename.endswith('.csv'):
        csv_files.append(filename)
        print(os.path.join(dataset_dir, filename))

print(f"\nFound {len(csv_files)} CSV files to process")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib
import warnings
warnings.filterwarnings('ignore')

# Memory optimization settings
pd.set_option('mode.copy_on_write', True)

# %%
class UnifiedDDoSAnalysis:
    def __init__(self, dataset_dir, max_samples_per_file=50000):
        self.dataset_dir = dataset_dir
        self.csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        self.max_samples_per_file = max_samples_per_file  # Limit samples per file
        self.unified_data = None
        self.scaler = None
        self.best_model = None
        self.feature_columns = None  # Store feature column names
        self.label_encoders = {}  # Store label encoders
        
    def load_all_datasets(self, target_column=' Label'):
        """Load and combine all CSV files into one unified dataset"""
        print(f"Loading and combining {len(self.csv_files)} CSV files...")
        
        all_dataframes = []
        total_samples = 0
        
        for csv_file in self.csv_files:
            file_path = os.path.join(self.dataset_dir, csv_file)
            print(f"\nProcessing: {csv_file}")
            
            try:
                # Load data with sample limit per file
                df = pd.read_csv(file_path, low_memory=False)
                print(f"  Original size: {len(df)} rows")
                
                # Sample data if too large
                if len(df) > self.max_samples_per_file:
                    # Stratified sampling to maintain class distribution
                    if target_column in df.columns:
                        df = df.groupby(target_column, group_keys=False).apply(
                            lambda x: x.sample(min(len(x), self.max_samples_per_file // df[target_column].nunique()))
                        ).reset_index(drop=True)
                    else:
                        df = df.sample(n=self.max_samples_per_file, random_state=42)
                
                print(f"  Sampled size: {len(df)} rows")
                
                # Add source file identifier
                df['source_file'] = csv_file.replace('.csv', '')
                
                all_dataframes.append(df)
                total_samples += len(df)
                
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
                continue
        
        if not all_dataframes:
            print("No data loaded successfully!")
            return None
        
        # Combine all dataframes
        print(f"\nCombining all datasets...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Combined dataset shape: {combined_df.shape}")
        print(f"Total samples: {len(combined_df)}")
        
        # Clean up individual dataframes
        del all_dataframes
        gc.collect()
        
        return combined_df
    
    def preprocess_unified_data(self, df, target_column=' Label'):
        """Preprocess the unified dataset"""
        print(f"\nPreprocessing unified dataset...")
        
        # Check target column
        if target_column not in df.columns:
            print(f"Warning: Target column '{target_column}' not found. Available columns: {list(df.columns)}")
            return None, None, None, None
        
        # Show class distribution
        print("Label distribution in unified dataset:")
        label_counts = df[target_column].value_counts()
        print(label_counts)
        print(f"Label percentages:")
        print(df[target_column].value_counts(normalize=True) * 100)
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        # Remove target and source_file from categorical processing
        categorical_cols = [col for col in categorical_cols if col not in [target_column, 'source_file']]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Drop unnecessary columns
        columns_to_drop = ['Timestamp', 'source_file']
        df = df.drop(columns_to_drop, axis=1, errors='ignore')
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature column names for later use
        self.feature_columns = list(X.columns)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns stored: {len(self.feature_columns)}")
        
        # Handle NaN and infinite values
        if X.isnull().any().any() or np.isinf(X.values).any():
            print("Handling NaN/Infinite values...")
            X = X.fillna(X.median())
            X = X.replace([np.inf, -np.inf], [X.max().max(), X.min().min()])
            X = X.clip(lower=-1e6, upper=1e6)
        
        # Convert to float32 to save memory
        X = X.astype(np.float32)
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print("Stratified split successful")
        except ValueError:
            # If stratify fails, do regular split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print("Regular split used (stratify failed)")
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)
        
        return X_train_scaled, X_test_scaled
    
    def train_best_model(self, X_train, X_test, y_train, y_test):
        """Train the best performing model"""
        print(f"\nTraining unified model...")
        print(f"Training set shape: {X_train.shape}")
        
        # Model configurations - Random Forest and SVM only
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [15, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            }
        }
        
        # Skip SVM for very large datasets
        if X_train.shape[0] < 100000:
            models['SVM'] = {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
        else:
            print("Skipping SVM due to large dataset size")
        
        best_model = None
        best_score = 0
        best_model_name = ""
        results = {}
        
        for name, setup in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Use fewer CV folds for large datasets
                cv_folds = 3 if X_train.shape[0] > 50000 else 5
                
                grid_search = GridSearchCV(
                    estimator=setup['model'],
                    param_grid=setup['params'],
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                current_model = grid_search.best_estimator_
                y_pred = current_model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'model': current_model,
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'test_accuracy': accuracy,
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                print(f"{name} Results:")
                print(f"  Best CV Score: {grid_search.best_score_:.4f}")
                print(f"  Test Accuracy: {accuracy:.4f}")
                print(f"  Best Params: {grid_search.best_params_}")
                
                # Track best model
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = current_model
                    best_model_name = name
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
            
            # Force garbage collection after each model
            gc.collect()
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"BEST ACCURACY: {best_score:.4f}")
        print(f"{'='*60}")
        
        self.best_model = best_model
        return results, best_model, best_model_name, best_score
    
    def save_model_and_scaler(self, model_name="unified_ddos_model"):
        """Save the best model, scaler, and metadata"""
        if self.best_model is not None and self.scaler is not None:
            model_filename = f"{model_name}.pkl"
            scaler_filename = f"{model_name}_scaler.pkl"
            metadata_filename = f"{model_name}_metadata.pkl"
            
            # Save model
            joblib.dump(self.best_model, model_filename)
            print(f"Model saved as: {model_filename}")
            
            # Save scaler
            joblib.dump(self.scaler, scaler_filename)
            print(f"Scaler saved as: {scaler_filename}")
            
            # Save metadata (feature columns and label encoders)
            metadata = {
                'feature_columns': self.feature_columns,
                'label_encoders': self.label_encoders,
                'n_features': len(self.feature_columns) if self.feature_columns else None
            }
            joblib.dump(metadata, metadata_filename)
            print(f"Metadata saved as: {metadata_filename}")
            
            return model_filename, scaler_filename, metadata_filename
        else:
            print("No model or scaler to save!")
            return None, None, None

# %%
# Initialize the unified analyzer
analyzer = UnifiedDDoSAnalysis(dataset_dir, max_samples_per_file=30000)

print(f"CSV files found: {analyzer.csv_files}")

# %%
# Load and combine all datasets
combined_df = analyzer.load_all_datasets()

if combined_df is not None:
    print(f"\nUnified dataset created successfully!")
    print(f"Shape: {combined_df.shape}")
    
    # Show source file distribution
    print("\nSamples per source file:")
    print(combined_df['source_file'].value_counts())
else:
    print("Failed to create unified dataset!")

# %%
# Preprocess the unified data
if combined_df is not None:
    X_train, X_test, y_train, y_test = analyzer.preprocess_unified_data(combined_df)
    
    if X_train is not None:
        # Scale features
        X_train_scaled, X_test_scaled = analyzer.scale_features(X_train, X_test)
        
        print(f"\nData preprocessing completed!")
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        # Clean up original dataframe
        del combined_df
        gc.collect()
    else:
        print("Preprocessing failed!")

# %%
# Train the best model on unified data
if 'X_train_scaled' in locals():
    results, best_model, best_model_name, best_score = analyzer.train_best_model(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED - UNIFIED MODEL RESULTS")
    print(f"{'='*80}")
    
    print(f"Best Model: {best_model_name}")
    print(f"Best Accuracy: {best_score:.4f}")
    
    print(f"\nAll Model Results:")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  CV Score: {result['best_cv_score']:.4f}")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
    
    # Show detailed results for best model
    if best_model_name in results:
        print(f"\nDetailed Results for {best_model_name}:")
        print(results[best_model_name]['classification_report'])

# %%
# Save the best model and scaler
def save_model_and_scaler(self, model_name="unified_ddos_model"):
    """Save the best model, scaler, and metadata"""
    if self.best_model is not None and self.scaler is not None:
        model_filename = f"{model_name}.pkl"
        scaler_filename = f"{model_name}_scaler.pkl"
        metadata_filename = f"{model_name}_metadata.pkl"

        # Save model
        joblib.dump(self.best_model, model_filename)
        print(f"Model saved as: {model_filename}")

        # Save scaler
        joblib.dump(self.scaler, scaler_filename)
        print(f"Scaler saved as: {scaler_filename}")

        # ✅ Save metadata as a dictionary
        metadata = {
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'n_features': len(self.feature_columns) if self.feature_columns else None
        }
        joblib.dump(metadata, metadata_filename)
        print(f"Metadata saved as: {metadata_filename}")

        return model_filename, scaler_filename, metadata_filename
    else:
        print("No model or scaler to save!")
        return None, None, None
    
if model_filename and scaler_filename and metadata_filename:
    print(f"\n{'='*60}")
    print("MODEL TRAINING AND SAVING COMPLETED!")
    print(f"{'='*60}")
    print(f"✓ Unified model trained on {len(analyzer.csv_files)} CSV files")
    print(f"✓ Best model: {best_model_name}")
    print(f"✓ Best accuracy: {best_score:.4f}")
    print(f"✓ Model file saved to: {model_file}")
    print(f"✓ Scaler file saved to: {scaler_file}")
    print(f"✓ Metadata file saved to: {metadata_file}")
    print(f"✓ Total feature columns: {len(analyzer.feature_columns) if analyzer.feature_columns else 0}")

    # Show feature importance if available
    if getattr(analyzer.best_model, 'feature_importances_', None) is not None and analyzer.feature_columns:
        importance_df = pd.DataFrame({
            'feature': analyzer.feature_columns,
            'importance': analyzer.best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

# %% Memory usage summary
import psutil
import os

process = psutil.Process(os.getpid())
memory_info = process.memory_info()

print(f"\n{'='*60}")
print("MEMORY USAGE SUMMARY")
print(f"{'='*60}")
print(f"Resident Set Size (RSS): {memory_info.rss / 1024**2:.2f} MB")
print(f"Virtual Memory Size (VMS): {memory_info.vms / 1024**2:.2f} MB")
print(f"Available System Memory: {psutil.virtual_memory().available / 1024**2:.2f} MB")

print(f"\n✔ SUCCESS: Unified model training and saving process completed.")
