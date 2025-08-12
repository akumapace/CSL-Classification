def predict_new_patient(self, patient_data, model_name='CatBoost'):
        """
        Make prediction for a new patient.
        
        Parameters:
        -----------
        patient_data : dict or pd.DataFrame
            Patient data with same features as training data
        model_name : str
            Name of the model to use for prediction
            
        Returns:
        --------
        dict : Prediction results with probabilities
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        # Convert to DataFrame if dict
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Apply same feature engineering
        patient_df = self.create_clinical_features(patient_df)
        
        # Remove target column if present
        target_cols = ['diagnosis', 'target', 'label']
        for col in target_cols:
            if col in patient_df.columns:
                patient_df = patient_df.drop(columns=[col])
        
        # Use appropriate preprocessing based on model
        if model_name == 'CatBoost':
            # Handle missing values and ensure string format for categorical features
            for col in self.categorical_features:
                if col in patient_df.columns:
                    if patient_df[col].isna().any():
                        mode_val = 'Unknown'  # Default value
                        patient_df[col].fillna(mode_val, inplace=True)
                    patient_df[col] = patient_df[col].astype(str)
            
            for col in self.numerical_features:
                if col in patient_df.columns and patient_df[col].isna().any():
                    patient_df[col].fillna(0, inplace=True)  # Default value
            
            patient_processed = patient_df
        else:
            # Use standard preprocessing pipeline
            patient_processed = self.preprocessor.transform(patient_df)
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(patient_processed)[0]
        probabilities = model.predict_proba(patient_processed)[0]
        
        # Convert back to original labels
        predicted_class = self.target_encoder.inverse_transform([prediction])[0]
        
        result = {
            'predicted_class': predicted_class,
            'prediction_confidence': max(probabilities),
            'class_probabilities': dict(zip(self.target_encoder.classes_, probabilities))
        }
        
        return result


"""
PID/SID Classification Framework
A comprehensive machine learning framework for predicting Primary Immunodeficiency (PID) 
and Secondary Immunodeficiency (SID) vs non-immunodeficient patients.

Author: Medical ML Framework
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_curve, roc_auc_score,
                           f1_score, precision_score, recall_score, roc_curve)

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Imbalance handling
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Categorical encoding
from category_encoders import TargetEncoder

# Visualization and interpretation
import matplotlib.pyplot as plt
import seaborn as sns
import shap
# Note: LIME would be imported as: import lime

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class PIDSIDClassifier:
    """
    A comprehensive classification framework for PID/SID prediction.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.preprocessors = {}
        self.feature_names = None
        self.target_encoder = None
        self.evaluation_results = {}
        self.categorical_features = []
        self.numerical_features = []
        
    def load_and_explore_data(self, data_path=None, df=None):
        """
        Load and perform initial exploration of the dataset.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the dataset file
        df : pd.DataFrame, optional
            Dataframe to use directly
            
        Returns:
        --------
        pd.DataFrame : The loaded dataset
        """
        if df is not None:
            self.df = df.copy()
        elif data_path is not None:
            if data_path.endswith('.csv'):
                self.df = pd.read_csv(data_path)
            elif data_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(data_path)
            else:
                raise ValueError("Unsupported file format")
        else:
            raise ValueError("Either data_path or df must be provided")
            
        print("=== Dataset Overview ===")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print("\n=== Target Distribution ===")
        if 'diagnosis' in self.df.columns:
            print(self.df['diagnosis'].value_counts())
            print(self.df['diagnosis'].value_counts(normalize=True))
        
        print("\n=== Missing Values ===")
        missing_data = self.df.isnull().sum()
        print(missing_data[missing_data > 0])
        
        print("\n=== Data Types ===")
        print(self.df.dtypes.value_counts())
        
        return self.df
    
    def create_clinical_features(self, df):
        """
        Create clinically relevant features based on domain knowledge.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame : DataFrame with engineered features
        """
        df = df.copy()
        
        # Example feature engineering (customize based on your actual features)
        
        # Warning signs count (based on 10 warning signs of PID)
        warning_sign_cols = [col for col in df.columns if 'infection' in col.lower() 
                           or 'pneumonia' in col.lower() or 'antibiotic' in col.lower()
                           or 'family_history' in col.lower()]
        
        if warning_sign_cols:
            df['warning_signs_count'] = df[warning_sign_cols].sum(axis=1)
        
        # Family history indicator
        family_history_cols = [col for col in df.columns if 'family' in col.lower() 
                             or 'consanguinity' in col.lower()]
        if family_history_cols:
            df['has_family_history'] = df[family_history_cols].any(axis=1).astype(int)
        
        # Infection severity score
        infection_cols = [col for col in df.columns if 'severe' in col.lower() 
                         and 'infection' in col.lower()]
        if infection_cols:
            df['severe_infection_score'] = df[infection_cols].sum(axis=1)
        
        # Age-related features
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 5, 18, 65, 100], 
                                   labels=['infant', 'child', 'adult', 'elderly'])
        
        print(f"Created {len(df.columns) - len(self.df.columns)} new features")
        return df
    
    def preprocess_data(self, target_col='diagnosis', test_size=0.2, use_catboost_encoding=True):
        """
        Comprehensive data preprocessing pipeline optimized for categorical features.
        
        Parameters:
        -----------
        target_col : str
            Name of the target column
        test_size : float
            Proportion of data for testing
        use_catboost_encoding : bool
            If True, use minimal preprocessing for CatBoost (keeps categorical as-is)
        """
        df = self.create_clinical_features(self.df)
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        # Identify categorical and numerical columns
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical columns ({len(self.categorical_features)}): {self.categorical_features}")
        print(f"Numerical columns ({len(self.numerical_features)}): {self.numerical_features}")
        
        # Split data first
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, 
            stratify=y_encoded, random_state=self.random_state
        )
        
        if use_catboost_encoding:
            # For CatBoost: minimal preprocessing, keep categorical features as strings
            # Handle missing values only
            X_train_cat = self.X_train.copy()
            X_test_cat = self.X_test.copy()
            
            # Fill missing values
            for col in self.categorical_features:
                mode_val = X_train_cat[col].mode()[0] if not X_train_cat[col].mode().empty else 'Unknown'
                X_train_cat[col].fillna(mode_val, inplace=True)
                X_test_cat[col].fillna(mode_val, inplace=True)
                
                # Ensure all values are strings for CatBoost
                X_train_cat[col] = X_train_cat[col].astype(str)
                X_test_cat[col] = X_test_cat[col].astype(str)
            
            for col in self.numerical_features:
                mean_val = X_train_cat[col].mean()
                X_train_cat[col].fillna(mean_val, inplace=True)
                X_test_cat[col].fillna(mean_val, inplace=True)
            
            # Store CatBoost-ready data
            self.X_train_catboost = X_train_cat
            self.X_test_catboost = X_test_cat
            self.categorical_feature_indices = [i for i, col in enumerate(X_train_cat.columns) 
                                              if col in self.categorical_features]
        
        # Create preprocessing pipelines for other models
        # For categorical features - use target encoding for high cardinality
        from category_encoders import TargetEncoder
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target_encoder', TargetEncoder())
        ])
        
        # For numerical features (if any)
        numerical_transformer = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Combine preprocessors
        self.preprocessor = ColumnTransformer([
            ('cat', categorical_transformer, self.categorical_features),
            ('num', numerical_transformer, self.numerical_features)
        ])
        
        # Fit preprocessor and transform data
        X_train_processed = self.preprocessor.fit_transform(self.X_train, self.y_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Store processed data
        self.X_train_processed = X_train_processed
        self.X_test_processed = X_test_processed
        
        # Get feature names after preprocessing
        self.feature_names = (list(self.categorical_features) + 
                            list(self.numerical_features))
        
        print(f"Training set shape: {X_train_processed.shape}")
        print(f"Test set shape: {X_test_processed.shape}")
        print(f"Class distribution in training: {np.bincount(self.y_train)}")
        print(f"CatBoost-ready data shape: {self.X_train_catboost.shape}")
        
    def handle_class_imbalance(self, method='smote'):
        """
        Handle class imbalance using various techniques.
        
        Parameters:
        -----------
        method : str
            Method to handle imbalance ('smote', 'undersample', 'smote_tomek', 'class_weight')
        """
        if method == 'smote':
            # For mixed data types, use SMOTE-NC if you have categorical features
            categorical_indices = list(range(len([col for col in self.X_train.columns 
                                               if self.X_train[col].dtype == 'object'])))
            
            if categorical_indices:
                smote = SMOTENC(categorical_features=categorical_indices, 
                               random_state=self.random_state)
            else:
                smote = SMOTE(random_state=self.random_state)
                
            self.X_train_processed, self.y_train = smote.fit_resample(
                self.X_train_processed, self.y_train)
            
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=self.random_state)
            self.X_train_processed, self.y_train = undersampler.fit_resample(
                self.X_train_processed, self.y_train)
            
        elif method == 'smote_tomek':
            smote_tomek = SMOTETomek(random_state=self.random_state)
            self.X_train_processed, self.y_train = smote_tomek.fit_resample(
                self.X_train_processed, self.y_train)
        
        print(f"After {method}: {np.bincount(self.y_train)}")
    
    def initialize_models(self):
        """Initialize various models for comparison, optimized for categorical features."""
        self.models = {
            'CatBoost': CatBoostClassifier(
                random_state=self.random_state,
                verbose=False,
                class_weights='Balanced',
                iterations=500,
                learning_rate=0.1,
                depth=6,
                cat_features=self.categorical_feature_indices if hasattr(self, 'categorical_feature_indices') else None
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='mlogloss',
                scale_pos_weight=2  # Helps with imbalanced data
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1,
                class_weight='balanced'
            ),
            'LogisticRegression': LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            ),
            'GradientBoosting': GradientBoostingClassifier(
                random_state=self.random_state
            )
        }
    
    def train_and_evaluate_models(self, cv_folds=5):
        """
        Train and evaluate all models using cross-validation.
        Handles CatBoost separately due to its categorical feature requirements.
        
        Parameters:
        -----------
        cv_folds : int
            Number of cross-validation folds
        """
        if not hasattr(self, 'models') or not self.models:
            self.initialize_models()
        
        results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Use appropriate data based on model type
            if name == 'CatBoost':
                X_train_model = self.X_train_catboost
                X_test_model = self.X_test_catboost
                
                # Cross-validation for CatBoost
                cv_scores = []
                for train_idx, val_idx in cv.split(X_train_model, self.y_train):
                    X_cv_train, X_cv_val = X_train_model.iloc[train_idx], X_train_model.iloc[val_idx]
                    y_cv_train, y_cv_val = self.y_train[train_idx], self.y_train[val_idx]
                    
                    model_cv = CatBoostClassifier(
                        random_state=self.random_state,
                        verbose=False,
                        class_weights='Balanced',
                        iterations=500,
                        cat_features=self.categorical_feature_indices
                    )
                    model_cv.fit(X_cv_train, y_cv_train)
                    y_pred_cv = model_cv.predict(X_cv_val)
                    cv_scores.append(f1_score(y_cv_val, y_pred_cv, average='macro'))
                
                cv_scores = np.array(cv_scores)
                
                # Fit final model
                model.fit(X_train_model, self.y_train)
                y_pred = model.predict(X_test_model)
                y_pred_proba = model.predict_proba(X_test_model)
                
            else:
                X_train_model = self.X_train_processed
                X_test_model = self.X_test_processed
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train_model, self.y_train, 
                                          cv=cv, scoring='f1_macro')
                
                # Fit model
                model.fit(X_train_model, self.y_train)
                y_pred = model.predict(X_test_model)
                y_pred_proba = model.predict_proba(X_test_model)
            
            # Calculate metrics
            results[name] = {
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'test_accuracy': accuracy_score(self.y_test, y_pred),
                'test_precision': precision_score(self.y_test, y_pred, average='macro'),
                'test_recall': recall_score(self.y_test, y_pred, average='macro'),
                'test_f1': f1_score(self.y_test, y_pred, average='macro'),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # ROC AUC for multiclass
            try:
                results[name]['test_auc'] = roc_auc_score(self.y_test, y_pred_proba, 
                                                        multi_class='ovr', average='macro')
            except:
                results[name]['test_auc'] = np.nan
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                results[name]['feature_importances'] = model.feature_importances_
            elif name == 'CatBoost':
                results[name]['feature_importances'] = model.get_feature_importance()
        
        self.evaluation_results = results
        return results
    
    def print_evaluation_summary(self):
        """Print a comprehensive evaluation summary."""
        if not self.evaluation_results:
            print("No evaluation results available. Run train_and_evaluate_models() first.")
            return
        
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        for name, results in self.evaluation_results.items():
            summary_data.append({
                'Model': name,
                'CV F1 (mean±std)': f"{results['cv_f1_mean']:.3f}±{results['cv_f1_std']:.3f}",
                'Test Accuracy': f"{results['test_accuracy']:.3f}",
                'Test Precision': f"{results['test_precision']:.3f}",
                'Test Recall': f"{results['test_recall']:.3f}",
                'Test F1': f"{results['test_f1']:.3f}",
                'Test AUC': f"{results['test_auc']:.3f}" if not np.isnan(results['test_auc']) else "N/A"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x]['test_f1'])
        print(f"\nBest Model (by F1 Score): {best_model_name}")
        
        # Detailed report for best model
        best_results = self.evaluation_results[best_model_name]
        print(f"\n=== Detailed Results for {best_model_name} ===")
        print("Classification Report:")
        print(classification_report(self.y_test, best_results['predictions'], 
                                  target_names=self.target_encoder.classes_))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, best_results['predictions'])
        print(cm)
    
    def plot_evaluation_metrics(self):
        """Plot various evaluation metrics for model comparison."""
        if not self.evaluation_results:
            print("No evaluation results available.")
            return
        
        # Prepare data for plotting
        models = list(self.evaluation_results.keys())
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.evaluation_results[model][metric] for model in models]
            axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("test_", "").title()}')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Plot ROC curves if available
        plt.figure(figsize=(10, 8))
        for name, results in self.evaluation_results.items():
            if not np.isnan(results['test_auc']):
                # This is simplified - you'd need to compute ROC for each class
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.text(0.5, 0.5, f"{name}: AUC={results['test_auc']:.3f}", 
                        transform=plt.gca().transAxes)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.show()
    
    def hyperparameter_tuning(self, model_name='CatBoost', search_type='grid'):
        """
        Perform hyperparameter tuning for a specific model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to tune
        search_type : str
            'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        # Define parameter grids for different models
        param_grids = {
            'CatBoost': {
                'iterations': [300, 500, 800],
                'learning_rate': [0.05, 0.1, 0.2],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs']
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return
        
        # Get appropriate training data
        if model_name == 'CatBoost':
            X_train_model = self.X_train_catboost
            base_model = CatBoostClassifier(
                random_state=self.random_state,
                verbose=False,
                class_weights='Balanced',
                cat_features=self.categorical_feature_indices
            )
        else:
            X_train_model = self.X_train_processed
            base_model = self.models[model_name]
        
        param_grid = param_grids[model_name]
        
        print(f"Tuning hyperparameters for {model_name}...")
        
        if search_type == 'grid':
            search = GridSearchCV(
                base_model, param_grid, 
                cv=3, scoring='f1_macro', 
                n_jobs=-1, verbose=1
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_grid, 
                n_iter=20, cv=3, scoring='f1_macro', 
                n_jobs=-1, verbose=1, random_state=self.random_state
            )
        
        search.fit(X_train_model, self.y_train)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score: {search.best_score_:.3f}")
        
        # Update the model with best parameters
        self.models[model_name] = search.best_estimator_
        
        return search.best_estimator_, search.best_params_
    
    def explain_predictions(self, model_name='XGBoost', sample_size=100):
        """
        Generate explanations for model predictions using SHAP.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to explain
        sample_size : int
            Number of samples to use for explanation
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        model = self.models[model_name]
        
        # Create SHAP explainer
        # Use a subset of training data as background for efficiency
        background = self.X_train_processed[:min(100, len(self.X_train_processed))]
        explainer = shap.Explainer(model, background)
        
        # Generate explanations for test set sample
        test_sample = self.X_test_processed[:sample_size]
        shap_values = explainer(test_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, test_sample, feature_names=self.feature_names)
        plt.show()
        
        # Feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, test_sample, feature_names=self.feature_names, 
                         plot_type="bar")
        plt.show()
        
    def plot_feature_importance(self, model_name='CatBoost', top_n=15):
        """
        Plot feature importance for tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to analyze
        top_n : int
            Number of top features to display
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        if 'feature_importances' not in self.evaluation_results[model_name]:
            print(f"No feature importance available for {model_name}")
            return
        
        importances = self.evaluation_results[model_name]['feature_importances']
        
        if model_name == 'CatBoost':
            feature_names = list(self.X_train_catboost.columns)
        else:
            feature_names = self.feature_names
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def save_model(self, model_name='XGBoost', file_path='best_model.pkl'):
        """
        Save the trained model and preprocessor.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
        file_path : str
            Path to save the model
        """
        import pickle
        
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        model_package = {
            'model': self.models[model_name],
            'preprocessor': self.preprocessor,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        """
        Load a saved model.
        
        Parameters:
        -----------
        file_path : str
            Path to the saved model
        """
        import pickle
        
        with open(file_path, 'rb') as f:
            model_package = pickle.load(f)
        
        self.preprocessor = model_package['preprocessor']
        self.target_encoder = model_package['target_encoder']
        self.feature_names = model_package['feature_names']
        
        # Add loaded model to models dictionary
        self.models['loaded_model'] = model_package['model']
        
        print("Model loaded successfully")
    
    def predict_new_patient(self, patient_data, model_name='XGBoost'):
        """
        Make prediction for a new patient.
        
        Parameters:
        -----------
        patient_data : dict or pd.DataFrame
            Patient data with same features as training data
        model_name : str
            Name of the model to use for prediction
            
        Returns:
        --------
        dict : Prediction results with probabilities
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return None
        
        # Convert to DataFrame if dict
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Apply same feature engineering
        patient_df = self.create_clinical_features(patient_df)
        
        # Remove target column if present
        target_cols = ['diagnosis', 'target', 'label']
        for col in target_cols:
            if col in patient_df.columns:
                patient_df = patient_df.drop(columns=[col])
        
        # Preprocess
        patient_processed = self.preprocessor.transform(patient_df)
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(patient_processed)[0]
        probabilities = model.predict_proba(patient_processed)[0]
        
        # Convert back to original labels
        predicted_class = self.target_encoder.inverse_transform([prediction])[0]
        
        result = {
            'predicted_class': predicted_class,
            'prediction_confidence': max(probabilities),
            'class_probabilities': dict(zip(self.target_encoder.classes_, probabilities))
        }
        
        return result

# Example usage and demonstration
def demonstrate_framework():
    """
    Demonstrate the framework with synthetic data.
    """
    print("PID/SID Classification Framework Demonstration")
    print("=" * 50)
    
    # Create synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic features
    data = {
        'age': np.random.randint(1, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'recurrent_infections': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'severe_infections': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
        'family_history': np.random.choice(['Yes', 'No'], n_samples, p=[0.15, 0.85]),
        'autoimmune_disease': np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9]),
        'consanguinity': np.random.choice(['Yes', 'No'], n_samples, p=[0.05, 0.95]),
        'failure_to_thrive': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
        'antibiotic_resistance': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
    }
    
    # Create target with realistic distribution
    targets = []
    for i in range(n_samples):
        # Create some logic to make realistic targets
        risk_score = (
            (data['recurrent_infections'][i] == 'Yes') * 2 +
            (data['severe_infections'][i] == 'Yes') * 2 +
            (data['family_history'][i] == 'Yes') * 1.5 +
            (data['autoimmune_disease'][i] == 'Yes') * 1 +
            (data['consanguinity'][i] == 'Yes') * 1.5 +
            np.random.random()  # Add some randomness
        )
        
        if risk_score > 3.5:
            targets.append('PID')
        elif risk_score > 2:
            targets.append('SID')
        else:
            targets.append('Non-immunodeficient')
    
    data['diagnosis'] = targets
    df = pd.DataFrame(data)
    
    print("Synthetic dataset created with shape:", df.shape)
    print("Target distribution:")
    print(df['diagnosis'].value_counts())
    
    # Initialize and run the framework
    classifier = PIDSIDClassifier(random_state=42)
    
    # Step 1: Load and explore data
    classifier.load_and_explore_data(df=df)
    
    # Step 2: Preprocess data
    classifier.preprocess_data(target_col='diagnosis')
    
    # Step 3: Handle class imbalance
    classifier.handle_class_imbalance(method='smote')
    
    # Step 4: Train and evaluate models
    results = classifier.train_and_evaluate_models(cv_folds=3)
    
    # Step 5: Print evaluation summary
    classifier.print_evaluation_summary()
    
    # Step 6: Hyperparameter tuning for best model (CatBoost)
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    classifier.hyperparameter_tuning(model_name='CatBoost', search_type='random')
    
    # Re-evaluate after tuning
    results_tuned = classifier.train_and_evaluate_models(cv_folds=3)
    classifier.print_evaluation_summary()
    
    # Step 7: Feature importance analysis
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    feature_importance_df = classifier.plot_feature_importance(model_name='CatBoost', top_n=10)
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df)
    
    # Step 8: Save the best model
    classifier.save_model(model_name='CatBoost', file_path='pid_sid_catboost_model.pkl')
    
    # Step 9: Test prediction on new patient
    new_patient = {
        'age': 25,
        'gender': 'M',
        'recurrent_infections': 'Yes',
        'severe_infections': 'Yes',
        'family_history': 'No',
        'autoimmune_disease': 'No',
        'consanguinity': 'No',
        'failure_to_thrive': 'No',
        'antibiotic_resistance': 'Yes'
    }
    
    prediction = classifier.predict_new_patient(new_patient, model_name='CatBoost')
    print("\n=== New Patient Prediction ===")
    print(f"Patient data: {new_patient}")
    print(f"Predicted class: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['prediction_confidence']:.3f}")
    print("Class probabilities:")
    for class_name, prob in prediction['class_probabilities'].items():
        print(f"  {class_name}: {prob:.3f}")
    
    return classifier

if __name__ == "__main__":
    # Run demonstration
    classifier = demonstrate_framework()