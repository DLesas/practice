"""
Solution: ML Model Pipeline Development
Demonstrates production-ready ML pipeline for churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionPipeline:
    """
    Complete ML pipeline for customer churn prediction
    """
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.feature_selector = None
        self.best_model = None
        self.feature_names = None
        
    def create_churn_target(self, customer_features, days_threshold=60):
        """Create churn target variable"""
        
        print("=== CREATING CHURN TARGET ===")
        
        # Calculate days since last purchase
        reference_date = datetime.now()
        
        # Convert to datetime if not already
        if 'last_purchase' in customer_features.columns:
            if customer_features['last_purchase'].dtype == 'object':
                customer_features['last_purchase'] = pd.to_datetime(customer_features['last_purchase'])
        
        # Create churn target (1 = churned, 0 = active)
        customer_features['churn'] = (customer_features['days_since_last_purchase'] > days_threshold).astype(int)
        
        churn_rate = customer_features['churn'].mean()
        print(f"Churn rate: {churn_rate:.2%}")
        print(f"Churned customers: {customer_features['churn'].sum():,}")
        print(f"Active customers: {(~customer_features['churn'].astype(bool)).sum():,}")
        
        return customer_features
    
    def engineer_ml_features(self, customer_features):
        """Engineer additional features specifically for ML"""
        
        print("\n=== ML FEATURE ENGINEERING ===")
        
        df = customer_features.copy()
        
        # 1. Interaction features
        df['avg_order_value_x_frequency'] = df['avg_order_value'] * df['total_transactions']
        df['revenue_per_day_active'] = df['total_revenue'] / (df['days_active'] + 1)  # +1 to avoid division by zero
        
        # 2. Ratio features
        df['profit_margin'] = df['total_profit'] / (df['total_revenue'] + 0.01)  # Avoid division by zero
        df['items_per_transaction'] = df['total_items'] / df['total_transactions']
        
        # 3. Behavioral indicators
        df['is_frequent_buyer'] = (df['purchase_frequency'] > df['purchase_frequency'].quantile(0.75)).astype(int)
        df['is_high_value'] = (df['total_revenue'] > df['total_revenue'].quantile(0.8)).astype(int)
        df['is_recent_customer'] = (df['days_since_last_purchase'] <= 30).astype(int)
        
        # 4. Engagement features
        if 'category_diversity' in df.columns:
            df['category_diversity_score'] = pd.qcut(df['category_diversity'], 5, labels=[1,2,3,4,5])
        
        # 5. Tenure-based features
        df['tenure_months'] = df['days_active'] / 30
        df['revenue_per_month'] = df['total_revenue'] / (df['tenure_months'] + 0.1)
        
        print(f"Created {len(df.columns) - len(customer_features.columns)} new ML features")
        
        return df
    
    def prepare_features(self, df, target_col='churn'):
        """Prepare features for ML pipeline"""
        
        print("\n=== FEATURE PREPARATION ===")
        
        # Separate features and target
        y = df[target_col]
        X = df.drop([target_col], axis=1)
        
        # Identify feature types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove datetime columns
        datetime_features = []
        for col in X.columns:
            if X[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                datetime_features.append(col)
        
        # Drop datetime features for now (could engineer more features from them)
        X = X.drop(datetime_features, axis=1)
        numeric_features = [f for f in numeric_features if f not in datetime_features]
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        print(f"Dropped datetime features: {len(datetime_features)}")
        
        # Handle missing values
        missing_summary = X.isnull().sum()
        if missing_summary.sum() > 0:
            print(f"\nMissing values in {missing_summary[missing_summary > 0].shape[0]} features")
            
        return X, y, numeric_features, categorical_features
    
    def create_preprocessor(self, numeric_features, categorical_features):
        """Create preprocessing pipeline"""
        
        print("\n=== CREATING PREPROCESSOR ===")
        
        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', LabelEncoder())  # Note: This won't work directly with Pipeline for multiple features
        ])
        
        # For simplicity, let's handle categorical features manually
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features)
        ], remainder='drop')  # Drop categorical for now, or handle separately
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def select_features(self, X, y, method='correlation', k=20):
        """Feature selection"""
        
        print(f"\n=== FEATURE SELECTION ({method}) ===")
        
        if method == 'correlation':
            # Remove highly correlated features
            corr_matrix = X.select_dtypes(include=[np.number]).corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation > 0.95
            high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            
            print(f"Removing {len(high_corr_features)} highly correlated features")
            X_selected = X.drop(high_corr_features, axis=1)
            
        elif method == 'univariate':
            # Univariate feature selection
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            self.feature_selector = selector
            
        elif method == 'rfe':
            # Recursive feature elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.support_]
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            self.feature_selector = selector
        
        print(f"Selected {X_selected.shape[1]} features from {X.shape[1]}")
        self.feature_names = X_selected.columns.tolist()
        
        return X_selected
    
    def train_models(self, X, y):
        """Train multiple models with cross-validation"""
        
        print("\n=== MODEL TRAINING ===")
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        # Cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            
            # Fit the model
            model.fit(X, y)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.models = results
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name} (CV ROC-AUC: {results[best_model_name]['cv_mean']:.4f})")
        
        return results
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation"""
        
        print(f"\n=== {model_name.upper()} EVALUATION ===")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            self.plot_feature_importance(model, X_test.columns)
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def plot_feature_importance(self, model, feature_names, top_n=15):
        """Plot feature importance"""
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            print(f"\nTop {top_n} Important Features:")
            for _, row in importance_df.iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
    
    def predict_churn(self, customer_data):
        """Production prediction pipeline"""
        
        if self.best_model is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        # Apply same preprocessing
        if self.preprocessor is not None:
            customer_data_processed = self.preprocessor.transform(customer_data)
        else:
            customer_data_processed = customer_data[self.feature_names]
        
        # Make predictions
        predictions = self.best_model.predict_proba(customer_data_processed)[:, 1]
        
        return predictions
    
    def model_monitoring_report(self, X_train, X_test):
        """Generate model monitoring report"""
        
        print("\n=== MODEL MONITORING REPORT ===")
        
        # Feature drift detection (simple version)
        print("Feature Drift Analysis:")
        
        for feature in X_train.select_dtypes(include=[np.number]).columns[:10]:  # Check top 10 features
            train_mean = X_train[feature].mean()
            test_mean = X_test[feature].mean()
            
            drift_percentage = abs((test_mean - train_mean) / train_mean) * 100
            
            if drift_percentage > 10:  # 10% threshold
                print(f"⚠️  {feature}: {drift_percentage:.1f}% drift detected")
            else:
                print(f"✓ {feature}: {drift_percentage:.1f}% drift (OK)")

def run_complete_ml_pipeline():
    """Run the complete ML pipeline"""
    
    print("=== CUSTOMER CHURN PREDICTION PIPELINE ===\n")
    
    # Load customer features (from Exercise 1)
    try:
        customer_features = pd.read_csv('customer_features.csv', index_col=0)
        print(f"Loaded customer features: {customer_features.shape}")
    except FileNotFoundError:
        print("Customer features not found. Please run Exercise 1 first.")
        return
    
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline()
    
    # Create churn target
    customer_features = pipeline.create_churn_target(customer_features)
    
    # Engineer ML features
    customer_features = pipeline.engineer_ml_features(customer_features)
    
    # Prepare features
    X, y, numeric_features, categorical_features = pipeline.prepare_features(customer_features)
    
    # Handle categorical features manually (simple approach)
    if categorical_features:
        le = LabelEncoder()
        for col in categorical_features:
            if col in X.columns:
                X[col] = le.fit_transform(X[col].astype(str))
        numeric_features.extend(categorical_features)
    
    # Create preprocessor
    preprocessor = pipeline.create_preprocessor(numeric_features, [])
    
    # Fit preprocessor and transform features
    X_processed = preprocessor.fit_transform(X)
    X_processed = pd.DataFrame(X_processed, columns=numeric_features, index=X.index)
    
    # Feature selection
    X_selected = pipeline.select_features(X_processed, y, method='correlation', k=25)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Train models
    model_results = pipeline.train_models(X_train, y_train)
    
    # Evaluate best model
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['cv_mean'])
    best_model = model_results[best_model_name]['model']
    
    evaluation_results = pipeline.evaluate_model(best_model, X_test, y_test, best_model_name)
    
    # Model monitoring
    pipeline.model_monitoring_report(X_train, X_test)
    
    # Business insights
    print("\n=== BUSINESS INSIGHTS ===")
    print(f"Model Performance Summary:")
    print(f"- Best Model: {best_model_name}")
    print(f"- ROC-AUC: {evaluation_results['roc_auc']:.4f}")
    print(f"- This model can identify {evaluation_results['roc_auc']:.1%} of churning customers correctly")
    
    # Save trained pipeline
    import pickle
    with open('churn_model_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"\nPipeline saved to 'churn_model_pipeline.pkl'")
    
    return pipeline, evaluation_results

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline, results = run_complete_ml_pipeline()