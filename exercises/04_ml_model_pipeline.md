# Exercise 4: ML Model Pipeline Development

**Difficulty**: Advanced  
**Time**: 45-60 minutes  
**Skills**: ML pipeline, model evaluation, feature selection, cross-validation

## Business Context

You're building a customer churn prediction model for the e-commerce platform. Using the customer features you created earlier, you need to build, evaluate, and deploy a production-ready ML pipeline.

## The Challenge

Using the customer feature matrix from Exercise 1, build a complete ML pipeline that:

1. Handles feature selection and preprocessing
2. Trains multiple models with proper validation
3. Evaluates model performance and interpretability
4. Implements production-ready prediction pipeline

## Requirements

### Part 1: Feature Engineering & Selection (15 min)

1. **Feature preprocessing**:

   - Handle missing values appropriately
   - Scale/normalize features for different algorithms
   - Encode categorical variables
   - Handle feature interactions

2. **Feature selection**:

   - Remove highly correlated features
   - Select top K features using statistical tests
   - Implement recursive feature elimination
   - Discuss business interpretability

3. **Target variable creation**:
   - Define churn (e.g., no purchase in last 60 days)
   - Handle class imbalance
   - Create train/validation/test splits

### Part 2: Model Development (20 min)

1. **Multiple algorithms**:

   - Logistic Regression (baseline)
   - Random Forest (tree-based)
   - XGBoost (gradient boosting)
   - Optionally: Neural network

2. **Hyperparameter optimization**:

   - Cross-validation strategy
   - Grid search or random search
   - Early stopping for gradient boosting

3. **Model validation**:
   - Stratified K-fold cross-validation
   - Time-based validation (if applicable)
   - Proper metric selection for imbalanced data

### Part 3: Model Evaluation & Interpretation (15 min)

1. **Performance metrics**:

   - Precision, Recall, F1-score by class
   - ROC-AUC and PR-AUC
   - Business metrics (cost of false positives/negatives)
   - Confusion matrix analysis

2. **Model interpretability**:

   - Feature importance analysis
   - SHAP values for individual predictions
   - Partial dependence plots
   - Business insights from model

3. **Model comparison**:
   - Statistical significance testing
   - Bias-variance analysis
   - Performance vs interpretability trade-offs

### Part 4: Production Pipeline (10 min)

1. **Prediction pipeline**:

   - Feature transformation pipeline
   - Model inference function
   - Batch vs real-time predictions
   - Error handling and monitoring

2. **Model monitoring**:
   - Feature drift detection
   - Model performance monitoring
   - Retraining triggers
   - A/B testing framework

## Key ML Concepts to Demonstrate

- **Pipeline design**: Scikit-learn pipelines, feature unions
- **Cross-validation**: Proper validation strategies, avoiding data leakage
- **Class imbalance**: SMOTE, class weights, threshold tuning
- **Feature engineering**: Domain knowledge, automated feature selection
- **Model selection**: Bias-variance trade-off, ensemble methods
- **Interpretability**: Business stakeholder communication

## Discussion Points

- How would you handle new customer segments not seen in training?
- What's your approach to feature drift in production?
- How do you balance model complexity with interpretability?
- What metrics matter most for business decision-making?
- How would you design an A/B test for this model?

## Success Criteria

- End-to-end ML pipeline implementation
- Proper validation methodology without data leakage
- Thoughtful feature engineering with business rationale
- Model interpretation and business insights
- Production deployment considerations
- Clear communication of trade-offs and limitations

## Sample Code Structure

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import shap

# Feature engineering pipeline
def create_ml_features(customer_features):
    # Your feature engineering here
    pass

# Model training pipeline
def train_churn_model(X, y):
    # Your model training here
    pass

# Model evaluation
def evaluate_model(model, X_test, y_test):
    # Your evaluation here
    pass

# Production prediction pipeline
def predict_churn(customer_data, model, preprocessor):
    # Your prediction pipeline here
    pass
```
