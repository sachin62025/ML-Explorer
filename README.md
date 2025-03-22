# ML Explorer App
application Link : https://ml-explorer-ds.streamlit.app/
An interactive machine learning application for exploring data, training models, and making predictions. This application is designed to help users understand machine learning concepts through an intuitive interface.

## Features

### Built-in Datasets
- Iris Dataset (Classification)
- Titanic Dataset (Classification)
- California Housing Dataset (Regression)
- Custom Dataset Upload

### Data Exploration
- Data preview and statistics
- Correlation analysis
- Feature visualizations
- Data distribution analysis

### Data Preprocessing
- Missing value handling (Mean imputation, Median imputation, Drop rows)
- Categorical encoding (One-Hot Encoding, Label Encoding)
- Feature scaling (StandardScaler, MinMaxScaler)
- Dimensionality reduction (PCA, t-SNE, LDA)

### Model Training
- Multiple algorithms:
  - Classification: Logistic Regression, Decision Trees, Random Forest, SVM, KNN, MLP, Gradient Boosting
  - Regression: Linear Regression, Ridge, Lasso, Decision Trees, Random Forest, SVR, KNN, MLP, Gradient Boosting
- Training mode options:
  - Basic training
  - Cross-validation
  - Hyperparameter tuning

### Model Evaluation
- Classification metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC
- Regression metrics: MAE, MSE, RMSE, R²
- Feature importance visualization
- Model comparison

### Prediction
- Make predictions on new data
- Visualize prediction results

## Getting Started

1. Install dependencies:
   ```
   pip install -r project_requirements.txt
   ```

2. Run the application:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py` - Main Streamlit application with UI components
- `data_utils.py` - Data loading, preprocessing, and dimensionality reduction functions
- `ml_utils.py` - Machine learning model training, evaluation, and prediction utilities
- `visualization.py` - Data and model visualization components
- `.streamlit/config.toml` - Streamlit configuration settings
