import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

def train_model(X_train, y_train, algorithm):
    """
    Train a model using the specified algorithm.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    algorithm : str
        The algorithm to use for training
        
    Returns:
    --------
    model : sklearn estimator
        The trained model
    """
    # Determine if it's a classification or regression problem
    unique_values = len(np.unique(y_train))
    is_classification = unique_values < 10  # Heuristic: if < 10 unique values, likely classification
    
    # Select the appropriate model based on the algorithm and problem type
    if algorithm == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm == "LinearRegression":
        model = LinearRegression()
    elif algorithm == "DecisionTree":
        if is_classification:
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = DecisionTreeRegressor(random_state=42)
    elif algorithm == "RandomForest":
        if is_classification:
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
    elif algorithm == "SVM":
        if is_classification:
            model = SVC(probability=True, random_state=42)
        else:
            model = SVR()
    elif algorithm == "KNN":
        if is_classification:
            model = KNeighborsClassifier()
        else:
            model = KNeighborsRegressor()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, problem_type=None):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : sklearn estimator
        The trained model
    X_test : pandas.DataFrame
        Testing features
    y_test : pandas.Series
        Testing target
    problem_type : str, optional
        'classification' or 'regression'. If None, will be inferred.
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Determine problem type if not provided
    if problem_type is None:
        unique_values = len(np.unique(y_test))
        problem_type = 'classification' if unique_values < 10 else 'regression'
    
    # Compute metrics based on problem type
    metrics = {}
    
    if problem_type == 'classification':
        # Calculate binary classification metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        # Only calculate precision, recall, f1 for binary classification
        if len(np.unique(y_test)) == 2:
            metrics['precision'] = precision_score(y_test, y_pred, average='binary')
            metrics['recall'] = recall_score(y_test, y_pred, average='binary')
            metrics['f1_score'] = f1_score(y_test, y_pred, average='binary')
            
            # Calculate ROC AUC if the model has predict_proba method
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        else:
            # Multi-class classification
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
            
            # Calculate ROC AUC for multi-class if the model has predict_proba method
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_test)
                    metrics['roc_auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
                except:
                    # Sometimes ROC AUC cannot be calculated for multi-class
                    metrics['roc_auc'] = np.nan
    
    else:  # Regression
        metrics['mae'] = mean_absolute_error(y_test, y_pred)
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_test, y_pred)
    
    return metrics

def make_prediction(model, X):
    """
    Make predictions using a trained model.
    
    Parameters:
    -----------
    model : sklearn estimator
        The trained model
    X : pandas.DataFrame
        Features to make predictions on
        
    Returns:
    --------
    predictions : numpy.ndarray
        The predicted values
    """
    return model.predict(X)

def get_model_instance(algorithm, is_classification=None):
    """
    Get a model instance based on algorithm name and problem type.
    
    Parameters:
    -----------
    algorithm : str
        The algorithm name
    is_classification : bool, optional
        Whether it's a classification problem. If None, will be inferred.
        
    Returns:
    --------
    model : sklearn estimator
        The model instance
    """
    if is_classification is None:
        # This function should be used with known problem type
        raise ValueError("Problem type (classification or regression) must be specified")
    
    if algorithm == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif algorithm == "LinearRegression":
        return LinearRegression()
    elif algorithm == "Ridge":
        return Ridge(random_state=42)
    elif algorithm == "Lasso":
        return Lasso(random_state=42)
    elif algorithm == "DecisionTree":
        if is_classification:
            return DecisionTreeClassifier(random_state=42)
        else:
            return DecisionTreeRegressor(random_state=42)
    elif algorithm == "RandomForest":
        if is_classification:
            return RandomForestClassifier(random_state=42)
        else:
            return RandomForestRegressor(random_state=42)
    elif algorithm == "GradientBoosting":
        if is_classification:
            return GradientBoostingClassifier(random_state=42)
        else:
            return GradientBoostingRegressor(random_state=42)
    elif algorithm == "SVM":
        if is_classification:
            return SVC(probability=True, random_state=42)
        else:
            return SVR()
    elif algorithm == "KNN":
        if is_classification:
            return KNeighborsClassifier()
        else:
            return KNeighborsRegressor()
    elif algorithm == "MLP":
        if is_classification:
            return MLPClassifier(max_iter=1000, random_state=42)
        else:
            return MLPRegressor(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def train_model_with_hyperparams(X_train, y_train, algorithm, hyperparams):
    """
    Train a model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    algorithm : str
        The algorithm to use for training
    hyperparams : dict
        Hyperparameters for tuning
        
    Returns:
    --------
    model : sklearn estimator
        The best trained model
    """
    # Determine if it's a classification or regression problem
    unique_values = len(np.unique(y_train))
    is_classification = unique_values < 10  # Heuristic: if < 10 unique values, likely classification
    
    # Convert hyperparams to grid format expected by GridSearchCV
    param_grid = {}
    for param, value in hyperparams.items():
        if isinstance(value, list):
            param_grid[param] = value
        else:
            param_grid[param] = [value]
    
    # Get the base model
    base_model = get_model_instance(algorithm, is_classification)
    
    # Use RandomizedSearchCV for efficiency when large parameter space
    if sum(len(values) for values in param_grid.values()) > 10:
        search = RandomizedSearchCV(
            base_model, 
            param_distributions=param_grid,
            n_iter=10,  # Number of parameter settings sampled
            cv=3,       # Number of cross-validation folds
            scoring='accuracy' if is_classification else 'neg_mean_squared_error',
            random_state=42
        )
    else:
        search = GridSearchCV(
            base_model, 
            param_grid=param_grid,
            cv=3,  # Number of cross-validation folds
            scoring='accuracy' if is_classification else 'neg_mean_squared_error'
        )
    
    # Fit the grid search
    search.fit(X_train, y_train)
    
    # Return the best model
    return search.best_estimator_

def train_model_with_cv(X_train, y_train, algorithm, cv_folds=5):
    """
    Train a model with cross-validation.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    algorithm : str
        The algorithm to use for training
    cv_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    model : sklearn estimator
        The trained model
    """
    # Determine if it's a classification or regression problem
    unique_values = len(np.unique(y_train))
    is_classification = unique_values < 10
    
    # Get the base model
    model = get_model_instance(algorithm, is_classification)
    
    # Perform cross-validation to estimate performance
    scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
    
    # Train final model on all training data
    model.fit(X_train, y_train)
    
    # Add CV results as attributes to the model for reference
    model.cv_scores_ = cv_scores
    model.cv_mean_score_ = np.mean(cv_scores)
    model.cv_std_score_ = np.std(cv_scores)
    
    return model
