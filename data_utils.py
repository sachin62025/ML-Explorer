import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris, fetch_california_housing, fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def load_dataset(dataset_name):
    """
    Load a selected dataset and return dataframe, feature columns, target column, and description.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load
        
    Returns:
    --------
    df : pandas.DataFrame
        The dataset as a dataframe
    feature_columns : list
        List of feature column names
    target_column : str
        Name of the target column
    description : str
        Description of the dataset
    """
    if dataset_name == "Iris":
        # Load Iris dataset
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        feature_columns = iris.feature_names
        target_column = 'target'
        
        # Map target classes to names
        target_names = {i: name for i, name in enumerate(iris.target_names)}
        df['target_name'] = df['target'].map(target_names)
        
        description = """
        ### Iris Dataset
        
        The Iris dataset is a classic dataset in machine learning and statistics. It contains measurements 
        for 150 iris flowers from three different species: Setosa, Versicolor, and Virginica.
        
        **Features**:
        - Sepal length (cm)
        - Sepal width (cm)
        - Petal length (cm)
        - Petal width (cm)
        
        **Target**: Species of Iris (Setosa, Versicolor, Virginica)
        
        This is a classification problem where the goal is to predict the species of iris flowers 
        based on their measurements.
        """
        
    elif dataset_name == "Titanic":
        # Load Titanic dataset
        titanic = fetch_openml(name='titanic', version=1, as_frame=True)
        df = titanic.data
        df['survived'] = titanic.target
        
        # Handle missing values for demonstration purposes
        df = df.drop(['boat', 'body', 'home.dest', 'name', 'cabin', 'ticket'], axis=1)
        
        feature_columns = df.columns.tolist()
        feature_columns.remove('survived')
        target_column = 'survived'
        
        description = """
        ### Titanic Dataset
        
        The Titanic dataset contains information about passengers aboard the RMS Titanic, which sank after 
        colliding with an iceberg on April 15, 1912.
        
        **Features**:
        - pclass: Passenger class (1st, 2nd, or 3rd)
        - sex: Gender of the passenger
        - age: Age of the passenger
        - sibsp: Number of siblings/spouses aboard
        - parch: Number of parents/children aboard
        - fare: Passenger fare
        - embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
        
        **Target**: Survived (0 = No, 1 = Yes)
        
        This is a binary classification problem where the goal is to predict whether a passenger survived 
        the Titanic disaster based on their characteristics.
        """
        
    elif dataset_name == "California Housing":
        # Load California Housing dataset
        california = fetch_california_housing()
        df = pd.DataFrame(california.data, columns=california.feature_names)
        df['MedHouseVal'] = california.target
        
        feature_columns = california.feature_names
        target_column = 'MedHouseVal'
        
        description = """
        ### California Housing Dataset
        
        The California Housing dataset contains information about housing in California from the 1990 census.
        
        **Features**:
        - MedInc: Median income in block group
        - HouseAge: Median house age in block group
        - AveRooms: Average number of rooms per household
        - AveBedrms: Average number of bedrooms per household
        - Population: Block group population
        - AveOccup: Average number of household members
        - Latitude: Block group latitude
        - Longitude: Block group longitude
        
        **Target**: MedHouseVal (Median house value in $100,000s)
        
        This is a regression problem where the goal is to predict the median house value based on 
        various demographic features about the neighborhoods.
        """
    
    return df, feature_columns, target_column, description

def apply_dimensionality_reduction(X_train, X_test, y_train, method='PCA', n_components=2):
    """
    Apply dimensionality reduction to the dataset.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training target (used for supervised methods like LDA)
    method : str, default='PCA'
        Dimensionality reduction method to use ('PCA', 'TSNE', or 'LDA')
    n_components : int, default=2
        Number of components to reduce to
        
    Returns:
    --------
    X_train_reduced : numpy.ndarray
        Reduced training features
    X_test_reduced : numpy.ndarray
        Reduced testing features
    reducer : object
        The fitted dimensionality reduction model
    """
    # Handle n_components validation
    n_components = min(n_components, X_train.shape[1])
    
    if method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=42)
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)
    elif method == 'TSNE':
        reducer = TSNE(n_components=n_components, random_state=42)
        X_train_reduced = reducer.fit_transform(X_train)
        
        # t-SNE doesn't have a separate transform method, so we need to fit it again
        # This is a limitation of t-SNE
        # For visualization purposes only, not to be used for actual model training
        X_test_reduced = TSNE(n_components=n_components, random_state=42).fit_transform(X_test)
    elif method == 'LDA':
        # LDA requires labels and can only reduce to n_classes - 1 components
        n_classes = len(np.unique(y_train))
        n_components = min(n_components, n_classes - 1)
        
        if n_components <= 0:
            # Fallback to PCA if n_components is invalid (possible with binary classification)
            return apply_dimensionality_reduction(X_train, X_test, y_train, method='PCA', n_components=2)
        
        reducer = LinearDiscriminantAnalysis(n_components=n_components)
        X_train_reduced = reducer.fit_transform(X_train, y_train)
        X_test_reduced = reducer.transform(X_test)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    return X_train_reduced, X_test_reduced, reducer

def preprocess_data(df, feature_columns, target_column, test_size=0.2, preprocessing_steps=None):
    """
    Preprocess the data according to the specified preprocessing steps.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
    feature_columns : list
        List of feature column names
    target_column : str
        Name of the target column
    test_size : float, default=0.2
        Proportion of the dataset to be used for testing
    preprocessing_steps : dict, default=None
        Dictionary specifying preprocessing steps
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature dataframe
    y : pandas.Series
        Target series
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Testing target
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Extract features and target
    X = df_copy[feature_columns]
    y = df_copy[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # If no preprocessing steps specified, return the raw split data
    if preprocessing_steps is None or not preprocessing_steps:
        return X, y, X_train, X_test, y_train, y_test
    
    # Handle missing values
    if 'missing_values' in preprocessing_steps:
        strategy = 'mean' if preprocessing_steps['missing_values'] == 'Mean imputation' else 'median'
        
        if preprocessing_steps['missing_values'] == 'Drop rows':
            # Drop rows with missing values
            X_train = X_train.dropna()
            X_test = X_test.dropna()
            # Get the indexes of rows that were kept
            train_idx = X_train.index
            test_idx = X_test.index
            # Update y_train and y_test to match
            y_train = y_train.loc[train_idx]
            y_test = y_test.loc[test_idx]
        else:
            # Apply imputation
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_features) > 0:
                imputer = SimpleImputer(strategy=strategy)
                X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features])
                X_test[numeric_features] = imputer.transform(X_test[numeric_features])
    
    # Handle categorical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    if 'categorical_encoding' in preprocessing_steps and len(categorical_features) > 0:
        if preprocessing_steps['categorical_encoding'] == 'One-Hot Encoding':
            # Apply one-hot encoding
            X_train = pd.get_dummies(X_train, columns=categorical_features)
            X_test = pd.get_dummies(X_test, columns=categorical_features)
            
            # Ensure both train and test have the same columns
            missing_cols = set(X_train.columns) - set(X_test.columns)
            for col in missing_cols:
                X_test[col] = 0
            X_test = X_test[X_train.columns]
            
        elif preprocessing_steps['categorical_encoding'] == 'Label Encoding':
            # Apply label encoding
            for col in categorical_features:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
    
    # Apply feature scaling
    if 'scaling' in preprocessing_steps and preprocessing_steps['scaling'] != 'None':
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_features) > 0:
            if preprocessing_steps['scaling'] == 'StandardScaler':
                scaler = StandardScaler()
            else:  # MinMaxScaler
                scaler = MinMaxScaler()
                
            X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
            X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    return X, y, X_train, X_test, y_train, y_test
