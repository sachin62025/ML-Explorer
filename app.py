import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from ml_utils import train_model, evaluate_model, make_prediction
from data_utils import load_dataset, preprocess_data
from visualization import visualize_data, visualize_results, plot_correlation_matrix, plot_feature_importance

# Set page config
st.set_page_config(
    page_title="ML Explorer",
    page_icon="🧠",
    layout="wide"
)

def main():
    # Sidebar navigation
    st.sidebar.title("ML Explorer")
    st.sidebar.markdown("An interactive machine learning application for beginners")
    
    # Navigation
    pages = ["Home", "Data Exploration", "Model Training", "Make Predictions", "Upload Dataset", "Documentation"]
    
    # Use the session state value if it exists, otherwise default to Home
    default_index = 0
    if 'navigation_selection' in st.session_state:
        if st.session_state['navigation_selection'] in pages:
            default_index = pages.index(st.session_state['navigation_selection'])
            # Clear the selection after we've used it
            st.session_state['navigation_selection'] = None
    
    selection = st.sidebar.radio("Go to", pages, index=default_index)
    
    # Session state initialization
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'preprocessing_steps' not in st.session_state:
        st.session_state.preprocessing_steps = {}
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    if 'custom_dataset' not in st.session_state:
        st.session_state.custom_dataset = None
    if 'custom_dataset_name' not in st.session_state:
        st.session_state.custom_dataset_name = None
    if 'custom_feature_columns' not in st.session_state:
        st.session_state.custom_feature_columns = None
    if 'custom_target_column' not in st.session_state:
        st.session_state.custom_target_column = None
    
    # Handle dataset selection
    if selection == "Upload Dataset":
        render_upload_dataset_page()
        return
    
    # Dataset selection in sidebar
    dataset_options = ["Iris", "Titanic", "California Housing"]
    
    # Add custom dataset to options if available
    custom_dataset_available = False
    if (st.session_state.custom_dataset is not None and 
        st.session_state.custom_dataset_name is not None and 
        isinstance(st.session_state.custom_dataset_name, str)):
        dataset_options.append(st.session_state.custom_dataset_name)
        custom_dataset_available = True
    
    selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options)
    
    # Load and preprocess data
    if custom_dataset_available and selected_dataset == st.session_state.custom_dataset_name:
        df = st.session_state.custom_dataset
        feature_columns = st.session_state.custom_feature_columns
        target_column = st.session_state.custom_target_column
        
        # Ensure we have valid data
        if df is not None and feature_columns is not None and target_column is not None:
            dataset_description = f"""
            ### Custom Dataset: {selected_dataset}
            
            This is a custom dataset you've uploaded. It contains {df.shape[0]} rows and {df.shape[1]} columns.
            
            **Features**: {', '.join(feature_columns)}
            
            **Target**: {target_column}
            
            Explore the dataset using the Data Exploration tab to better understand its characteristics.
            """
        else:
            st.error("There was an issue with the custom dataset. Please try uploading it again.")
            # Fallback to a default dataset
            df, feature_columns, target_column, dataset_description = load_dataset("Iris")
    else:
        df, feature_columns, target_column, dataset_description = load_dataset(selected_dataset)
    
    # Render selected page
    if selection == "Home":
        render_home_page(dataset_description)
    elif selection == "Data Exploration":
        render_data_exploration(df, selected_dataset)
    elif selection == "Model Training":
        render_model_training(df, feature_columns, target_column, selected_dataset)
    elif selection == "Make Predictions":
        render_prediction_page(df, feature_columns, target_column, selected_dataset)
    elif selection == "Documentation":
        render_documentation()

def render_upload_dataset_page():
    st.title("Upload Your Own Dataset")
    
    st.markdown("""
    ## Custom Dataset Upload
    
    You can upload your own dataset to analyze and build models. The file should be a CSV file with the following characteristics:
    
    - Each row should represent a single observation
    - Each column should represent a feature or the target variable
    - The first row should contain column names
    - Missing values should be represented as empty cells or standard missing value indicators (NA, NaN, etc.)
    
    After uploading, you'll need to specify which column is your target variable for prediction.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load the dataset
            df = pd.read_csv(uploaded_file)
            
            # Display the first few rows
            st.markdown("### Dataset Preview")
            st.dataframe(df.head())
            
            # Get column information
            st.markdown("### Column Information")
            column_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(column_info)
            
            # Dataset name
            dataset_name = st.text_input("Give your dataset a name", value="My Dataset")
            
            # Target selection
            target_column = st.selectbox("Select the target column", df.columns.tolist())
            
            # Confirm upload
            if st.button("Confirm Dataset"):
                # Set feature columns (all columns except target)
                feature_columns = df.columns.tolist()
                feature_columns.remove(target_column)
                
                # Store dataset in session state
                st.session_state.custom_dataset = df
                st.session_state.custom_dataset_name = dataset_name
                st.session_state.custom_feature_columns = feature_columns
                st.session_state.custom_target_column = target_column
                
                st.success(f"Dataset '{dataset_name}' has been loaded successfully!")
                st.info("You can now select this dataset from the dropdown menu in the sidebar and proceed with your analysis.")
                
                # Add button to go to data exploration
                if st.button("Explore Dataset Now"):
                    # Use session state to navigate
                    st.session_state['navigation_selection'] = "Data Exploration"
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error loading the dataset: {str(e)}")
            st.markdown("""
            ### Common Upload Issues
            
            - Ensure your CSV file is properly formatted
            - Check for encoding issues (try UTF-8)
            - Large files may take longer to process
            - Make sure there are no special characters in column names
            """)
            
    st.markdown("""
    ## Working With Your Dataset
    
    After uploading your dataset, you'll be able to:
    
    1. **Explore** the data through visualizations and statistics
    2. **Preprocess** the data to handle missing values, encoding, etc.
    3. **Train** machine learning models on your data
    4. **Evaluate** different models to find the best performer
    5. **Make predictions** with your trained models
    
    You can switch between the built-in datasets and your custom dataset at any time using the dropdown menu in the sidebar.
    """)

def render_home_page(dataset_description):
    st.title("ML Explorer")
    st.markdown("""
    ## Welcome to ML Explorer!
    
    This interactive application helps beginners explore machine learning concepts through:
    
    - **Dataset exploration and visualization**
    - **Data preprocessing workflows**
    - **Model training with different algorithms**
    - **Model evaluation and comparison**
    - **Interactive prediction interface**
    
    ### Getting Started
    1. Select a dataset from the sidebar
    2. Explore the data in the "Data Exploration" section
    3. Train models in the "Model Training" section
    4. Make predictions with your trained models in the "Make Predictions" section
    5. Learn more about machine learning in the "Documentation" section
    
    ### About the Current Dataset
    """)
    st.markdown(dataset_description)
    
    st.markdown("""
    ### What is Machine Learning?
    
    Machine learning is a field of artificial intelligence that gives systems the ability to learn 
    from data and improve from experience without being explicitly programmed.
    
    ### Types of Machine Learning Algorithms
    
    - **Supervised Learning**: Training on labeled data (classification, regression)
    - **Unsupervised Learning**: Finding patterns in unlabeled data (clustering, dimensionality reduction)
    - **Reinforcement Learning**: Learning through interaction with an environment
    
    This application focuses on supervised learning techniques.
    """)

def render_data_exploration(df, selected_dataset):
    st.title(f"Exploring the {selected_dataset} Dataset")
    
    st.markdown("### Dataset Overview")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    st.write(df.describe())
    
    # Data preview
    st.markdown("### Data Preview")
    st.dataframe(df.head(10))
    
    # Missing values analysis
    st.markdown("### Missing Values Analysis")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    st.write(missing_df[missing_df['Missing Values'] > 0] if not missing_df[missing_df['Missing Values'] > 0].empty else "No missing values found!")
    
    # Data types
    st.markdown("### Data Types")
    dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
    st.write(dtypes_df)
    
    # Data Distribution
    st.markdown("### Data Distribution")
    
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_columns:
        selected_column = st.selectbox("Select column for histogram", numeric_columns)
        fig = px.histogram(df, x=selected_column, marginal="box", title=f"Distribution of {selected_column}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Matrix
    st.markdown("### Correlation Matrix")
    plot_correlation_matrix(df)
    
    # Pairplot for selected columns
    st.markdown("### Feature Relationships")
    if len(numeric_columns) > 1:
        if len(numeric_columns) > 5:
            default_columns = numeric_columns[:5]
        else:
            default_columns = numeric_columns
            
        selected_columns = st.multiselect(
            "Select columns for pairplot (max 5 recommended)",
            options=numeric_columns,
            default=default_columns
        )
        
        if selected_columns and len(selected_columns) >= 2:
            if len(selected_columns) <= 5:
                visualize_data(df, selected_columns)
            else:
                st.warning("Too many columns selected. This may cause performance issues. Consider selecting fewer columns.")
                if st.button("Generate Pairplot Anyway"):
                    visualize_data(df, selected_columns)
    else:
        st.write("Not enough numeric columns for pairplot visualization.")

def render_model_training(df, feature_columns, target_column, selected_dataset):
    st.title(f"Train Models on {selected_dataset} Dataset")
    
    # Data preprocessing options
    st.markdown("## Data Preprocessing")
    
    # Test size selection
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
    
    # Handle preprocessing options
    preprocessing_steps = {}
    
    with st.expander("Feature Scaling Options"):
        scaling_option = st.radio(
            "Apply feature scaling",
            ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            index=0
        )
        if scaling_option != "None":
            preprocessing_steps['scaling'] = scaling_option
            
    with st.expander("Dimensionality Reduction (Advanced)"):
        dim_reduction = st.radio(
            "Apply dimensionality reduction",
            ["None", "PCA", "t-SNE"],
            index=0
        )
        if dim_reduction != "None":
            preprocessing_steps['dim_reduction'] = dim_reduction
            components = st.slider("Number of components", min_value=2, max_value=min(10, len(feature_columns)), value=2)
            preprocessing_steps['dim_reduction_components'] = components
    
    # Handle categorical features
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_columns:
        with st.expander("Categorical Features Handling"):
            encoding_option = st.radio(
                "Encoding method",
                ["One-Hot Encoding", "Label Encoding"],
                index=0
            )
            preprocessing_steps['categorical_encoding'] = encoding_option
    
    # Handle missing values
    with st.expander("Missing Values Handling"):
        missing_strategy = st.radio(
            "Strategy for numerical features",
            ["Drop rows", "Mean imputation", "Median imputation"],
            index=0
        )
        preprocessing_steps['missing_values'] = missing_strategy
    
    # Feature selection
    with st.expander("Feature Selection"):
        all_features = st.checkbox("Use all available features", value=True)
        if not all_features:
            selected_features = st.multiselect("Select features", feature_columns, default=feature_columns)
            if selected_features:
                feature_columns = selected_features
    
    # Store preprocessing steps in session state
    st.session_state.preprocessing_steps = preprocessing_steps
    
    # Preprocess data based on options
    X, y, X_train, X_test, y_train, y_test = preprocess_data(
        df, 
        feature_columns, 
        target_column, 
        test_size=test_size, 
        preprocessing_steps=preprocessing_steps
    )
    
    # Store test data for prediction
    st.session_state.test_data = (X_test, y_test)
    
    # Model selection
    st.markdown("## Model Selection")
    
    # Determine if it's a classification or regression task
    problem_type = "classification" if len(np.unique(y)) < 10 else "regression"
    
    # Advanced ML settings
    training_mode = st.radio(
        "Training Mode",
        ["Standard", "Hyperparameter Tuning", "Cross-Validation"],
        help="Standard: Basic model training. Hyperparameter Tuning: Find optimal parameters. Cross-Validation: More robust evaluation."
    )
    
    if problem_type == "classification":
        algorithms = {
            "Logistic Regression": "LogisticRegression",
            "Decision Tree": "DecisionTree",
            "Random Forest": "RandomForest",
            "Gradient Boosting": "GradientBoosting",
            "Support Vector Machine": "SVM",
            "K-Nearest Neighbors": "KNN",
            "Neural Network (MLP)": "MLP"
        }
    else:
        algorithms = {
            "Linear Regression": "LinearRegression",
            "Ridge Regression": "Ridge",
            "Lasso Regression": "Lasso",
            "Decision Tree": "DecisionTree",
            "Random Forest": "RandomForest",
            "Gradient Boosting": "GradientBoosting",
            "Support Vector Machine": "SVM",
            "K-Nearest Neighbors": "KNN",
            "Neural Network (MLP)": "MLP"
        }
    
    selected_algorithms = st.multiselect(
        "Select algorithms to train",
        list(algorithms.keys()),
        default=list(algorithms.keys())[:2]
    )
    
    # Hyperparameter settings based on selected algorithms
    hyperparams = {}
    if training_mode == "Hyperparameter Tuning":
        st.markdown("### Hyperparameter Settings")
        st.markdown("Select hyperparameter ranges for selected algorithms")
        
        for algorithm_name in selected_algorithms:
            with st.expander(f"Hyperparameters for {algorithm_name}"):
                if "Logistic Regression" in algorithm_name or "Linear Regression" in algorithm_name:
                    hyperparams[algorithm_name] = {
                        'C': st.select_slider(f"{algorithm_name}: Regularization Strength", 
                                              options=[0.001, 0.01, 0.1, 1, 10, 100], value=1)
                    }
                elif "Ridge" in algorithm_name:
                    hyperparams[algorithm_name] = {
                        'alpha': st.select_slider(f"{algorithm_name}: Regularization Strength", 
                                                  options=[0.001, 0.01, 0.1, 1, 10, 100], value=1)
                    }
                elif "Lasso" in algorithm_name:
                    hyperparams[algorithm_name] = {
                        'alpha': st.select_slider(f"{algorithm_name}: Regularization Strength", 
                                                  options=[0.001, 0.01, 0.1, 1, 10, 100], value=1)
                    }
                elif "Decision Tree" in algorithm_name:
                    hyperparams[algorithm_name] = {
                        'max_depth': st.slider(f"{algorithm_name}: Max Depth", 2, 20, 5),
                        'min_samples_split': st.slider(f"{algorithm_name}: Min Samples Split", 2, 10, 2)
                    }
                elif "Random Forest" in algorithm_name:
                    hyperparams[algorithm_name] = {
                        'n_estimators': st.slider(f"{algorithm_name}: Number of Trees", 10, 200, 100),
                        'max_depth': st.slider(f"{algorithm_name}: Max Depth", 2, 20, 5)
                    }
                elif "Gradient Boosting" in algorithm_name:
                    hyperparams[algorithm_name] = {
                        'n_estimators': st.slider(f"{algorithm_name}: Number of Boosting Stages", 10, 200, 100),
                        'learning_rate': st.select_slider(f"{algorithm_name}: Learning Rate", 
                                                          options=[0.001, 0.01, 0.1, 0.2, 0.5], value=0.1)
                    }
                elif "SVM" in algorithm_name:
                    hyperparams[algorithm_name] = {
                        'C': st.select_slider(f"{algorithm_name}: Regularization Parameter", 
                                             options=[0.1, 1, 10, 100], value=1),
                        'kernel': st.selectbox(f"{algorithm_name}: Kernel Type", 
                                              ["linear", "poly", "rbf", "sigmoid"], index=2)
                    }
                elif "KNN" in algorithm_name:
                    hyperparams[algorithm_name] = {
                        'n_neighbors': st.slider(f"{algorithm_name}: Number of Neighbors", 1, 20, 5),
                        'weights': st.selectbox(f"{algorithm_name}: Weight Function", 
                                               ["uniform", "distance"], index=0)
                    }
                elif "MLP" in algorithm_name:
                    hyperparams[algorithm_name] = {
                        'hidden_layer_sizes': st.slider(f"{algorithm_name}: Neurons in Hidden Layer", 5, 100, 50),
                        'alpha': st.select_slider(f"{algorithm_name}: Regularization Strength", 
                                                 options=[0.0001, 0.001, 0.01, 0.1], value=0.001),
                        'learning_rate': st.selectbox(f"{algorithm_name}: Learning Rate Schedule", 
                                                     ["constant", "adaptive"], index=0)
                    }
    
    # Cross-validation settings
    cv_folds = 5
    if training_mode == "Cross-Validation":
        cv_folds = st.slider("Number of CV Folds", 2, 10, 5)
    
    # Button to start training
    if st.button("Train Selected Models"):
        if not selected_algorithms:
            st.error("Please select at least one algorithm to train.")
        else:
            with st.spinner("Training models... This may take some time, especially with hyperparameter tuning."):
                for algorithm_name in selected_algorithms:
                    algorithm = algorithms[algorithm_name]
                    
                    # Pass hyperparameters if tuning is enabled
                    algorithm_hyperparams = hyperparams.get(algorithm_name, {}) if training_mode == "Hyperparameter Tuning" else {}
                    
                    # Train the model with the appropriate mode
                    if training_mode == "Standard":
                        model = train_model(X_train, y_train, algorithm)
                    elif training_mode == "Hyperparameter Tuning":
                        model = train_model_with_hyperparams(X_train, y_train, algorithm, algorithm_hyperparams)
                    elif training_mode == "Cross-Validation":
                        model = train_model_with_cv(X_train, y_train, algorithm, cv_folds)
                    
                    # Store the trained model
                    st.session_state.trained_models[algorithm_name] = model
                    
                    # Evaluate the model
                    metrics = evaluate_model(model, X_test, y_test, problem_type)
                    st.session_state.metrics[algorithm_name] = metrics
                
                st.success("Models trained successfully!")
    
    # Display model comparison if models have been trained
    if st.session_state.metrics:
        st.markdown("## Model Comparison")
        
        # Create a dataframe to display metrics
        metrics_df = pd.DataFrame(st.session_state.metrics).T
        st.dataframe(metrics_df)
        
        # Visualize model comparison
        visualize_results(metrics_df, problem_type)
        
        # Feature importance for applicable models
        st.markdown("## Feature Importance")
        models_with_feature_importance = [
            model_name for model_name in st.session_state.trained_models.keys() 
            if 'Decision Tree' in model_name or 'Random Forest' in model_name
        ]
        
        if models_with_feature_importance:
            selected_model = st.selectbox(
                "Select model to view feature importance",
                models_with_feature_importance
            )
            
            model = st.session_state.trained_models[selected_model]
            plot_feature_importance(model, feature_columns)

def render_prediction_page(df, feature_columns, target_column, selected_dataset):
    st.title("Make Predictions")
    
    if not st.session_state.trained_models:
        st.warning("No trained models found. Please go to the Model Training section to train models first.")
        return
    
    st.markdown("## Predict with Trained Models")
    
    # Select a model for prediction
    model_name = st.selectbox("Select a model", list(st.session_state.trained_models.keys()))
    model = st.session_state.trained_models[model_name]
    
    # Show model information
    st.info(f"Selected model: **{model_name}**")
    
    # Add a download option for the trained model using pickle
    import pickle
    import io
    
    model_buffer = io.BytesIO()
    pickle.dump(model, model_buffer)
    model_buffer.seek(0)
    
    st.download_button(
        label="Download Trained Model",
        data=model_buffer,
        file_name=f"{selected_dataset}_{model_name}_model.pkl",
        mime="application/octet-stream",
        help="Download the trained model to use it in your own code or for later use."
    )
    
    # Prediction options
    prediction_option = st.radio(
        "Prediction method",
        ["Use test data", "Input custom values"]
    )
    
    if prediction_option == "Use test data":
        if st.session_state.test_data is None:
            st.error("No test data available. Please train models first.")
            return
            
        X_test, y_test = st.session_state.test_data
        
        # Option to select data sample size
        st.sidebar.subheader("Test Data Options")
        max_samples = min(100, len(X_test))
        sample_size = st.sidebar.slider("Maximum samples to show", 1, max_samples, min(5, max_samples))
        
        # Get a sample of test data to display
        sample_indices = st.multiselect(
            "Select test data samples",
            list(range(len(X_test))),
            default=list(range(sample_size))
        )
        
        if sample_indices:
            X_sample = X_test.iloc[sample_indices]
            y_sample = y_test.iloc[sample_indices]
            
            st.markdown("### Selected Test Data")
            st.dataframe(X_sample)
            
            if st.button("Make Predictions"):
                with st.spinner("Making predictions..."):
                    predictions = make_prediction(model, X_sample)
                    
                    # Create a dataframe to display results
                    results_df = pd.DataFrame({
                        'Actual': y_sample,
                        'Predicted': predictions
                    })
                    
                    # Calculate error metrics for the predictions
                    if len(predictions) > 1:
                        import sklearn.metrics as metrics
                        
                        # Determine if it's classification or regression
                        unique_targets = np.unique(y_sample)
                        is_classification = len(unique_targets) < 10
                        
                        if is_classification:
                            accuracy = metrics.accuracy_score(y_sample, predictions)
                            st.metric("Prediction Accuracy", f"{accuracy:.2%}")
                            
                            if len(unique_targets) == 2:  # Binary classification
                                results_df['Error Type'] = [
                                    'True Positive' if a == 1 and p == 1 else
                                    'True Negative' if a == 0 and p == 0 else
                                    'False Positive' if a == 0 and p == 1 else
                                    'False Negative'
                                    for a, p in zip(y_sample, predictions)
                                ]
                        else:  # Regression
                            mae = metrics.mean_absolute_error(y_sample, predictions)
                            rmse = np.sqrt(metrics.mean_squared_error(y_sample, predictions))
                            r2 = metrics.r2_score(y_sample, predictions)
                            
                            st.metric("Mean Absolute Error", f"{mae:.4f}")
                            st.metric("Root Mean Squared Error", f"{rmse:.4f}")
                            st.metric("R² Score", f"{r2:.4f}")
                            
                            # Add error column
                            results_df['Error'] = y_sample - predictions
                    
                    st.markdown("### Prediction Results")
                    st.dataframe(results_df)
                    
                    # Add download button for predictions
                    csv_buffer = io.BytesIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_buffer,
                        file_name=f"{selected_dataset}_{model_name}_predictions.csv",
                        mime="text/csv",
                        help="Download the prediction results for further analysis."
                    )
                    
                    # Visualization of actual vs predicted
                    if len(predictions) > 1:
                        st.markdown("### Actual vs Predicted Values")
                        
                        # For regression or multi-value classification
                        import plotly.express as px
                        
                        fig = px.scatter(
                            results_df, 
                            x='Actual', 
                            y='Predicted',
                            title="Actual vs Predicted Values",
                            labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
                            trendline='ols' if not is_classification else None
                        )
                        
                        # Add the ideal prediction line (y=x)
                        fig.add_shape(
                            type='line',
                            line=dict(dash='dash', color='gray'),
                            y0=min(y_sample.min(), predictions.min()),
                            y1=max(y_sample.max(), predictions.max()),
                            x0=min(y_sample.min(), predictions.min()),
                            x1=max(y_sample.max(), predictions.max())
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("### Enter Custom Values")
        
        # Create input fields for each feature
        custom_data = {}
        
        for feature in feature_columns:
            # Get feature data type
            feature_type = df[feature].dtype
            
            if np.issubdtype(feature_type, np.number):
                # For numeric features
                feature_min = float(df[feature].min())
                feature_max = float(df[feature].max())
                feature_mean = float(df[feature].mean())
                
                # Use slider for numeric inputs
                custom_data[feature] = st.slider(
                    f"{feature}",
                    min_value=feature_min,
                    max_value=feature_max,
                    value=feature_mean
                )
            else:
                # For categorical features
                options = df[feature].unique().tolist()
                custom_data[feature] = st.selectbox(f"{feature}", options)
        
        # Create a dataframe from the input
        input_df = pd.DataFrame([custom_data])
        
        st.markdown("### Input Data Preview")
        st.dataframe(input_df)
        
        if st.button("Make Prediction"):
            with st.spinner("Making prediction..."):
                # Apply the same preprocessing as during training
                # This is simplified - in a real app, you'd need to apply the same transformations
                
                # Make prediction
                prediction = make_prediction(model, input_df)
                
                st.markdown("### Prediction Result")
                st.success(f"Predicted value: {prediction[0]}")
                
                # Add option to export the prediction result
                result_df = pd.DataFrame({
                    **custom_data,
                    'Predicted': prediction[0]
                }, index=[0])
                
                csv_buffer = io.BytesIO()
                result_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="Download This Prediction",
                    data=csv_buffer,
                    file_name=f"{selected_dataset}_{model_name}_single_prediction.csv",
                    mime="text/csv",
                    help="Download the prediction result along with input features."
                )

def render_documentation():
    st.title("Machine Learning Documentation")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["ML Concepts", "Preprocessing", "Algorithms", "Evaluation", "Best Practices"])
    
    with tab1:
        st.header("Machine Learning Concepts")
        
        st.subheader("What is Machine Learning?")
        st.markdown("""
        Machine learning is a subfield of artificial intelligence that gives computers the ability to learn from data without being explicitly programmed. The key focus is on developing algorithms that can:
        
        - Learn patterns from data
        - Make decisions or predictions based on those patterns
        - Improve performance with more experience/data
        
        Machine learning is particularly useful when:
        1. Traditional programming approaches are too complex
        2. The rules are constantly changing
        3. The problem involves recognizing patterns in large datasets
        """)
        
        st.subheader("Types of Machine Learning")
        st.markdown("""
        #### Supervised Learning
        - Uses labeled data (input-output pairs)
        - Goal: Learn a mapping from inputs to outputs
        - Examples: Classification, regression
        
        #### Unsupervised Learning
        - Uses unlabeled data (inputs only)
        - Goal: Find patterns or structure in data
        - Examples: Clustering, dimensionality reduction, anomaly detection
        
        #### Reinforcement Learning
        - Agent learns through interaction with an environment
        - Goal: Maximize a reward signal
        - Examples: Game playing, robotics, resource management
        
        #### Semi-supervised Learning
        - Uses a small amount of labeled data and a large amount of unlabeled data
        - Goal: Improve learning accuracy
        - Examples: Speech recognition, image classification with limited labels
        """)
        
        st.subheader("The Machine Learning Workflow")
        st.markdown("""
        1. **Problem Definition**: Define the problem to solve
        2. **Data Collection**: Gather relevant data
        3. **Data Exploration**: Understand data characteristics
        4. **Data Preprocessing**: Clean and prepare data
        5. **Feature Engineering**: Create meaningful features
        6. **Model Selection**: Choose appropriate algorithms
        7. **Training**: Fit models to data
        8. **Evaluation**: Assess model performance
        9. **Hyperparameter Tuning**: Optimize model parameters
        10. **Deployment**: Use model in production
        11. **Monitoring**: Track performance over time
        """)
    
    with tab2:
        st.header("Data Preprocessing")
        
        st.subheader("Feature Scaling")
        st.markdown("""
        **Why Scale Features?**
        Many machine learning algorithms perform better or converge faster when features are on a similar scale. Feature scaling helps prevent features with larger values from dominating the learning process.
        
        **StandardScaler**
        - Standardizes features by removing the mean and scaling to unit variance
        - Results in a distribution with a mean of 0 and a standard deviation of 1
        - Formula: z = (x - μ) / σ
        - Best used when: Your data doesn't have outliers and you need to preserve the shape of the distribution
        
        **MinMaxScaler**
        - Transforms features by scaling each feature to a given range (usually [0, 1])
        - Formula: x' = (x - min(x)) / (max(x) - min(x))
        - Best used when: You need bounded values or your algorithm assumes data is within a specific range
        """)
        
        st.subheader("Handling Categorical Data")
        st.markdown("""
        **One-Hot Encoding**
        - Creates binary columns for each category
        - Each column represents one possible value of the original feature
        - Advantages: No ordinal relationship implied between categories
        - Disadvantages: Can create high-dimensional data ("curse of dimensionality")
        
        **Label Encoding**
        - Assigns a unique integer to each category
        - Advantages: Simple and maintains a single column
        - Disadvantages: Implies an ordinal relationship between categories that may not exist
        - Best used for: True ordinal data or tree-based models that don't assume relationships based on magnitude
        """)
        
        st.subheader("Missing Values")
        st.markdown("""
        **Dropping**
        - Remove rows or columns with missing values
        - Advantages: Simple and effective when data is missing completely at random
        - Disadvantages: Loss of information, can introduce bias if data is not missing at random
        
        **Imputation**
        - Replace missing values with estimated values
        - Common strategies:
          - **Mean/Median**: For numerical features
          - **Mode**: For categorical features
          - **K-nearest neighbors**: Use similar samples
          - **Regression models**: Predict missing values
        - Advantages: Preserves data size and potentially valuable information
        - Disadvantages: Can introduce bias if imputation method doesn't reflect true data generation process
        """)
        
        st.subheader("Feature Engineering")
        st.markdown("""
        Feature engineering is the process of creating new features from existing data to improve model performance.
        
        **Common Techniques**:
        - **Polynomial features**: Creating interaction terms (e.g., x₁ × x₂)
        - **Binning**: Converting continuous variables to categorical
        - **Log/Power transformations**: Handling skewed distributions
        - **Domain-specific features**: Using expert knowledge
        - **Feature extraction**: Creating new features from raw data (e.g., in text, images)
        """)
    
    with tab3:
        st.header("Machine Learning Algorithms")
        
        st.subheader("Linear Models")
        st.markdown("""
        **Linear Regression**
        - Predicts a continuous target variable as a linear combination of features
        - Formula: y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
        - Assumptions: Linearity, independence, homoscedasticity, normality
        - Strengths: Simple, interpretable
        - Weaknesses: Can't capture non-linear relationships
        
        **Logistic Regression**
        - Predicts probability of an event using logistic function
        - Used for classification problems
        - Formula: P(y=1) = 1 / (1 + e^-(b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ))
        - Strengths: Probabilistic interpretation, works well with linearly separable data
        - Weaknesses: Assumes linear decision boundary
        """)
        
        st.subheader("Tree-Based Models")
        st.markdown("""
        **Decision Trees**
        - Creates a flowchart-like structure where each node represents a feature, each branch a decision, and each leaf an outcome
        - Builds tree by selecting splits that maximize information gain
        - Strengths: Interpretable, handles non-linear relationships, no scaling required
        - Weaknesses: Prone to overfitting, unstable (small changes in data can lead to different trees)
        
        **Random Forest**
        - Ensemble of decision trees
        - Each tree is built on a random subset of data and features
        - Final prediction is average (regression) or majority vote (classification)
        - Strengths: Reduces overfitting, handles high dimensionality
        - Weaknesses: Less interpretable, computationally intensive
        """)
        
        st.subheader("Support Vector Machines")
        st.markdown("""
        **Support Vector Machines (SVM)**
        - Finds the hyperplane that best separates classes with maximum margin
        - Uses kernel functions to operate in higher dimensions
        - Strengths: Effective in high dimensions, works well with clear margin of separation
        - Weaknesses: Not suitable for large datasets, sensitive to kernel choice and parameters
        """)
        
        st.subheader("K-Nearest Neighbors")
        st.markdown("""
        **K-Nearest Neighbors (KNN)**
        - Classification: Assigns label based on majority vote of k nearest neighbors
        - Regression: Predicts average value of k nearest neighbors
        - Strengths: Simple, no training phase, adapts to new data easily
        - Weaknesses: Computationally expensive for large datasets, sensitive to irrelevant features
        """)
    
    with tab4:
        st.header("Model Evaluation")
        
        st.subheader("Classification Metrics")
        st.markdown("""
        **Confusion Matrix**
        - A table showing correct and incorrect predictions for each class
        - Components: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
        
        **Accuracy**
        - Proportion of correct predictions among the total number of cases
        - Formula: (TP + TN) / (TP + TN + FP + FN)
        - Limitations: Misleading for imbalanced datasets
        
        **Precision**
        - Ability of a model not to label a negative sample as positive
        - Formula: TP / (TP + FP)
        - When to use: When false positives are costly
        
        **Recall (Sensitivity)**
        - Ability of a model to find all positive samples
        - Formula: TP / (TP + FN)
        - When to use: When false negatives are costly
        
        **F1 Score**
        - Harmonic mean of precision and recall
        - Formula: 2 × (Precision × Recall) / (Precision + Recall)
        - When to use: When you need a balance between precision and recall
        
        **ROC Curve and AUC**
        - Receiver Operating Characteristic curve plots True Positive Rate vs. False Positive Rate
        - Area Under the Curve (AUC) measures the model's ability to discriminate between classes
        - AUC ranges from 0 to 1 (higher is better, with 0.5 being random)
        """)
        
        st.subheader("Regression Metrics")
        st.markdown("""
        **Mean Absolute Error (MAE)**
        - Average of absolute differences between predicted and actual values
        - Formula: (1/n) × Σ|y_true - y_pred|
        - Properties: Less sensitive to outliers
        
        **Mean Squared Error (MSE)**
        - Average of squared differences between predicted and actual values
        - Formula: (1/n) × Σ(y_true - y_pred)²
        - Properties: Penalizes larger errors more
        
        **Root Mean Squared Error (RMSE)**
        - Square root of MSE
        - Formula: √MSE
        - Properties: Same units as the target variable
        
        **R² Score (Coefficient of Determination)**
        - Proportion of variance in the dependent variable predictable from the independent variables
        - Ranges from 0 to 1 (higher is better)
        - Formula: 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_mean)²)
        - Properties: Scale-independent, allows comparison between different models
        """)
        
        st.subheader("Cross-Validation")
        st.markdown("""
        **K-Fold Cross-Validation**
        - Data is divided into k subsets (folds)
        - Model is trained on k-1 folds and tested on the remaining fold
        - Process is repeated k times, with each fold used once as the test set
        - Final performance is the average of all iterations
        
        **Benefits of Cross-Validation**
        - More reliable estimate of model performance
        - Reduces overfitting
        - Makes better use of limited data
        """)
    
    with tab5:
        st.header("Machine Learning Best Practices")
        
        st.subheader("Avoiding Overfitting")
        st.markdown("""
        **Overfitting** occurs when a model learns the training data too well, including its noise and outliers, leading to poor generalization on new data.
        
        **Strategies to Prevent Overfitting**:
        - Use more training data
        - Simplify the model (fewer parameters)
        - Apply regularization (L1, L2)
        - Early stopping
        - Dropout (for neural networks)
        - Ensemble methods
        - Cross-validation
        """)
        
        st.subheader("Feature Selection")
        st.markdown("""
        **Why Feature Selection?**
        - Reduces overfitting
        - Improves model performance
        - Reduces training time
        - Enhances interpretability
        
        **Feature Selection Methods**:
        - **Filter methods**: Statistical measures (correlation, chi-square)
        - **Wrapper methods**: Evaluate subsets of features (recursive feature elimination)
        - **Embedded methods**: Feature selection during model training (L1 regularization)
        """)
        
        st.subheader("Hyperparameter Tuning")
        st.markdown("""
        **Hyperparameters** are model settings that cannot be learned from the data and must be set before training.
        
        **Tuning Methods**:
        - **Grid Search**: Exhaustive search over specified parameter values
        - **Random Search**: Random combinations of parameters
        - **Bayesian Optimization**: Uses past evaluations to choose new parameters
        
        **Best Practices**:
        - Define a clear search space
        - Use cross-validation
        - Balance computation time vs. thoroughness
        - Monitor for overfitting
        """)
        
        st.subheader("Model Interpretability")
        st.markdown("""
        **Why Interpretability Matters**:
        - Builds trust in model predictions
        - Helps identify potential biases
        - Required for regulatory compliance in some fields
        - Provides insights for domain experts
        
        **Interpretability Techniques**:
        - Feature importance
        - Partial dependence plots
        - SHAP (SHapley Additive exPlanations) values
        - LIME (Local Interpretable Model-agnostic Explanations)
        - Rule extraction
        """)
        
        st.subheader("Ethical Considerations")
        st.markdown("""
        **Key Ethical Issues in Machine Learning**:
        - **Bias and fairness**: Ensuring models don't discriminate against protected groups
        - **Privacy**: Protecting sensitive information in training data
        - **Transparency**: Making model decisions understandable
        - **Accountability**: Determining responsibility for model outcomes
        - **Social impact**: Considering broader societal effects
        
        **Best Practices**:
        - Diverse and representative training data
        - Regular bias audits
        - Privacy-preserving techniques
        - Clear documentation
        - Human oversight
        """)
    
    # Add resources section
    st.markdown("""
    ## Resources for Further Learning
    
    - [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
    - [Kaggle Courses](https://www.kaggle.com/learn/overview)
    - [Machine Learning Mastery](https://machinelearningmastery.com/)
    - [Towards Data Science](https://towardsdatascience.com/)
    - [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
    
    ## About This Application
    
    This interactive machine learning application is designed to help beginners understand the machine learning workflow:
    
    1. **Data Exploration**: Understand the dataset through visualizations and statistics
    2. **Data Preprocessing**: Clean and transform the data for model training
    3. **Model Training**: Train different machine learning algorithms
    4. **Model Evaluation**: Compare models using appropriate metrics
    5. **Prediction**: Use trained models to make predictions
    
    All features are implemented using industry-standard Python libraries like scikit-learn, pandas, NumPy, Matplotlib, Seaborn, and Plotly.
    """)

if __name__ == "__main__":
    main()
