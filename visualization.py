import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def visualize_data(df, columns):
    """
    Create a pairplot for the selected columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    columns : list
        List of columns to include in the pairplot
    """
    # Limit to selected columns
    subset_df = df[columns]
    
    # Create a pairplot for selected columns
    fig = px.scatter_matrix(
        subset_df,
        dimensions=columns,
        title="Feature Relationships (Scatter Matrix)",
        opacity=0.7
    )
    
    # Update layout for better visualization
    fig.update_layout(
        width=800,
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df):
    """
    Plot a correlation matrix for numeric columns in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if len(numeric_df.columns) < 2:
        st.write("Not enough numeric columns for correlation analysis.")
        return
    
    # Calculate correlation
    corr = numeric_df.corr()
    
    # Create correlation heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix",
        zmin=-1, zmax=1
    )
    
    st.plotly_chart(fig, use_container_width=True)

def visualize_results(metrics_df, problem_type):
    """
    Visualize model comparison metrics.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        Dataframe containing model metrics
    problem_type : str
        'classification' or 'regression'
    """
    if problem_type == 'classification':
        # Compare accuracy across models
        fig = px.bar(
            metrics_df, 
            y=metrics_df.index, 
            x='accuracy',
            orientation='h',
            title="Model Accuracy Comparison",
            labels={'accuracy': 'Accuracy', 'index': 'Model'},
            color='accuracy',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Compare precision, recall, F1 if available
        if 'precision' in metrics_df.columns and 'recall' in metrics_df.columns and 'f1_score' in metrics_df.columns:
            # Melt the dataframe for grouped bar chart
            melted_df = pd.melt(
                metrics_df.reset_index(),
                id_vars='index',
                value_vars=['precision', 'recall', 'f1_score'],
                var_name='Metric',
                value_name='Value'
            )
            
            fig = px.bar(
                melted_df,
                x='index',
                y='Value',
                color='Metric',
                barmode='group',
                title="Precision, Recall, and F1 Score Comparison",
                labels={'index': 'Model', 'Value': 'Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # ROC AUC comparison if available
        if 'roc_auc' in metrics_df.columns:
            if not metrics_df['roc_auc'].isna().all():
                fig = px.bar(
                    metrics_df,
                    y=metrics_df.index,
                    x='roc_auc',
                    orientation='h',
                    title="ROC AUC Comparison",
                    labels={'roc_auc': 'ROC AUC', 'index': 'Model'},
                    color='roc_auc',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # Regression
        # Compare MAE, MSE, RMSE across models
        melted_df = pd.melt(
            metrics_df.reset_index(),
            id_vars='index',
            value_vars=['mae', 'mse', 'rmse'],
            var_name='Metric',
            value_name='Value'
        )
        
        fig = px.bar(
            melted_df,
            x='index',
            y='Value',
            color='Metric',
            barmode='group',
            title="Error Metrics Comparison",
            labels={'index': 'Model', 'Value': 'Error'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # R² comparison
        fig = px.bar(
            metrics_df,
            y=metrics_df.index,
            x='r2',
            orientation='h',
            title="R² Score Comparison",
            labels={'r2': 'R² Score', 'index': 'Model'},
            color='r2',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn estimator
        The trained model (DecisionTree or RandomForest)
    feature_names : list
        List of feature names
    """
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Plot horizontal bar chart
        fig = px.bar(
            df,
            y='Feature',
            x='Importance',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("This model does not support feature importance visualization.")

def plot_dimensionality_reduction(X_reduced, y, method, feature_names=None, components_explained_variance=None):
    """
    Plot the results of dimensionality reduction.
    
    Parameters:
    -----------
    X_reduced : numpy.ndarray
        The reduced feature array
    y : pandas.Series or numpy.ndarray
        The target values
    method : str
        The dimensionality reduction method used ('PCA', 'TSNE', or 'LDA')
    feature_names : list, optional
        The names of the original features (used for PCA loadings plot)
    components_explained_variance : list, optional
        The explained variance ratios for PCA components
    """
    # Convert y to numpy array if it's a pandas Series
    if isinstance(y, pd.Series):
        y = y.values
    
    # Determine the number of dimensions in the reduced data
    n_dimensions = X_reduced.shape[1]
    
    if n_dimensions < 1:
        st.error("Dimensionality reduction resulted in 0 components.")
        return
    
    if n_dimensions == 1:
        # For 1D reduction, create a 1D scatter plot
        df = pd.DataFrame({
            'Component 1': X_reduced.flatten(),
            'Target': y
        })
        
        fig = px.scatter(
            df, 
            x='Component 1',
            y=[0] * len(df),  # All points at y=0
            color='Target',
            title=f"{method} 1D Projection",
            opacity=0.7
        )
        
        # Hide y-axis
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif n_dimensions == 2:
        # For 2D reduction, create a 2D scatter plot
        df = pd.DataFrame({
            'Component 1': X_reduced[:, 0],
            'Component 2': X_reduced[:, 1],
            'Target': y
        })
        
        fig = px.scatter(
            df, 
            x='Component 1',
            y='Component 2',
            color='Target',
            title=f"{method} 2D Projection",
            opacity=0.7
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif n_dimensions == 3:
        # For 3D reduction, create a 3D scatter plot
        df = pd.DataFrame({
            'Component 1': X_reduced[:, 0],
            'Component 2': X_reduced[:, 1],
            'Component 3': X_reduced[:, 2],
            'Target': y
        })
        
        fig = px.scatter_3d(
            df, 
            x='Component 1',
            y='Component 2',
            z='Component 3',
            color='Target',
            title=f"{method} 3D Projection",
            opacity=0.7
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # For higher dimensions, show multiple 2D projections of the first 4 components
        st.write(f"{method} projections (showing first 4 components):")
        
        # Create a grid of scatter plots for pairs of components
        components = min(4, n_dimensions)
        pairs = [(i, j) for i in range(components) for j in range(i+1, components)]
        
        for i, j in pairs:
            df = pd.DataFrame({
                f'Component {i+1}': X_reduced[:, i],
                f'Component {j+1}': X_reduced[:, j],
                'Target': y
            })
            
            fig = px.scatter(
                df, 
                x=f'Component {i+1}',
                y=f'Component {j+1}',
                color='Target',
                title=f"{method}: Component {i+1} vs Component {j+1}",
                opacity=0.7
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # If method is PCA and we have explained variance, show the explained variance plot
    if method == 'PCA' and components_explained_variance is not None:
        # Cumulative explained variance
        cumulative_variance = np.cumsum(components_explained_variance)
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(components_explained_variance))],
            'Explained Variance (%)': components_explained_variance * 100,
            'Cumulative Variance (%)': cumulative_variance * 100
        })
        
        # Plot individual explained variance
        fig1 = px.bar(
            df,
            x='Component',
            y='Explained Variance (%)',
            title='PCA: Explained Variance by Component',
            color='Explained Variance (%)',
            color_continuous_scale='viridis'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Plot cumulative explained variance
        fig2 = px.line(
            df,
            x='Component',
            y='Cumulative Variance (%)',
            title='PCA: Cumulative Explained Variance',
            markers=True
        )
        
        # Add threshold line at 95%
        fig2.add_hline(y=95, line_dash="dash", line_color="red", 
                      annotation_text="95% Variance", annotation_position="bottom right")
        
        st.plotly_chart(fig2, use_container_width=True)
        
    # If method is PCA and we have feature names, show feature loadings
    if method == 'PCA' and feature_names is not None and hasattr(feature_names, '__len__') and n_dimensions <= 2:
        st.write("PCA Feature Loadings:")
        
        # For 1D PCA
        if n_dimensions == 1 and hasattr(components_explained_variance, '__len__'):
            loadings = pd.DataFrame(
                {'PC1': components_explained_variance},
                index=feature_names
            )
            
            # Plot loadings
            fig = px.bar(
                loadings,
                orientation='h',
                title='PCA: Feature Loadings for PC1',
                color=loadings['PC1'],
                color_continuous_scale='RdBu_r'
            )
            
            st.plotly_chart(fig, use_container_width=True)
