import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# Set page configuration
st.set_page_config(
    page_title="Agricultural Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style for better visualizations
plt.style.use('default')
sns.set_style("whitegrid")
sns.set_palette("husl")
np.random.seed(42)

# Title and description
st.title("🌾 Agricultural Crop Yield Prediction")
st.markdown(
    "This application predicts crop yields based on environmental factors "
    "to help farmers with better planning and decision-making."
)


# Create sample agricultural dataset
@st.cache_data
def create_agricultural_data(n_samples=1000):
    """Create synthetic agricultural dataset for crop yield prediction"""
    np.random.seed(42)

    data = {
        'temperature': np.random.normal(25, 5, n_samples),
        'rainfall': np.random.gamma(2, 50, n_samples),
        'humidity': np.random.normal(65, 15, n_samples),
        'soil_ph': np.random.normal(6.5, 0.8, n_samples),
        'soil_nitrogen': np.random.normal(50, 15, n_samples),
        'soil_phosphorus': np.random.normal(40, 12, n_samples),
        'soil_potassium': np.random.normal(45, 10, n_samples),
        'sunlight_hours': np.random.normal(8, 2, n_samples),
    }

    df = pd.DataFrame(data)

    # Create realistic crop types
    crops = ['Wheat', 'Rice', 'Corn', 'Soybean', 'Barley']
    df['crop_type'] = np.random.choice(crops, n_samples)

    # Generate realistic yield based on features with some noise
    base_yield = (
        df['temperature'] * 0.3 +
        df['rainfall'] * 0.2 +
        df['humidity'] * 0.1 +
        df['soil_nitrogen'] * 0.4 +
        df['soil_phosphorus'] * 0.3 +
        df['soil_potassium'] * 0.3 +
        df['sunlight_hours'] * 0.5
    )

    # Adjust base yield by crop type
    crop_multipliers = {
        'Wheat': 1.2, 'Rice': 1.5, 'Corn': 1.8, 'Soybean': 1.1, 'Barley': 1.0
    }
    for crop, multiplier in crop_multipliers.items():
        mask = df['crop_type'] == crop
        base_yield[mask] = base_yield[mask] * multiplier

    # Add some noise and scale to realistic yield values
    noise = np.random.normal(0, 50, n_samples)
    df['yield'] = base_yield + noise
    df['yield'] = df['yield'].clip(lower=100)  # Ensure minimum yield

    return df


# Generate dataset immediately
crop_data = create_agricultural_data(1000)

# Initialize session state for results if not exists
if 'results' not in st.session_state:
    st.session_state.results = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a section",
    [
        "Project Overview", "Data Exploration",
        "Model Training", "Results & Insights"
    ]
)

# Display content based on selected mode
if app_mode == "Project Overview":
    st.header("Project Overview")

    st.markdown(
        "### Agricultural Context\n"
        "Agriculture faces challenges from climate volatility and resource "
        "scarcity. Accurate yield prediction enables optimal resource "
        "allocation, efficient supply chain management, and better financial "
        "planning.\n\n"
        "### Objectives\n"
        "- Integrate and analyze historical weather, soil, and yield data\n"
        "- Identify key environmental factors influencing crop yield\n"
        "- Develop robust regression models for accurate yield prediction\n"
        "- Provide actionable insights for farmers and planners\n\n"
        "### Dataset Information\n"
        "The dataset contains synthetic agricultural data with features:\n"
        "- **Environmental Factors**: Temperature, Rainfall, Humidity, "
        "Sunlight Hours\n"
        "- **Soil Properties**: pH, Nitrogen, Phosphorus, Potassium levels\n"
        "- **Crop Information**: Crop type (Wheat, Rice, Corn, Soybean, "
        "Barley)\n"
        "- **Target Variable**: Crop Yield"
    )

    # Show dataset sample
    st.subheader("Dataset Sample")
    st.dataframe(crop_data.head(10))

    st.info(f"Dataset Shape: {crop_data.shape}")

    # Show basic statistics
    st.subheader("Quick Dataset Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Samples", len(crop_data))
        st.metric("Number of Features", len(crop_data.columns) - 1)

    with col2:
        st.metric("Crop Types", len(crop_data['crop_type'].unique()))
        avg_yield = crop_data['yield'].mean()
        st.metric("Average Yield", f"{avg_yield:.1f}")

    with col3:
        st.metric("Missing Values", crop_data.isnull().sum().sum())
        data_size = crop_data.memory_usage(deep=True).sum() / 1024
        st.metric("Data Size", f"{data_size:.1f} KB")

elif app_mode == "Data Exploration":
    st.header("Data Exploration")

    # Dataset info in expandable sections
    with st.expander("Dataset Information", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Info")
            st.write(f"**Shape:** {crop_data.shape}")
            st.write(f"**Columns:** {list(crop_data.columns)}")
            st.write("**Data Types:**")
            st.write(crop_data.dtypes)

        with col2:
            st.subheader("Missing Values")
            missing_data = crop_data.isnull().sum()
            if missing_data.sum() == 0:
                st.success("No missing values in the dataset!")
            else:
                st.dataframe(missing_data[missing_data > 0])

    # Statistical summary
    with st.expander("Statistical Summary", expanded=True):
        st.subheader("Numerical Features Summary")
        st.dataframe(crop_data.describe())

    # Visualizations
    st.header("Data Visualizations")

    # Distribution of numerical features
    with st.expander("Feature Distributions", expanded=True):
        st.write("### Distribution of Numerical Features")
        numerical_features = [
            'temperature', 'rainfall', 'humidity', 'soil_ph',
            'soil_nitrogen', 'soil_phosphorus', 'soil_potassium',
            'sunlight_hours', 'yield'
        ]

        selected_feature = st.selectbox(
            "Select feature to view distribution", numerical_features
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(crop_data[selected_feature], kde=True, ax=ax)
        ax.set_title(f'Distribution of {selected_feature}')
        st.pyplot(fig)

    # Crop type distribution
    with st.expander("Crop Analysis", expanded=True):
        st.write("### Crop Type Analysis")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            crop_counts = crop_data['crop_type'].value_counts()
            ax.pie(
                crop_counts.values,
                labels=crop_counts.index,
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title('Distribution of Crop Types')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=crop_data, x='crop_type', y='yield', ax=ax)
            ax.set_title('Yield Distribution by Crop Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

    # Correlation heatmap
    with st.expander("Correlation Analysis", expanded=True):
        st.write("### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = crop_data.corr(numeric_only=True)
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax
        )
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)

    # Feature relationships with yield
    with st.expander("Feature-Yield Relationships", expanded=True):
        st.write("### Feature Relationships with Yield")
        features_to_plot = [
            'temperature', 'rainfall', 'soil_nitrogen',
            'soil_phosphorus', 'sunlight_hours'
        ]
        selected_relationship = st.selectbox(
            "Select feature to view relationship with yield", features_to_plot
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=crop_data,
            x=selected_relationship,
            y='yield',
            alpha=0.6,
            ax=ax
        )
        ax.set_title(f'{selected_relationship} vs Yield')
        st.pyplot(fig)

elif app_mode == "Model Training":
    st.header("Model Training")

    # Data preprocessing
    st.subheader("Data Preprocessing")

    # Prepare features and target
    X = crop_data.drop('yield', axis=1)
    y = crop_data['yield']

    # Encode categorical variables
    label_encoder = LabelEncoder()
    X_encoded = X.copy()
    X_encoded['crop_type'] = label_encoder.fit_transform(X['crop_type'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.success("✅ Data preprocessing completed successfully!")
    st.write(f"**Training set size:** {X_train.shape[0]} samples")
    st.write(f"**Test set size:** {X_test.shape[0]} samples")
    st.write(f"**Number of features:** {X_train.shape[1]}")

    # Model selection and training
    st.subheader("Machine Learning Models")

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Support Vector Machine': SVR(kernel='rbf')
    }

    # Display model descriptions
    st.write("### Available Models")
    model_descriptions = {
        'Linear Regression': 'Simple linear relationship',
        'Ridge Regression': 'Linear regression with L2 regularization',
        'Lasso Regression': 'Linear regression with L1 regularization',
        'Random Forest': 'Ensemble of decision trees',
        'Gradient Boosting': 'Sequential building of trees',
        'Decision Tree': 'Single tree-based model',
        'Support Vector Machine': 'Uses kernel trick'
    }

    for model_name, description in model_descriptions.items():
        st.write(f"**{model_name}:** {description}")

    # Train models button
    if st.button("🚀 Train All Models", type="primary"):
        st.write("Training models... This may take a few moments.")

        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (name, model) in enumerate(models.items()):
            status_text.text(f'Training {name}... ({i+1}/{len(models)})')

            # Use scaled data for linear models and SVM
            if name in [
                'Linear Regression', 'Ridge Regression',
                'Lasso Regression', 'Support Vector Machine'
            ]:
                X_tr = X_train_scaled
                X_te = X_test_scaled
            else:
                X_tr = X_train
                X_te = X_test

            # Train model
            model.fit(X_tr, y_train)

            # Make predictions
            y_pred = model.predict(X_te)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Store results
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }

            # Update progress
            progress_bar.progress((i + 1) / len(models))

        # Store results in session state
        st.session_state.results = results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.models = models
        st.session_state.models_trained = True

        status_text.text("✅ All models trained successfully!")
        st.balloons()

    # Show results if available
    if st.session_state.models_trained and st.session_state.results:
        st.subheader("📊 Model Performance Results")

        # Compare model performance
        performance_df = pd.DataFrame({
            'Model': list(st.session_state.results.keys()),
            'RMSE': [
                st.session_state.results[model]['rmse']
                for model in st.session_state.results
            ],
            'MAE': [
                st.session_state.results[model]['mae']
                for model in st.session_state.results
            ],
            'R²': [
                st.session_state.results[model]['r2']
                for model in st.session_state.results
            ]
        }).sort_values('RMSE')

        # Display performance table
        st.dataframe(
            performance_df.style
            .format({'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'R²': '{:.4f}'})
            .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
            .highlight_max(subset=['R²'], color='lightgreen')
        )

        # Visualize model performance
        st.write("### Performance Visualization")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=performance_df, x='RMSE', y='Model',
                palette='viridis', ax=ax
            )
            ax.set_title('Model Comparison - RMSE\n(Lower is Better)')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=performance_df, x='R²', y='Model',
                palette='viridis', ax=ax
            )
            ax.set_title('Model Comparison - R²\n(Higher is Better)')
            st.pyplot(fig)

        # Show best model
        best_model_row = performance_df.iloc[0]
        st.success(
            f"🎯 **Best Model:** {best_model_row['Model']} "
            f"(RMSE: {best_model_row['RMSE']:.2f}, "
            f"R²: {best_model_row['R²']:.4f})"
        )

elif app_mode == "Results & Insights":
    st.header("Results & Insights")

    if not st.session_state.models_trained or st.session_state.results is None:
        st.warning(
            "⚠️ Please train the models first in the 'Model Training' "
            "section to see results."
        )

        # Show sample of what will be available
        st.info(
            "After training models, this section will show:\n"
            "- Feature importance analysis\n"
            "- Prediction vs actual plots\n"
            "- Model performance by crop type\n"
            "- Key insights and recommendations"
        )
    else:
        # Get best model
        performance_df = pd.DataFrame({
            'Model': list(st.session_state.results.keys()),
            'RMSE': [
                st.session_state.results[model]['rmse']
                for model in st.session_state.results
            ],
            'MAE': [
                st.session_state.results[model]['mae']
                for model in st.session_state.results
            ],
            'R²': [
                st.session_state.results[model]['r2']
                for model in st.session_state.results
            ]
        }).sort_values('RMSE')

        best_model_name = performance_df.iloc[0]['Model']
        best_model = st.session_state.results[best_model_name]['model']
        best_predictions = (
            st.session_state.results[best_model_name]['predictions']
        )

        # Final results
        st.subheader("🏆 Final Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Best Model", best_model_name)
        with col2:
            rmse_val = st.session_state.results[best_model_name]['rmse']
            st.metric("RMSE", f"{rmse_val:.2f}")
        with col3:
            r2_val = st.session_state.results[best_model_name]['r2']
            st.metric("R² Score", f"{r2_val:.4f}")
        with col4:
            mae_val = st.session_state.results[best_model_name]['mae']
            st.metric("MAE", f"{mae_val:.2f}")

        # Feature importance
        st.subheader("🔍 Feature Importance Analysis")

        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': st.session_state.X_test.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=feature_importance,
                    x='importance',
                    y='feature',
                    palette='rocket',
                    ax=ax
                )
                ax.set_title(f'Feature Importance - {best_model_name}')
                ax.set_xlabel('Importance Score')
                st.pyplot(fig)

            with col2:
                st.write("### Top 5 Most Important Features:")
                for i, (_, row) in enumerate(
                    feature_importance.head().iterrows(), 1
                ):
                    st.write(
                        f"{i}. **{row['feature']}**: {row['importance']:.4f}"
                    )

                st.write("### Insights:")
                st.info(
                    "- Soil nutrients (N, P, K) are critical predictors\n"
                    "- Environmental factors like temperature matter\n"
                    "- Different crops respond differently to factors"
                )

        # Prediction visualization
        st.subheader("📈 Prediction Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(st.session_state.y_test, best_predictions, alpha=0.6)
            ax.plot(
                [st.session_state.y_test.min(), st.session_state.y_test.max()],
                [st.session_state.y_test.min(), st.session_state.y_test.max()],
                'r--',
                lw=2
            )
            ax.set_xlabel('Actual Yield')
            ax.set_ylabel('Predicted Yield')
            ax.set_title(f'Actual vs Predicted Yield\n({best_model_name})')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            residuals = st.session_state.y_test - best_predictions
            ax.scatter(best_predictions, residuals, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Yield')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            st.pyplot(fig)

        # Model performance by crop type
        st.subheader("🌱 Performance by Crop Type")

        crop_test = crop_data.loc[st.session_state.y_test.index, 'crop_type']
        performance_by_crop = {}

        for crop in crop_data['crop_type'].unique():
            crop_mask = crop_test == crop
            if crop_mask.sum() > 0:
                crop_y_true = st.session_state.y_test[crop_mask]
                crop_y_pred = best_predictions[crop_mask]

                performance_by_crop[crop] = {
                    'RMSE': np.sqrt(
                        mean_squared_error(crop_y_true, crop_y_pred)
                    ),
                    'R²': r2_score(crop_y_true, crop_y_pred),
                    'Samples': crop_mask.sum()
                }

        crop_performance_df = pd.DataFrame(performance_by_crop).T
        st.dataframe(crop_performance_df.style.format({
            'RMSE': '{:.2f}',
            'R²': '{:.4f}'
        }))

        # Insights and recommendations
        st.subheader("💡 Key Insights & Recommendations")

        st.markdown(
            "### Key Insights:\n\n"
            "**1. Critical Success Factors:**\n"
            "- Soil nutrients are important predictors\n"
            "- Environmental conditions impact crop yield\n"
            "- Different crops show varying sensitivity\n\n"
            "**2. Model Performance:**\n"
            "- Reliable predictions across crop types\n"
            "- Performance varies by crop\n"
            "- Good model fit with minimal bias\n\n"
            "### 🚜 Recommendations for Farmers:\n\n"
            "**Soil Management:**\n"
            "- Regular soil testing for nutrient levels\n"
            "- Balanced fertilization based on requirements\n"
            "- Crop-specific nutrient planning\n\n"
            "**Environmental Monitoring:**\n"
            "- Track temperature patterns\n"
            "- Optimize planting schedules\n"
            "- Monitor rainfall and irrigation\n\n"
            "**Crop Selection:**\n"
            "- Choose crops based on local conditions\n"
            "- Consider crop rotation\n"
            "- Implement precision agriculture\n\n"
            "### 📊 For Agricultural Planners:\n\n"
            "**Resource Allocation:**\n"
            "- Use predictive models for planning\n"
            "- Optimize supply chain logistics\n"
            "- Develop risk mitigation strategies\n\n"
            "**Policy Support:**\n"
            "- Support data-driven decision tools\n"
            "- Invest in agricultural technology\n"
            "- Develop early warning systems"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This agricultural crop yield prediction tool uses machine learning "
    "to help farmers and agricultural planners make data-driven decisions."
)
