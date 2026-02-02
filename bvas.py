import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from io import BytesIO

# Page config
st.set_page_config(page_title="Geospatial ML", layout="wide")

# Title
st.title("🌍 Petrol Station Sales Predictor")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar menu
menu = st.sidebar.selectbox(
    "Select Section",
    ["📁 Upload", "🔍 Explore", "🧹 Clean", "🗺️ Map", "🤖 Train Model", "📊 Predict"]
)

# ==================== UPLOAD ====================
if menu == "📁 Upload":
    st.header("Upload Data")
    
    # File uploader for multiple formats
    uploaded_file = st.file_uploader(
        "Choose file", 
        type=['csv', 'geojson', 'gpkg', 'xlsx'],
        help="Upload CSV, GeoJSON, GeoPackage, or Excel files"
    )
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
                
            elif file_ext == 'geojson':
                # Read GeoJSON
                gdf = gpd.read_file(uploaded_file)
                st.session_state.gdf = gdf
                
                # Extract attributes to DataFrame
                if 'geometry' in gdf.columns:
                    df = pd.DataFrame(gdf.drop(columns='geometry'))
                else:
                    df = pd.DataFrame(gdf)
                
                st.session_state.df = df
                st.success(f"✅ GeoJSON loaded: {len(df)} rows, {len(df.columns)} columns")
                st.info(f"📐 Contains {len(gdf)} spatial features")
                
            elif file_ext == 'gpkg':
                # Read GeoPackage
                gdf = gpd.read_file(uploaded_file)
                st.session_state.gdf = gdf
                
                if 'geometry' in gdf.columns:
                    df = pd.DataFrame(gdf.drop(columns='geometry'))
                else:
                    df = pd.DataFrame(gdf)
                
                st.session_state.df = df
                st.success(f"✅ GeoPackage loaded: {len(df)} rows, {len(df.columns)} columns")
                st.info(f"📐 Contains {len(gdf)} spatial features")
                
            elif file_ext == 'xlsx':
                df = pd.read_excel(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ Excel loaded: {len(df)} rows, {len(df.columns)} columns")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Show data preview if loaded
    if st.session_state.df is not None:
        st.subheader("📋 Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(st.session_state.df))
        with col2:
            st.metric("Columns", len(st.session_state.df.columns))
        with col3:
            null_count = st.session_state.df.isnull().sum().sum()
            st.metric("Missing Values", null_count)
        
        # Show first few rows
        st.dataframe(st.session_state.df.head())
        
        # Show column types
        st.subheader("📊 Data Types")
        dtype_df = pd.DataFrame(st.session_state.df.dtypes, columns=['Data Type'])
        st.dataframe(dtype_df)

# ==================== EXPLORE ====================
elif menu == "🔍 Explore" and st.session_state.df is not None:
    st.header("🔍 Explore Data")
    df = st.session_state.df
    
    # Check if sales_volume exists
    if 'sales_volume' in df.columns:
        st.success("✅ Target variable 'sales_volume' found!")
        # Show distribution of sales volume
        import plotly.express as px
        fig = px.histogram(df, x='sales_volume', title="Distribution of Sales Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📈 Statistics", "📊 Distributions", "🔗 Correlations"])
    
    with tab1:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
        
        # Missing values
        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
    
    with tab2:
        st.subheader("Feature Distributions")
        selected_col = st.selectbox("Select column to visualize", df.columns)
        
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            # Histogram
            import plotly.express as px
            fig = px.histogram(df, x=selected_col, 
                             title=f"Distribution of {selected_col}",
                             nbins=30)
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot
            fig2 = px.box(df, y=selected_col, 
                         title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            # Bar chart for categorical
            import plotly.express as px
            value_counts = df[selected_col].value_counts().head(20)
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"Top values in {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Correlation matrix
            import plotly.express as px
            corr_matrix = df[numeric_cols].corr()
            
            # Highlight correlation with sales_volume if exists
            if 'sales_volume' in corr_matrix.columns:
                # Sort by correlation with sales_volume
                sales_corr = corr_matrix['sales_volume'].sort_values(ascending=False)
                st.subheader("Correlation with Sales Volume")
                st.dataframe(sales_corr)
            
            fig = px.imshow(corr_matrix,
                          labels=dict(color="Correlation"),
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation")

# ==================== CLEAN ====================
elif menu == "🧹 Clean" and st.session_state.df is not None:
    st.header("🧹 Clean Data")
    df = st.session_state.df.copy()
    
    st.write(f"**Current data shape:** {df.shape}")
    
    # Cleaning options
    st.subheader("Cleaning Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_nulls = st.checkbox("Remove rows with null values", value=True)
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        fill_nulls = st.checkbox("Fill nulls with mean (numeric only)", value=False)
    
    with col2:
        drop_columns = st.multiselect("Drop columns (optional)", df.columns)
        keep_columns = st.multiselect("Keep only these columns (optional)", df.columns)
    
    # Convert empty strings to NaN
    convert_empty = st.checkbox("Convert empty strings to NaN", value=True)
    
    if st.button("🚀 Apply Cleaning", type="primary"):
        with st.spinner("Cleaning data..."):
            initial_shape = df.shape
            
            # Apply cleaning operations
            if convert_empty:
                df = df.replace(r'^\s*$', np.nan, regex=True)
                st.info("Converted empty strings to NaN")
            
            if drop_columns:
                df = df.drop(columns=drop_columns)
                st.info(f"Dropped {len(drop_columns)} columns")
            
            if keep_columns:
                df = df[keep_columns]
                st.info(f"Kept {len(keep_columns)} columns")
            
            if fill_nulls:
                # Fill numeric columns with mean
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].mean())
                st.info("Filled nulls with column means")
            
            if remove_nulls:
                initial_rows = len(df)
                df = df.dropna()
                removed = initial_rows - len(df)
                if removed > 0:
                    st.info(f"Removed {removed} rows with null values")
            
            if remove_duplicates:
                initial_rows = len(df)
                df = df.drop_duplicates()
                removed = initial_rows - len(df)
                if removed > 0:
                    st.info(f"Removed {removed} duplicate rows")
            
            # Update session state
            st.session_state.df = df
            
            # Show results
            st.success(f"✅ Cleaned data: {df.shape}")
            st.write(f"**Before:** {initial_shape}")
            st.write(f"**After:** {df.shape}")
            
            # Show sample
            st.subheader("Cleaned Data Preview")
            st.dataframe(df.head())

# ==================== MAP ====================
elif menu == "🗺️ Map" and st.session_state.gdf is not None:
    st.header("🗺️ Interactive Map")
    gdf = st.session_state.gdf
    
    try:
        import folium
        from streamlit_folium import st_folium
        
        # Add sales volume to color if exists
        color_by = None
        if 'sales_volume' in gdf.columns:
            color_by = 'sales_volume'
            st.info("Coloring stations by sales volume")
        
        # Get center of map
        center_lat = gdf.geometry.centroid.y.mean()
        center_lon = gdf.geometry.centroid.x.mean()
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add features to map
        for idx, row in gdf.iterrows():
            if hasattr(row.geometry, 'x'):  # Point geometry
                popup_content = f"""
                <b>Station:</b> {row.get('station_name_geo', f'Station {idx}')}<br>
                <b>Sales Volume:</b> {row.get('sales_volume', 'N/A')}<br>
                <b>State:</b> {row.get('State', 'N/A')}<br>
                <b>Roads within 1km:</b> {row.get('roads_within_1000m', 'N/A')}
                """
                
                # Color by sales volume if available
                color = 'blue'
                if color_by and pd.notna(row.get(color_by)):
                    # Simple color gradient based on sales volume
                    sales = row[color_by]
                    if sales > gdf[color_by].quantile(0.75):
                        color = 'green'  # High sales
                    elif sales > gdf[color_by].quantile(0.25):
                        color = 'blue'   # Medium sales
                    else:
                        color = 'red'    # Low sales
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=8,
                    popup=folium.Popup(popup_content, max_width=300),
                    color=color,
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
        
        # Display map
        st_folium(m, width=1200, height=600)
        
        # Add legend
        if color_by:
            st.markdown("""
            **Map Legend:**
            - 🟢 Green: High sales volume (top 25%)
            - 🔵 Blue: Medium sales volume (25%-75%)
            - 🔴 Red: Low sales volume (bottom 25%)
            """)
        
    except Exception as e:
        st.warning(f"Could not display map: {str(e)}")
        st.info("Make sure you have 'folium' installed: `pip install folium streamlit-folium`")

# ==================== TRAIN MODEL ====================
elif menu == "🤖 Train Model" and st.session_state.df is not None:
    st.header("🤖 Train Prediction Model")
    df = st.session_state.df
    
    # Check if we have the required features
    required_features = [
        'distance_to_nearest_road_m', 'distance_to_nearest_road_km',
        'roads_within_1000m', 'road_length_within_1000m', 'road_density_1000m',
        'roads_within_3000m', 'road_length_within_3000m', 'road_density_3000m',
        'roads_within_5000m', 'road_length_within_5000m', 'road_density_5000m',
        'lga_population'
    ]
    
    if 'sales_volume' not in df.columns:
        st.error("❌ 'sales_volume' column not found! This is required as the target variable.")
    else:
        # Check which features are available
        available_features = [f for f in required_features if f in df.columns]
        missing_features = [f for f in required_features if f not in df.columns]
        
        if available_features:
            st.success(f"✅ Found {len(available_features)}/{len(required_features)} required features")
            st.write("**Available features:**", available_features)
            
            if missing_features:
                st.warning(f"⚠️ Missing {len(missing_features)} features: {missing_features}")
            
            # Model selection
            model_type = st.selectbox(
                "Select Model Type",
                ["Random Forest", "Linear Regression", "XGBoost", "Decision Tree"]
            )
            
            # Training settings
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
            with col2:
                random_state = st.number_input("Random state", value=42, min_value=0, step=1)
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare data
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                        from sklearn.preprocessing import StandardScaler
                        
                        X = df[available_features]
                        y = df['sales_volume']
                        
                        # Handle missing values
                        X = X.fillna(X.mean())
                        y = y.fillna(y.mean())
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=int(random_state)
                        )
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train model
                        if model_type == "Random Forest":
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(n_estimators=100, random_state=int(random_state))
                        elif model_type == "Linear Regression":
                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                        # elif model_type == "XGBoost":
                        #     from xgboost import XGBRegressor
                        #     model = XGBRegressor(n_estimators=100, random_state=int(random_state))
                        # elif model_type == "Decision Tree":
                        #     from sklearn.tree import DecisionTreeRegressor
                        #     model = DecisionTreeRegressor(random_state=int(random_state))
                        
                        model.fit(X_train_scaled, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test_scaled)
                        
                        # Calculate metrics
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        # Store model in session state
                        st.session_state.model = {
                            'model': model,
                            'scaler': scaler,
                            'features': available_features,
                            'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae}
                        }
                        
                        # Show results
                        st.success("✅ Model trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.2f}")
                        with col3:
                            st.metric("MAE", f"{mae:.2f}")
                        
                        # Feature importance
                        st.subheader("🔑 Feature Importance")
                        
                        if hasattr(model, 'feature_importances_'):
                            importance = pd.DataFrame({
                                'Feature': available_features,
                                'Importance': model.feature_importances_ # pyright: ignore[reportAttributeAccessIssue]
                            }).sort_values('Importance', ascending=False)
                            
                            import plotly.express as px
                            fig = px.bar(importance, x='Feature', y='Importance',
                                       title='Feature Importance')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif hasattr(model, 'coef_'):
                            importance = pd.DataFrame({
                                'Feature': available_features,
                                'Coefficient': model.coef_, # pyright: ignore[reportAttributeAccessIssue]
                                'Absolute Impact': np.abs(model.coef_) # pyright: ignore[reportAttributeAccessIssue]
                            }).sort_values('Absolute Impact', ascending=False)
                            st.dataframe(importance)
                        
                        # Plot predictions vs actual
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=y_test, y=y_pred, 
                            mode='markers', 
                            name='Predictions',
                            marker=dict(size=8, opacity=0.6)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[y_test.min(), y_test.max()], 
                            y=[y_test.min(), y_test.max()],
                            mode='lines', 
                            name='Perfect Fit',
                            line=dict(dash='dash', color='red')
                        ))
                        fig.update_layout(
                            title="📈 Predictions vs Actual",
                            xaxis_title="Actual Sales Volume",
                            yaxis_title="Predicted Sales Volume",
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("✅ Model saved! Go to '📊 Predict' section to make predictions.")
                        
                    except Exception as e:
                        st.error(f"❌ Error training model: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.error(f"❌ None of the required features found in the data.")
            st.write("Required features:", required_features)

# ==================== PREDICT ====================
elif menu == "📊 Predict":
    st.header("📊 Predict Sales Volume")
    
    if st.session_state.model is None:
        st.warning("⚠️ No model trained yet!")
        st.info("Please go to '🤖 Train Model' section to train a model first.")
    else:
        model_info = st.session_state.model
        features = model_info['features']
        
        st.success(f"✅ Model loaded with {len(features)} features")
        st.write(f"Model R² Score: {model_info['metrics']['r2']:.4f}")
        
        # Create input form
        st.subheader("Enter Station Features")
        
        # Create columns for inputs
        col1, col2, col3 = st.columns(3)
        
        input_values = {}
        
        # Group features logically
        distance_features = [f for f in features if 'distance' in f]
        road_features = [f for f in features if 'road' in f]
        population_features = [f for f in features if 'population' in f]
        
        with col1:
            st.markdown("**📍 Distance Features**")
            for feature in distance_features:
                if feature in features:
                    if 'km' in feature:
                        input_values[feature] = st.number_input(
                            f"{feature} (km)",
                            min_value=0.0,
                            max_value=100.0,
                            value=1.0,
                            step=0.1,
                            help="Distance in kilometers"
                        )
                    else:
                        input_values[feature] = st.number_input(
                            f"{feature} (m)",
                            min_value=0,
                            max_value=100000,
                            value=1000,
                            step=100,
                            help="Distance in meters"
                        )
        
        with col2:
            st.markdown("**🛣️ Road Network Features**")
            for feature in road_features:
                if feature in features and feature not in distance_features:
                    if 'density' in feature:
                        input_values[feature] = st.number_input(
                            f"{feature}",
                            min_value=0.0,
                            max_value=10.0,
                            value=0.5,
                            step=0.1,
                            help="Road density (road length/area)"
                        )
                    elif 'roads_within' in feature:
                        input_values[feature] = st.number_input(
                            f"{feature}",
                            min_value=0,
                            max_value=50,
                            value=5,
                            step=1,
                            help="Number of roads within radius"
                        )
                    elif 'road_length' in feature:
                        input_values[feature] = st.number_input(
                            f"{feature} (m)",
                            min_value=0,
                            max_value=50000,
                            value=5000,
                            step=100,
                            help="Total road length in meters"
                        )
        
        with col3:
            st.markdown("**👥 Population Features**")
            for feature in population_features:
                if feature in features:
                    input_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=0,
                        max_value=10000000,
                        value=50000,
                        step=1000,
                        help="Local Government Area population"
                    )
            
            # Station name (optional)
            station_name = st.text_input("Station Name (optional)", "New Station")
        
        # Make prediction
        if st.button("🔮 Predict Sales Volume", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    # Prepare input data
                    input_df = pd.DataFrame([input_values])
                    
                    # Scale features
                    input_scaled = model_info['scaler'].transform(input_df)
                    
                    # Make prediction
                    prediction = model_info['model'].predict(input_scaled)[0]
                    
                    # Display result
                    st.success("✅ Prediction Complete!")
                    
                    # Create result card
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric(
                            "Predicted Sales Volume", 
                            f"{prediction:,.0f}",
                            help="Predicted sales in units"
                        )
                    
                    with col2:
                        st.markdown(f"""
                        **Station:** {station_name}
                        
                        **Key Factors:**
                        - Distance to nearest road: {input_values.get('distance_to_nearest_road_km', 'N/A')} km
                        - Roads within 1km: {input_values.get('roads_within_1000m', 'N/A')}
                        - Local population: {input_values.get('lga_population', 'N/A'):,}
                        """)
                    
                    # Feature importance for this prediction
                    st.subheader("📊 What's driving this prediction?")
                    
                    if hasattr(model_info['model'], 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': model_info['model'].feature_importances_
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        # Add current values
                        importance_df['Your Value'] = importance_df['Feature'].map(input_values)
                        
                        st.dataframe(importance_df)
                        
                        # Simple visualization
                        import plotly.express as px
                        fig = px.bar(importance_df, x='Feature', y='Importance',
                                   title='Top 10 Most Important Features',
                                   hover_data=['Your Value'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Save prediction to CSV
                    if st.button("💾 Save Prediction"):
                        # Add prediction to input data
                        result_df = input_df.copy()
                        result_df['station_name'] = station_name
                        result_df['predicted_sales_volume'] = prediction
                        result_df['prediction_timestamp'] = pd.Timestamp.now()
                        
                        # Save to CSV
                        result_df.to_csv('prediction_result.csv', index=False)
                        st.success("✅ Prediction saved to 'prediction_result.csv'")
                        
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")

# ==================== NO DATA WARNING ====================
elif st.session_state.df is None and menu not in ["📊 Predict"]:
    st.warning("⚠️ No data loaded!")
    st.info("Go to '📁 Upload' section to load your data")
    st.markdown("""
    **Supported formats:**
    - 📄 CSV files
    - 🗺️ GeoJSON files (with geometry)
    - 📦 GeoPackage files (with geometry)
    - 📊 Excel files (.xlsx)
    """)

# Footer
st.markdown("---")
st.caption("🌍 Petrol Station Sales Predictor | Upload data and predict sales volume")

# Instructions in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
 📋 Quick Guide:
1. **Upload** your station data (CSV/GeoJSON)
2. **Explore** data statistics and distributions  
3. **Clean** data by removing nulls/duplicates
4. **View** stations on map (if coordinates available)
5. **Train** prediction model with sales data
6. **Predict** sales for new stations

### 🎯 Target Variable:
- **sales_volume**: The value we want to predict
- Model uses road network and population features
""")
# ```

# ## **Key New Features:**

# 1. **🤖 Train Model Section**: 
#    - Trains model using your features (road distances, road counts, population)
#    - Shows feature importance
#    - Stores model in session state

# 2. **📊 Predict Section**:
#    - **Input form** for all your features:
#      - Distance to nearest road (meters/kilometers)
#      - Road counts within 1000m, 3000m, 5000m
#      - Road lengths and densities
#      - LGA population
#    - **Real-time prediction** with feature importance
#    - **Save predictions** to CSV
#    - **Visual explanations** of what's driving predictions

# 3. **🗺️ Map Enhancements**:
#    - Colors stations by sales volume (green=high, red=low)
#    - Shows station details in popups
#    - Uses your geometry column if available

# ## **To Use:**

# 1. **Upload** your CSV with the features you listed
# 2. **Go to "🤖 Train Model"** to train the prediction model
# 3. **Go to "📊 Predict"** to input new station features and get sales predictions

# The app will automatically detect which of your features are available and use them for training and prediction!