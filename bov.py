# Prescriptive analytics**, not just Predictive. Actionable insights and risk assessments.

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from io import BytesIO
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Page config
st.set_page_config(page_title="PetroSite Analyzer", layout="wide")

# Title
st.title("⛽ PetroSite Analyzer - Sales Potential Assessment")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'sales_stats' not in st.session_state:
    st.session_state.sales_stats = None

# Helper function for safe DataFrame display
def safe_dataframe_display(df, max_rows=10):
    """Display DataFrame without Arrow serialization warnings"""
    if df is None or len(df) == 0:
        return st.write("No data to display")
    
    df_display = df.head(max_rows).copy()
    for col in df_display.columns:
        if df_display[col].dtype == 'object':
            df_display[col] = df_display[col].astype(str)
    return st.dataframe(df_display)

# Risk assessment function
def assess_sales_potential(predicted_sales, sales_percentiles):
    """
    Assess sales potential with recommendations
    
    Parameters:
    -----------
    predicted_sales : float
        Predicted sales volume
    sales_percentiles : dict
        Percentile values from training data
    
    Returns:
    --------
    dict with assessment details
    """
    
    # Determine risk level
    if predicted_sales >= sales_percentiles.get('q75', 0):
        risk_level = "🟢 LOW RISK"
        category = "HIGH POTENTIAL"
        color = "green"
        recommendation = "✅ **GOLDMINE SITE** - Strong recommendation for investment"
        confidence = "High"
        
    elif predicted_sales >= sales_percentiles.get('q50', 0):
        risk_level = "🟡 MODERATE RISK"
        category = "AVERAGE POTENTIAL"
        color = "orange"
        recommendation = "⚠️ **MODERATE INVESTMENT** - Expect average returns, consider competitive analysis"
        confidence = "Medium"
        
    else:
        risk_level = "🔴 HIGH RISK"
        category = "LOW POTENTIAL"
        color = "red"
        recommendation = "❌ **HIGH-RISK SITE** - Proceed with caution, consider alternative locations"
        confidence = "Medium"
    
    # Calculate percentile rank
    if sales_percentiles:
        if predicted_sales >= sales_percentiles.get('q90', 0):
            percentile_rank = "Top 10%"
        elif predicted_sales >= sales_percentiles.get('q75', 0):
            percentile_rank = "Top 25%"
        elif predicted_sales >= sales_percentiles.get('q50', 0):
            percentile_rank = "Top 50%"
        else:
            percentile_rank = "Bottom 50%"
    else:
        percentile_rank = "N/A"
    
    return {
        'risk_level': risk_level,
        'category': category,
        'color': color,
        'recommendation': recommendation,
        'confidence': confidence,
        'percentile_rank': percentile_rank
    }

# Sidebar menu
menu = st.sidebar.selectbox(
    "Select Section",
    ["📁 Upload", "🔍 Analyze", "🧹 Clean", "🗺️ Map", "🤖 Train Model", "📊 Predict & Assess"]
)

# ==================== UPLOAD ====================
if menu == "📁 Upload":
    st.header("📁 Upload Historical Data")
    
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
                gdf = gpd.read_file(uploaded_file)
                st.session_state.gdf = gdf
                
                if 'geometry' in gdf.columns:
                    df = pd.DataFrame(gdf.drop(columns='geometry'))
                else:
                    df = pd.DataFrame(gdf)
                
                st.session_state.df = df
                st.success(f"✅ GeoJSON loaded: {len(df)} rows, {len(df.columns)} columns")
                st.info(f"📐 Contains {len(gdf)} spatial features")
                
            elif file_ext == 'gpkg':
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
        
        safe_dataframe_display(st.session_state.df)

# ==================== ANALYZE ====================
elif menu == "🔍 Analyze" and st.session_state.df is not None:
    st.header("🔍 Market Analysis Dashboard")
    df = st.session_state.df
    
    if 'sales_volume' in df.columns:
        # Calculate sales statistics for later use
        sales_stats = {
            'mean': df['sales_volume'].mean(),
            'median': df['sales_volume'].median(),
            'std': df['sales_volume'].std(),
            'q25': df['sales_volume'].quantile(0.25),
            'q50': df['sales_volume'].quantile(0.50),
            'q75': df['sales_volume'].quantile(0.75),
            'q90': df['sales_volume'].quantile(0.90)
        }
        st.session_state.sales_stats = sales_stats
        
        # Sales distribution analysis
        st.subheader("📊 Sales Volume Distribution")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Sales", f"{sales_stats['mean']:,.0f}")
        with col2:
            st.metric("Median Sales", f"{sales_stats['median']:,.0f}")
        with col3:
            st.metric("High Performers (Q3+)", f"{sales_stats['q75']:,.0f}+")
        with col4:
            st.metric("Top 10%", f"{sales_stats['q90']:,.0f}+")
        
        # Sales categories
        st.subheader("🎯 Performance Categories")
        
        performance_df = pd.DataFrame({
            'Category': ['Low (Bottom 25%)', 'Average (25-75%)', 'High (Top 25%)', 'Premium (Top 10%)'],
            'Sales Range': [
                f"< {sales_stats['q25']:,.0f}",
                f"{sales_stats['q25']:,.0f} - {sales_stats['q75']:,.0f}",
                f"{sales_stats['q75']:,.0f} - {sales_stats['q90']:,.0f}",
                f"> {sales_stats['q90']:,.0f}"
            ],
            'Count': [
                len(df[df['sales_volume'] < sales_stats['q25']]),
                len(df[(df['sales_volume'] >= sales_stats['q25']) & (df['sales_volume'] < sales_stats['q75'])]),
                len(df[(df['sales_volume'] >= sales_stats['q75']) & (df['sales_volume'] < sales_stats['q90'])]),
                len(df[df['sales_volume'] >= sales_stats['q90']])
            ]
        })
        
        safe_dataframe_display(performance_df)
        
        # Feature correlation with sales
        st.subheader("📈 Key Drivers Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if 'sales_volume' in numeric_cols:
            correlations = df[numeric_cols].corr()['sales_volume'].sort_values(ascending=False)
            
            # Get top 10 correlations
            top_correlations = correlations.iloc[1:11]  # Exclude sales_volume itself
            
            import plotly.express as px
            fig = px.bar(x=top_correlations.values, y=top_correlations.index,
                        title="Top 10 Features Correlated with Sales",
                        labels={'x': 'Correlation', 'y': 'Feature'},
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabs for detailed analysis
    tab1, tab2 = st.tabs(["📋 Basic Stats", "📊 Distributions"])
    
    with tab1:
        safe_dataframe_display(df.describe())
    
    with tab2:
        selected_col = st.selectbox("Select feature to analyze", df.columns)
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            import plotly.express as px
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

# ==================== CLEAN ====================
elif menu == "🧹 Clean" and st.session_state.df is not None:
    st.header("🧹 Clean Data")
    df = st.session_state.df.copy()
    
    st.write(f"**Current data shape:** {df.shape}")
    
    col1, col2 = st.columns(2)
    with col1:
        remove_nulls = st.checkbox("Remove null values", value=True)
        remove_duplicates = st.checkbox("Remove duplicates", value=True)
    with col2:
        drop_columns = st.multiselect("Drop columns", df.columns)
        keep_columns = st.multiselect("Keep only", df.columns)
    
    if st.button("🚀 Clean Data"):
        with st.spinner("Cleaning..."):
            if drop_columns:
                df = df.drop(columns=drop_columns)
            if keep_columns:
                df = df[keep_columns]
            if remove_nulls:
                df = df.dropna()
            if remove_duplicates:
                df = df.drop_duplicates()
            
            st.session_state.df = df
            st.success(f"✅ Cleaned: {df.shape}")
            safe_dataframe_display(df.head())

# ==================== MAP ====================
elif menu == "🗺️ Map" and st.session_state.gdf is not None:
    st.header("🗺️ Station Network Map")
    gdf = st.session_state.gdf
    
    try:
        import folium
        from streamlit_folium import st_folium
        
        if 'sales_volume' in gdf.columns:
            center_lat = gdf.geometry.centroid.y.mean()
            center_lon = gdf.geometry.centroid.x.mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            for idx, row in gdf.iterrows():
                if hasattr(row.geometry, 'x'):
                    sales = row.get('sales_volume', 0)
                    
                    # Determine color based on sales
                    if sales >= st.session_state.get('sales_stats', {}).get('q75', 0):
                        color = 'green'
                        size = 10
                    elif sales >= st.session_state.get('sales_stats', {}).get('q25', 0):
                        color = 'blue'
                        size = 8
                    else:
                        color = 'red'
                        size = 6
                    
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=size,
                        popup=f"Sales: {sales:,.0f}",
                        color=color,
                        fill=True
                    ).add_to(m)
            
            st_folium(m, width=1200, height=600)
            
    except Exception as e:
        st.warning(f"Map display error: {str(e)}")

# ==================== TRAIN MODEL ====================
elif menu == "🤖 Train Model" and st.session_state.df is not None:
    st.header("🤖 Train Prediction Model")
    df = st.session_state.df
    
    if 'sales_volume' not in df.columns:
        st.error("❌ 'sales_volume' column required")
    else:
        # Feature selection
        feature_options = [
            'distance_to_nearest_road_m', 'distance_to_nearest_road_km',
            'roads_within_1000m', 'road_length_within_1000m', 'road_density_1000m',
            'roads_within_3000m', 'road_length_within_3000m', 'road_density_3000m',
            'roads_within_5000m', 'road_length_within_5000m', 'road_density_5000m',
            'lga_population'
        ]
        
        available_features = [f for f in feature_options if f in df.columns]
        
        if available_features:
            st.success(f"✅ {len(available_features)} features available")
            
            model_type = st.selectbox("Model Type", ["Linear Regression","Lasso Regression","Random Forest", "XGBoost"])
            test_size = st.slider("Test Size %", 10, 40, 20) / 100
            
            if st.button("🚀 Train Model"):
                with st.spinner("Training..."):
                    try:
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.ensemble import RandomForestRegressor
                        from sklearn.linear_model import LinearRegression,Lasso
                        from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
                        
                        X = df[available_features].fillna(df[available_features].mean())
                        y = df['sales_volume'].fillna(df['sales_volume'].mean())
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        if model_type == "Random Forest":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif model_type == "Linear Regression":
                            model = LinearRegression()
                        elif model_type == "Lasso Regression":
                            model = Lasso(alpha=0.1,max_iter=1000,random_state=42)
                        else:
                            try:
                                from xgboost import XGBRegressor
                                model = XGBRegressor(n_estimators=100,max_depth=6,learning_rate=0.1,random_state=42,n_jobs=-1)
                            except:
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                        
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        st.session_state.model = {
                            'model': model,
                            'scaler': scaler,
                            'features': available_features,
                            'r2': r2,
                            'mse':mse,
                            'mae':mae
                        }
                        
                        st.success(f"✅ Model trained! R²: {r2:.4f}")
                        st.success(f"✅ Model trained! MAE: {mae:.4f}")
                        st.success(f"✅ Model trained! MSE: {mse:.4f}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# ==================== PREDICT & ASSESS ====================
elif menu == "📊 Predict & Assess":
    st.header("📊 Site Assessment & Recommendation")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in '🤖 Train Model' section")
    else:
        model_info = st.session_state.model
        features = model_info['features']
        
        st.success(f"✅ Model ready with R²: {model_info['r2']:.4f}")
        st.success(f"✅ Model ready with MAE: {model_info['mae']:.4f}")
        st.success(f"✅ Model ready with MSE: {model_info['mse']:.4f}")
        
        
        # Input form
        st.subheader("📍 Enter Site Parameters")
        
        col1, col2, col3 = st.columns(3)
        input_values = {}
        
        with col1:
            st.markdown("**🛣️ Road Accessibility**")
            for feat in [f for f in features if 'distance' in f]:
                if 'km' in feat:
                    input_values[feat] = st.number_input(f"{feat}", 0.0, 50.0, 1.0, 0.1)
                else:
                    input_values[feat] = st.number_input(f"{feat}", 0, 50000, 1000, 100)
        
        with col2:
            st.markdown("**📏 Road Network Density**")
            for feat in [f for f in features if 'road' in f and 'distance' not in f]:
                if 'density' in feat:
                    input_values[feat] = st.number_input(f"{feat}", 0.0, 5.0, 0.5, 0.1)
                elif 'roads_within' in feat:
                    input_values[feat] = st.number_input(f"{feat}", 0, 100, 5, 1)
                elif 'road_length' in feat:
                    input_values[feat] = st.number_input(f"{feat}", 0, 100000, 5000, 100)
        
        with col3:
            st.markdown("**👥 Market Potential**")
            for feat in [f for f in features if 'population' in f]:
                input_values[feat] = st.number_input(f"{feat}", 0, 1000000, 50000, 1000)
            
            station_name = st.text_input("Site Name", "Proposed Station")
        
        # Make prediction
        if st.button("🔮 Assess Site Potential", type="primary"):
            with st.spinner("Analyzing..."):
                try:
                    # Prepare input
                    input_df = pd.DataFrame([input_values])
                    input_scaled = model_info['scaler'].transform(input_df)
                    
                    # Predict sales
                    predicted_sales = model_info['model'].predict(input_scaled)[0]
                    
                    # Get sales statistics if available
                    sales_stats = st.session_state.get('sales_stats', {
                        'q25': predicted_sales * 0.5,
                        'q50': predicted_sales * 0.75,
                        'q75': predicted_sales * 0.9,
                        'q90': predicted_sales * 1.1
                    })
                    
                    # Assess potential
                    assessment = assess_sales_potential(predicted_sales, sales_stats)
                    
                    # Display results
                    st.markdown("---")
                    
                    # Result header
                    st.markdown(f"## ⛽ **{station_name}** - Site Assessment Report")
                    
                    # Main metrics card
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Predicted Sales Volume",
                            f"{predicted_sales:,.0f}",
                            "units/month"
                        )
                    
                    with col2:
                        st.metric(
                            "Risk Level",
                            assessment['risk_level'],
                            assessment['category']
                        )
                    
                    with col3:
                        st.metric(
                            "Performance Rank",
                            assessment['percentile_rank'],
                            f"Confidence: {assessment['confidence']}"
                        )
                    
                    # Recommendation box
                    st.markdown(f"""
                    <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {assessment["color"]};'>
                    <h3>📋 Investment Recommendation</h3>
                    <p style='font-size: 18px;'>{assessment['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed analysis
                    st.markdown("---")
                    st.subheader("📊 Detailed Analysis")
                    
                    tab1, tab2, tab3 = st.tabs(["📈 Performance Metrics", "🔍 Feature Impact", "💡 Action Plan"])
                    
                    with tab1:
                        st.markdown(f"""
                        **Sales Performance Context:**
                        - Your predicted sales: **{predicted_sales:,.0f} units/month**
                        - Market average (Q2): **{sales_stats.get('q50', 'N/A'):,.0f} units/month**
                        - High performer threshold (Q3): **{sales_stats.get('q75', 'N/A'):,.0f} units/month**
                        - Premium performer (Q4): **{sales_stats.get('q90', 'N/A'):,.0f} units/month**
                        
                        **Interpretation:**
                        - This site falls in the **{assessment['category'].lower()}** category
                        - Compared to existing stations: **{assessment['percentile_rank']}**
                        """)
                    
                    with tab2:
                        # Feature importance
                        if hasattr(model_info['model'], 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': features,
                                'Impact': model_info['model'].feature_importances_,
                                'Your Value': [input_values.get(f, 'N/A') for f in features]
                            }).sort_values('Impact', ascending=False)
                            
                            # Highlight key factors
                            top_features = importance_df.head(5)
                            
                            st.markdown("**🎯 Top 5 Influencing Factors:**")
                            for _, row in top_features.iterrows():
                                value = row['Your Value']
                                if 'distance' in row['Feature'] and value is not None:
                                    if value > 3000:
                                        status = "⚠️ Too far"
                                    elif value < 1000:
                                        status = "✅ Optimal"
                                    else:
                                        status = "🟡 Acceptable"
                                elif 'population' in row['Feature'] and value is not None:
                                    if value > 100000:
                                        status = "✅ Strong market"
                                    elif value > 50000:
                                        status = "🟡 Moderate market"
                                    else:
                                        status = "⚠️ Small market"
                                else:
                                    status = ""
                                
                                st.write(f"- **{row['Feature']}**: {value} {status}")
                    
                    with tab3:
                        # Action plan based on assessment
                        if assessment['category'] == "HIGH POTENTIAL":
                            st.markdown("""
                            **✅ Recommended Actions:**
                            1. **Immediate Acquisition** - Secure site quickly
                            2. **Premium Investment** - Allocate maximum budget
                            3. **Fast-track Development** - Aim for 6-month launch
                            4. **Brand Positioning** - Market as flagship location
                            5. **Competitive Monitoring** - Watch for new entrants
                            """)
                        elif assessment['category'] == "AVERAGE POTENTIAL":
                            st.markdown("""
                            **⚠️ Recommended Actions:**
                            1. **Negotiate Terms** - Seek favorable lease/purchase
                            2. **Phased Investment** - Start with basic facilities
                            3. **Competitive Analysis** - Study nearby stations
                            4. **Market Testing** - Consider pilot operations
                            5. **Exit Strategy** - Have contingency plans
                            """)
                        else:
                            st.markdown("""
                            **❌ Recommended Actions:**
                            1. **Reconsider Investment** - Explore alternatives
                            2. **Cost-Benefit Analysis** - Verify all assumptions
                            3. **Site Improvement** - Can accessibility be enhanced?
                            4. **Partnership Model** - Consider franchise instead
                            5. **Exit Strategy** - Clear cancellation clauses
                            """)
                    
                    # Export report
                    st.markdown("---")
                    if st.button("📄 Generate Full Report (PDF)"):
                        st.info("Report generation would create a PDF with:")
                        st.write("1. Executive Summary")
                        st.write("2. Site Analysis Details")
                        st.write("3. Market Context")
                        st.write("4. Financial Projections")
                        st.write("5. Risk Assessment Matrix")
                        
                except Exception as e:
                    st.error(f"Assessment error: {str(e)}")

# ==================== NO DATA ====================
elif st.session_state.df is None and menu not in ["📊 Predict & Assess"]:
    st.warning("⚠️ Please upload data first!")
    st.info("Go to '📁 Upload' to start analysis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<b>⛽ PetroSite Analyzer v2.0</b> | Predictive Site Assessment for Fuel Retail | 
<span style='color: #1E88E5;'>Beyond Prediction → Prescriptive Analytics</span>
</div>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🎯 Assessment Framework:
**🟢 GREEN ZONE** (Q3+)
- Predicted sales > 75th percentile
- Low risk, high confidence
- **Action:** Invest aggressively

**🟡 YELLOW ZONE** (Q2-Q3)
- Sales between 25th-75th percentile
- Moderate risk/return
- **Action:** Proceed with caution

**🔴 RED ZONE** (Bottom 25%)
- Sales < 25th percentile
- High risk, low confidence
- **Action:** Re-evaluate or avoid
""")
# ```

# ## **Key New Features:**

# ### **1. 🎯 Three-Tier Assessment System:**
# - **🟢 GREEN ZONE** (Top 25%): "GOLDMINE SITE" - Strong recommendation
# - **🟡 YELLOW ZONE** (25-75%): "MODERATE INVESTMENT" - Proceed with analysis
# - **🔴 RED ZONE** (Bottom 25%): "HIGH-RISK SITE" - Proceed with caution

# ### **2. 📊 Performance Context:**
# - Shows where predicted sales rank compared to existing stations
# - Provides percentile rankings (Top 10%, Top 25%, etc.)
# - Compares against market benchmarks

# ### **3. 💡 Actionable Recommendations:**
# - **High Potential Sites:** Immediate acquisition, premium investment
# - **Average Sites:** Negotiate terms, phased investment
# - **Low Potential Sites:** Reconsider, cost-benefit analysis, exit strategies

# ### **4. 📈 Detailed Analysis Tabs:**
# 1. **Performance Metrics** - Sales context and interpretation
# 2. **Feature Impact** - What's driving the prediction
# 3. **Action Plan** - Specific steps based on assessment

# ### **5. 🎨 Visual Risk Communication:**
# - Color-coded risk levels
# - Clear investment recommendations
# - Confidence indicators

# ## **How It Works:**

# 1. **Upload** historical station data
# 2. **Analyze** market performance categories
# 3. **Train** the ML model
# 4. **Input** new site parameters
# 5. **Get** a complete assessment with:
#    - Predicted sales volume
#    - Risk level (Green/Yellow/Red)
#    - Percentile ranking
#    - Specific recommendations
#    - Action plan

# ## **Business Value:**
# - **Transforms predictions into decisions**
# - **Reduces investment risk** through clear risk categories
# - **Provides justification** for investment decisions
# - **Creates audit trail** for site selection process
# - **Aligns with business KPIs** (ROI, risk management)

# This creates a **decision support system** that tells investors not just "what sales will be" but **"what to do about it"** - exactly what you wanted!