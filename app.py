import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import json
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings
import os
import sqlite3
from datetime import datetime, timedelta
import joblib

# Suppress specific warnings from statsmodels and pandas
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning) # Suppress pmdarima deprecation warnings

# --- Global Configurations / Data Sources ---
TARGET_FOOD_ITEMS = ['gari', 'groundnuts', 'maize', 'millet', 'sorghum', "cassava_meal"]
CAPITALIZED_FOOD_ITEMS = [item.capitalize() for item in TARGET_FOOD_ITEMS]

DATABASE_NAME = 'food_price_data.db' # SQLite database file

# --- Database Connection ---
def get_db_connection():
    """Establishes and returns a SQLite database connection."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

# --- Cached Data Loading Functions (FROM DATABASE) ---

@st.cache_data(ttl=3600 * 24) # Cache for 24 hours
def load_geojson():
    """Loads the GeoJSON file for Nigeria states."""
    try:
        # Check if the file exists in the current directory or a subdirectory 'nga'
        if os.path.exists("nga/ngs.json"):
            filepath = "nga/ngs.json"
        elif os.path.exists("ngs.json"): # Fallback for local testing if file is in root
            filepath = "ngs.json"
        else:
            st.error("GeoJSON file 'ngs.json' not found. Please ensure it's in 'nga/' or the root directory.")
            return None

        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return None

@st.cache_data(ttl=3600 * 24) # Cache for 24 hours
def fetch_food_prices_from_db(selected_food_items_lower, years_back):
    """Fetches food price data from the database for selected items and years."""
    conn = get_db_connection()
    current_year = datetime.now().year
    start_year = current_year - years_back
    
    placeholders = ', '.join('?' * len(selected_food_items_lower))
    query = f'''
        SELECT State, Year, Month, Food_Item, Unit, Price
        FROM food_prices
        WHERE Food_Item IN ({placeholders}) AND Year >= ?
        ORDER BY State, Year, Month, Food_Item
    '''
    
    try:
        df = pd.read_sql_query(query, conn, params=selected_food_items_lower + [start_year])
        if df.empty:
            st.warning("No food price data found in the database for the selected criteria.")
        return df
    except Exception as e:
        st.error(f"Error fetching food prices from database: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=3600 * 24) # Cache for 24 hours
def load_inflation_data_from_db():
    """Loads inflation data from the database."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT Year, Month, foodYearOn FROM inflation_data ORDER BY Year, Month", conn)
        if df.empty:
            st.warning("No inflation data found in the database.")
        return df
    except Exception as e:
        st.error(f"Error loading inflation data from database: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=3600 * 24) # Cache for 24 hours
def get_rainfall_data_from_db():
    """Fetches rainfall data from the database."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT state, Year, Month, rain FROM rainfall_data ORDER BY state, Year, Month", conn)
        if df.empty:
            st.warning("No rainfall data found in the database.")
        return df
    except Exception as e:
        st.error(f"Error loading rainfall data from database: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@st.cache_data(ttl=3600 * 24) # Cache for 24 hours
def get_historical_average_rainfall(state, df_rainfall_history):
    """
    Calculates the multi-year monthly average rainfall for a given state.
    """
    if df_rainfall_history.empty:
        # st.warning(f"No historical rainfall data available to compute averages for {state}.")
        return pd.DataFrame()

    state_rainfall = df_rainfall_history[df_rainfall_history['state'] == state].copy()
    if state_rainfall.empty:
        # st.warning(f"No historical rainfall data for {state} to compute averages.")
        return pd.DataFrame()

    # Calculate multi-year monthly average
    monthly_avg_rain = state_rainfall.groupby('Month')['rain'].mean().reset_index()
    monthly_avg_rain.rename(columns={'rain': 'avg_rain'}, inplace=True)
    return monthly_avg_rain

@st.cache_data(ttl=3600 * 24) # Cache the final merged dataset for 24 hours
def load_and_merge_all_data_from_db(selected_food_items_lower, years_back):
    """Loads and merges food price, inflation, and rainfall data from the database."""
    with st.spinner("Loading food prices from local database..."):
        df_food_prices = fetch_food_prices_from_db(selected_food_items_lower, years_back)
        if df_food_prices.empty:
            return None, None

    with st.spinner("Loading inflation data from local database..."):
        df_inflation = load_inflation_data_from_db()
        if df_inflation.empty:
            return None, None

    df_merged = df_food_prices.merge(df_inflation, on=['Year', 'Month'], how='left')

    with st.spinner("Loading historical rainfall data from local database..."):
        df_rainfall = get_rainfall_data_from_db()
        if df_rainfall.empty:
            st.warning("Could not fetch any historical rainfall data from DB.")
            # Ensure df_merged_rain_temp is compatible for merge even if empty
            df_merged_rain_temp = pd.DataFrame(columns=['state', 'Year', 'Month', 'rain'])
        else:
            df_merged_rain_temp = df_rainfall

    df_final_merged = df_merged.merge(
        df_merged_rain_temp,
        left_on=['State', 'Year', 'Month'],
        right_on=['state', 'Year', 'Month'],
        how='left'
    )
    if 'state' in df_final_merged.columns:
        df_final_merged.drop(columns=['state'], inplace=True)
    
    return df_final_merged, df_food_prices

# --- Helper function for extending exogenous variables for SARIMAX ---
def extend_exog_for_forecast(exog, steps, selected_state, df_full_merged):
    """
    Extends exogenous variables (CPI and Rainfall) for the forecast horizon.
    Future CPI is based on known data; future Rainfall uses multi-year monthly average.
    """
    if not isinstance(exog.index, pd.DatetimeIndex):
        raise ValueError("Exogenous variable index must be a DatetimeIndex.")

    future_dates = pd.date_range(start=exog.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')

    # --- Handling future CPI (foodYearOn) ---
    known_future_cpi_data = {
        '2025-06-01': 24,
        '2025-09-01': 24,
        '2025-12-01': 20,
        '2026-03-01': 18.5
    }
    known_future_cpi_series = pd.Series(known_future_cpi_data).astype(float)
    known_future_cpi_series.index = pd.to_datetime(known_future_cpi_series.index)

    future_cpi_filled = pd.Series(index=future_dates, dtype=float)
    for date, value in known_future_cpi_series.items():
        if date in future_cpi_filled.index:
            future_cpi_filled.loc[date] = value
    
    # Fill any remaining NaNs in future_cpi_filled with the last known CPI or last historical CPI
    if not known_future_cpi_series.empty:
        last_known_cpi_value = known_future_cpi_series.iloc[-1]
    else:
        last_known_cpi_value = exog['foodYearOn'].iloc[-1] if 'foodYearOn' in exog.columns and not exog['foodYearOn'].empty else 0.0
    future_cpi_filled = future_cpi_filled.fillna(last_known_cpi_value)
    
    future_cpi = future_cpi_filled.values

    # --- Handling future Rainfall: Using multi-year monthly average ---
    df_rainfall_history = df_full_merged[['State', 'Year', 'Month', 'rain']].drop_duplicates().rename(columns={'State': 'state'}).dropna(subset=['rain'])
    monthly_avg_rain = get_historical_average_rainfall(selected_state, df_rainfall_history)

    future_rain = np.full(steps, np.nan)

    if not monthly_avg_rain.empty:
        for i, date in enumerate(future_dates):
            month_match = monthly_avg_rain[monthly_avg_rain['Month'] == date.month]
            if not month_match.empty:
                future_rain[i] = month_match['avg_rain'].iloc[0]
        
        # Fallback for any months not found in average or if no historical rain at all
        if np.isnan(future_rain).any():
            if 'rain' in exog.columns and not exog['rain'].empty:
                last_historical_rain = exog['rain'].iloc[-1]
                future_rain = np.nan_to_num(future_rain, nan=last_historical_rain)
            else:
                future_rain = np.nan_to_num(future_rain, nan=0.0) # Default if no historical rain data
    else:
        # If no monthly_avg_rain available, use last historical rain or 0.0
        last_historical_rain_fallback = exog['rain'].iloc[-1] if 'rain' in exog.columns and not exog['rain'].empty else 0.0
        future_rain = np.full(steps, last_historical_rain_fallback)

    future_exog = pd.DataFrame({
        'foodYearOn': future_cpi,
        'rain': future_rain
    }, index=future_dates)

    return future_exog

# --- Prepare time series and exogenous variables for SARIMAX ---
def prepare_time_series_with_exog(df, state_name, food_item):
    """
    Prepares the time series (Price) and exogenous variables (foodYearOn, rain)
    for SARIMAX modeling for the selected state and food item.
    Handles missing values by forward-filling and then backward-filling.
    """
    series_data = df[
        (df['State'] == state_name) &
        (df['Food_Item'] == food_item)
    ].copy()

    if series_data.empty:
        return pd.Series(dtype='float64'), pd.DataFrame(), pd.DataFrame()

    series_data['Date'] = pd.to_datetime(series_data['Year'].astype(str) + '-' + series_data['Month'].astype(str) + '-01')
    series_data.set_index('Date', inplace=True)
    series_data.sort_index(inplace=True)

    # Handle missing values: ffill then bfill for all relevant columns
    for col in ['Price', 'foodYearOn', 'rain']:
        if col in series_data.columns:
            series_data[col] = series_data[col].fillna(method='ffill')
            series_data[col] = series_data[col].fillna(method='bfill')
        else:
            st.warning(f"Column '{col}' not found in prepared data for {food_item} in {state_name}. This might affect model accuracy.")
            series_data[col] = 0.0 # Default if column is entirely missing

    # Ensure no NaNs after imputation, drop rows if any remain (shouldn't if ffill/bfill work)
    series_data.dropna(subset=['Price', 'foodYearOn', 'rain'], inplace=True)

    if series_data.empty:
        st.warning(f"No complete data available for {food_item} in {state_name} after handling missing values.")
        return pd.Series(dtype='float64'), pd.DataFrame(), pd.DataFrame()

    ts = series_data['Price']
    exog = series_data[['foodYearOn', 'rain']]
    
    return ts, exog, series_data # Also return series_data for visualization

# --- SARIMAX Forecasting Function ---
@st.cache_resource(ttl=3600) # Use st.cache_resource for models and large objects
def forecast_food_prices_sarimax(ts, exog, food_item, state_name, forecast_steps, df_full_merged):
    """
    Fits SARIMAX on full series with exogenous variables and forecasts
    the next forecast_steps months.
    Uses auto_arima to find the best SARIMAX order.
    """
    st.info(f"Training SARIMAX model for {food_item} in {state_name}. This may take a moment...")

    model_filename = f"sarimax_model_{food_item.lower()}_{state_name.replace(' ', '_').lower()}.joblib"

    # Try to load the model if it exists
    if os.path.exists(model_filename):
        try:
            model_fit = joblib.load(model_filename)
            st.info(f"Loaded pre-trained model from {model_filename}")
        except Exception as e:
            st.error(f"Error loading model from {model_filename}: {e}. Retraining model.")
            model_fit = None # Force retraining if load fails
    else:
        model_fit = None

    if model_fit is None:
        # Find best orders using auto_arima with exogenous variables
        try:
            model_auto = auto_arima(ts, exogenous=exog, seasonal=True, m=12, trace=False,
                                    suppress_warnings=True, error_action='ignore', stepwise=True,
                                    max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2)
            order = model_auto.order
            seasonal_order = model_auto.seasonal_order
            st.write(f"Optimal SARIMAX order: {order}, Seasonal order: {seasonal_order}")

            # Fit final SARIMAX model
            model = SARIMAX(ts, exog=exog, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)

            # Save the fitted model
            joblib.dump(model_fit, model_filename)
            st.info(f"Model saved to {model_filename}")
        except Exception as e:
            st.error(f"Error during SARIMAX model training: {e}")
            return pd.Series(dtype='float64') # Return empty series on error

    # Extend exogenous variables for the forecast period
    future_exog = extend_exog_for_forecast(exog, forecast_steps, state_name, df_full_merged)

    # Forecast forecast_steps ahead
    try:
        forecast = model_fit.forecast(steps=forecast_steps, exog=future_exog)
        return forecast
    except Exception as e:
        st.error(f"Error during SARIMAX forecast: {e}")
        return pd.Series(dtype='float64') # Return empty series on error

# --- Streamlit App Setup ---
st.set_page_config(layout="wide", page_title="Nigerian Food Price Dashboard")

# --- Sidebar UI ---
st.sidebar.title("🧊 Filter Options")
selected_food_items_explorer = st.sidebar.multiselect("Select Food Items:", CAPITALIZED_FOOD_ITEMS, default=['Maize'])
years_back_explorer = st.sidebar.slider("Years Back for Explorer:", min_value=1, max_value=10, value=5)

# --- Main Page UI ---
st.title("🥦 Nigerian Food Price Data Explorer & Predictor")
st.markdown("""
Welcome to the interactive dashboard to explore food price trends across Nigerian states and predict future prices.

👉 Data is periodically fetched and stored in a local database for faster access and reduced dependency on external APIs.
""")

tab1, tab2 = st.tabs(["📊 Data Explorer", "🧠 Predictor"])

# Initialize session state for merged data if not already present
if 'df_full_merged' not in st.session_state:
    st.session_state.df_full_merged = None
if 'df_food_prices_for_explorer' not in st.session_state:
    st.session_state.df_food_prices_for_explorer = None

# Pre-load all data from the database once for the app to use
# This uses st.cache_data, so it will only run on first load or after cache invalidation (e.g., ttl expires)
all_food_items_lower = [item.lower() for item in CAPITALIZED_FOOD_ITEMS]
st.session_state.df_full_merged, st.session_state.df_food_prices_for_explorer = load_and_merge_all_data_from_db(
    all_food_items_lower, years_back=10 # Load a good range for both explorer and predictor
)

# --- Data Explorer Tab ---
with tab1:
    st.markdown("This tab lets you analyze food price trends, map data, and download cleaned datasets.")

    if st.session_state.df_food_prices_for_explorer is not None and not st.session_state.df_food_prices_for_explorer.empty:
        # Filter the full loaded data based on sidebar selections for explorer
        food_data_explorer_filtered = st.session_state.df_food_prices_for_explorer[
            (st.session_state.df_food_prices_for_explorer['Food_Item'].isin(selected_food_items_explorer)) &
            (st.session_state.df_food_prices_for_explorer['Year'] >= (datetime.now().year - years_back_explorer))
        ].copy()

        if food_data_explorer_filtered.empty:
            st.info("No data available for the selected food items and years in the explorer. Try adjusting filters.")
        else:
            food_data_explorer_filtered['Year'] = food_data_explorer_filtered['Year'].astype(int)
            food_data_explorer_filtered['Date'] = pd.to_datetime(food_data_explorer_filtered[['Year', 'Month']].assign(DAY=1))

            # Data Quality Info
            st.markdown("#### 📊 Data Quality Check")
            st.markdown("This section helps you assess the completeness and reliability of the dataset.")
            total = len(food_data_explorer_filtered)
            missing_price = food_data_explorer_filtered['Price'].isna().sum()
            zero_price = (food_data_explorer_filtered['Price'] == 0).sum()
            st.info(f"Missing prices: {missing_price} | Zero prices: {zero_price} | Total entries: {total}")

            # Choropleth Map
            st.markdown("#### 🗺️ Choropleth Map")
            st.markdown("Visualize average food prices by state using a color-coded map.")
            nigeria_geojson = load_geojson() # Load GeoJSON
            if nigeria_geojson: # Proceed only if geojson loaded
                try:
                    available_map_items = food_data_explorer_filtered['Food_Item'].unique()
                    if selected_food_items_explorer and available_map_items.size > 0:
                        # Ensure the default selected item for map is in the available items
                        initial_map_item = next((item for item in selected_food_items_explorer if item in available_map_items), available_map_items[0])
                        map_item = st.selectbox("Select food item to map:", available_map_items, index=list(available_map_items).index(initial_map_item))
                    else:
                        map_item = None
                        st.info("No food items available to map after filtering.")
                    
                    if map_item:
                        latest_data = food_data_explorer_filtered[food_data_explorer_filtered['Food_Item'] == map_item].groupby("State")['Price'].mean().reset_index()
                        fig_map = px.choropleth(
                            latest_data, geojson=nigeria_geojson, featureidkey="properties.NAME_1",
                            locations="State", color="Price", color_continuous_scale="YlOrRd",
                            title=f"Avg. Price of {map_item} by State", scope="africa"
                        )
                        fig_map.update_geos(fitbounds="locations", visible=False)
                        st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.error(f"GeoJSON or mapping error: {e}")
            else:
                st.warning("Map cannot be displayed as GeoJSON data is missing.")

            # Line Chart (Price Trends)
            st.markdown("#### 📉 Price Trends Over Time")
            st.markdown("Shows monthly trends in average prices for each selected food item.")

            unique_states_trends = sorted(food_data_explorer_filtered['State'].dropna().unique())
            if unique_states_trends:
                selected_state_trend = st.selectbox("Select State to View Trends:", unique_states_trends, key="state_trend_select")

                for item in selected_food_items_explorer:
                    df_item_state = food_data_explorer_filtered[
                        (food_data_explorer_filtered['Food_Item'] == item) &
                        (food_data_explorer_filtered['State'] == selected_state_trend)
                    ].groupby('Date')['Price'].mean().reset_index()

                    if not df_item_state.empty:
                        fig = px.line(df_item_state, x='Date', y='Price', title=f"{item} - Price Over Time in {selected_state_trend}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No data available for {item} in {selected_state_trend} for the selected period.")
            else:
                st.info("No states available for trend analysis.")

            # Multi-Food Trend in One State
            st.markdown("#### 📌 Multi-Food Trend in a Single State")
            st.markdown("Compare the price movement of different food items within a selected state.")
            unique_states_explorer = sorted(food_data_explorer_filtered['State'].dropna().unique())
            if unique_states_explorer:
                selected_state_compare_explorer = st.selectbox("Select State for Comparison:", unique_states_explorer, key="multi_food_state_explorer")

                df_state_explorer = food_data_explorer_filtered[food_data_explorer_filtered['State'] == selected_state_compare_explorer].copy()
                if not df_state_explorer.empty:
                    fig_multi_food = px.line(
                        df_state_explorer, x='Date', y='Price', color='Food_Item',
                        title=f"Food Prices Over Time in {selected_state_compare_explorer}",
                        labels={"Price": "Price (₦ per 100 KG)", "Date": "Date", "Food_Item": "Food Item"}
                    )
                    fig_multi_food.update_layout(legend_title_text="Food Item")
                    st.plotly_chart(fig_multi_food, use_container_width=True)
                else:
                    st.info(f"No data for selected food items in {selected_state_compare_explorer}.")
            else:
                st.info("No states available for comparison after data fetch.")

            # --- Smart Insights Section ---
            st.markdown("---")
            st.markdown("#### ✨ Smart Insights: Highest Average Prices by State")
            st.markdown("This section highlights which state has the highest average price for each food item across the selected period.")

            avg_prices_by_state_item = food_data_explorer_filtered.groupby(['Food_Item', 'State'])['Price'].mean().reset_index()

            if not avg_prices_by_state_item.empty:
                insights_data = []
                for food_item in avg_prices_by_state_item['Food_Item'].unique():
                    df_item_prices = avg_prices_by_state_item[avg_prices_by_state_item['Food_Item'] == food_item]
                    if not df_item_prices.empty:
                        highest_price_state_row = df_item_prices.loc[df_item_prices['Price'].idxmax()]
                        insights_data.append({
                            "Food Item": food_item,
                            "State with Highest Avg Price": highest_price_state_row['State'],
                            "Highest Avg Price (₦/100KG)": f"₦{highest_price_state_row['Price']:.2f}"
                        })
                
                if insights_data:
                    insights_df = pd.DataFrame(insights_data)
                    insights_df.set_index("Food Item", inplace=True)
                    st.dataframe(insights_df, use_container_width=True)
                else:
                    st.info("Could not generate insights. No sufficient data found for price comparison across states.")
            else:
                st.info("No data available to generate insights on highest average prices by state.")
            
            st.markdown("---")

            # Download CSV
            st.download_button(
                label="📥 Download Explorer Data (CSV)",
                data=food_data_explorer_filtered.to_csv(index=False).encode('utf-8'),
                file_name="nigeria_food_prices_explorer.csv",
                mime="text/csv"
            )
    else:
        st.info("No data loaded from the local database. Ensure the ingestion script has been run.")

# --- Predictor Tab ---
with tab2:
    st.header("📈 Food Price Predictor")
    st.markdown("Forecast future food prices based on historical data, inflation, and rainfall.")

    if st.session_state.df_full_merged is None or st.session_state.df_full_merged.empty:
        st.warning("No data available for prediction. Please ensure the local database has been populated by running the ingestion script.")
    else:
        # User inputs for Prediction
        st.subheader("Prediction Settings")

        unique_states_predictor = sorted(st.session_state.df_full_merged['State'].unique().tolist())
        selected_state_predictor = st.selectbox("Select State", unique_states_predictor, key="predictor_state_select")

        available_food_items_predictor = sorted(
            st.session_state.df_full_merged[st.session_state.df_full_merged['State'] == selected_state_predictor]['Food_Item'].unique().tolist()
        )
        selected_crop_predictor = st.selectbox("Select Crop", available_food_items_predictor, key="predictor_crop_select")

        forecast_months = st.slider("Number of Months to Forecast (Max 12)", 1, 12, 3, key="forecast_months_slider")

        if st.button("Generate Price Forecast", key="generate_forecast_button"):
            st.markdown("---")
            st.subheader(f"Forecasting {selected_crop_predictor} Prices in {selected_state_predictor}")

            ts, exog, series_data_for_viz = prepare_time_series_with_exog(st.session_state.df_full_merged, selected_state_predictor, selected_crop_predictor)

            if ts.empty or exog.empty:
                st.warning(f"No complete historical data (price, inflation, or rainfall) available for {selected_crop_predictor} in {selected_state_predictor} after handling missing values. Please select another combination or check data.")
            elif len(ts) < 24: # Require at least 2 years of monthly data for a reasonable SARIMAX
                st.warning(f"Insufficient historical data ({len(ts)} months) for {selected_crop_predictor} in {selected_state_predictor} to build a robust SARIMAX model. At least 24 months recommended.")
            else:
                # --- Historical Data Visualizations (Before Forecast) ---
                st.subheader("Historical Data Visualizations")
                chart_option = st.selectbox(
                    "Select Historical Chart Type:",
                    ["Exogenous Variables (Inflation & Rainfall)", "Correlation Heatmap"],
                    key="historical_chart_type"
                )

                if chart_option == "Exogenous Variables (Inflation & Rainfall)":
                    df_viz_exog = series_data_for_viz[['foodYearOn', 'rain']].copy()

                    for col in ['foodYearOn', 'rain']:
                        min_val = df_viz_exog[col].min()
                        max_val = df_viz_exog[col].max()
                        if max_val > min_val:
                            df_viz_exog[f'{col}_normalized'] = (df_viz_exog[col] - min_val) / (max_val - min_val)
                        else:
                            df_viz_exog[f'{col}_normalized'] = 0.5 # Handle case where all values are the same

                    fig_exog = px.line(
                        df_viz_exog, x=df_viz_exog.index,
                        y=['foodYearOn_normalized', 'rain_normalized'],
                        title=f'Normalized Inflation (foodYearOn) and Rainfall Trends in {selected_state_predictor}',
                        labels={'value': 'Normalized Value', 'x': 'Date', 'variable': 'Variable'},
                        hover_data={
                            'foodYearOn_normalized': False,
                            'rain_normalized': False,
                            'foodYearOn': ':.2f', # Show original values in hover
                            'rain': ':.2f'      # Show original values in hover
                        }
                    )
                    fig_exog.for_each_trace(lambda t: t.update(name = 'Inflation (Normalized)' if t.name == 'foodYearOn_normalized' else 'Rainfall (Normalized)'))
                    st.plotly_chart(fig_exog, use_container_width=True)
                    st.caption("This chart displays the trends of inflation (foodYearOn) and rainfall, normalized to a common scale for visual comparison. Original values are shown in tooltips.")

                elif chart_option == "Correlation Heatmap":
                    correlation_matrix = series_data_for_viz[['Price', 'foodYearOn', 'rain']].corr()
                    fig_corr = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Correlation Heatmap (Price, Inflation, Rainfall)"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption("This heatmap shows the Pearson correlation coefficients between Price, Inflation (foodYearOn), and Rainfall. Values close to 1 indicate a strong positive correlation, -1 a strong negative correlation, and 0 no linear correlation.")

                st.markdown("---")
                st.subheader("Forecast Results")

                with st.spinner("Generating forecast using SARIMAX..."):
                    forecast_series = forecast_food_prices_sarimax(
                        ts,
                        exog,
                        selected_crop_predictor,
                        selected_state_predictor,
                        forecast_months,
                        st.session_state.df_full_merged # Pass df_full_merged for historical rainfall access
                    )

                if not forecast_series.empty:
                    st.success("Forecast generated successfully!")

                    forecast_df = pd.DataFrame({
                        'Date': forecast_series.index,
                        'Predicted_Price': forecast_series.values
                    })
                    forecast_df.set_index('Date', inplace=True)

                    historical_prices = ts.to_frame(name='Price')

                    plot_data = pd.concat([
                        historical_prices.assign(Type='Historical', Predicted_Price=np.nan),
                        forecast_df.assign(Type='Forecast', Price=np.nan)
                    ])
                    plot_data = plot_data.reset_index().melt(id_vars=['Date', 'Type'], var_name='Metric', value_name='Value')

                    plot_data = plot_data[plot_data['Metric'].isin(['Price', 'Predicted_Price'])]

                    plot_data['Metric'] = plot_data['Metric'].map({
                        'Price': 'Historical Price',
                        'Predicted_Price': 'Predicted Price'
                    })

                    fig_forecast = px.line(
                        plot_data, x='Date', y='Value', color='Metric',
                        title=f"Historical and Predicted {selected_crop_predictor} Prices in {selected_state_predictor}",
                        labels={'Value': 'Price (₦ per 100 KG)', 'Date': 'Date', 'Metric': 'Data Type'}
                    )
                    fig_forecast.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    st.markdown("#### Detailed Forecast")
                    st.dataframe(forecast_df, use_container_width=True)

                    st.download_button(
                        label="📥 Download Forecast Data (CSV)",
                        data=forecast_df.to_csv().encode('utf-8'),
                        file_name=f"{selected_crop_predictor}_{selected_state_predictor}_forecast.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Forecast could not be generated. Please check the data and try again.")
