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
from datetime import datetime, timedelta
import joblib # Still needed for model saving/loading if we re-introduce it, but removed for now

st.set_page_config(layout="wide", page_title="Nigerian Food Price Dashboard")

# Suppress specific warnings from statsmodels and pandas
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning) # Suppress pmdarima deprecation warnings

# --- Global Configurations / Data Sources ---
API_URL = "https://microdata.worldbank.org/index.php/api/tables/data/fcv/wld_2021_rtfp_v02_m"
TARGET_FOOD_ITEMS = ['gari', 'groundnuts', 'maize', 'millet', 'sorghum', "cassava_meal"]
CAPITALIZED_FOOD_ITEMS = [item.capitalize() for item in TARGET_FOOD_ITEMS]
INFLATION_FILEPATH = 'Inflation_Data_in_Excel.xlsx' # Ensure this file is in your app's directory

# Coordinates for rainfall API calls
state_coords = {
    "Kaduna": (10.609319, 7.429504),
    "Kebbi": (11.6600, 4.0900),
    "Katsina": (12.9881, 7.6000),
    "Borno": (11.8333, 13.1500),
    "Kano": (12.0000, 8.5167),
    "Abia": (5.5000, 7.5000),
    "Adamawa": (9.3265, 12.3984),
    "Zamfara": (12.1833, 6.2333),
    "Lagos": (6.5244, 3.3792),
    "Oyo": (7.8500, 3.9333),
    "Gombe": (10.2904, 11.1700),
    "Jigawa": (12.2280, 9.5616),
    "Yobe": (12.0000, 11.5000),
}

# --- Functions to Fetch Data from External Sources (APIs, Files) ---

@st.cache_data(ttl=3600 * 24) # Cache the result of this API call for 24 hours
def fetch_food_prices_from_api(api_url, food_items, country='Nigeria', years_back=10):
    """Fetches food price data from the World Bank API and returns a DataFrame."""
    limit = 10000
    offset = 0
    all_records = []

    close_price_columns = [f'c_{item}' for item in food_items]
    desired_api_fields = ['country', 'adm1_name', 'year', 'month', 'DATES'] + close_price_columns
    fields_param = ','.join(desired_api_fields)

    while True:
        params = {
            'limit': limit,
            'offset': offset,
            'country': country,
            'fields': fields_param
        }
        try:
            response = requests.get(api_url, params=params, timeout=60) # Increased timeout
            response.raise_for_status()
            data = response.json()
            if 'data' in data:
                records = data['data']
                if not records:
                    break
                all_records.extend(records)
                offset += limit
            else:
                break
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data from World Bank API: {e}")
            break
        except json.JSONDecodeError as e:
            st.error(f"Failed to decode JSON from API response for food prices: {e}")
            break

    df = pd.DataFrame(all_records)
    if df.empty:
        return pd.DataFrame()

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['month'] = pd.to_numeric(df['month'], errors='coerce')

    if 'DATES' in df.columns:
        df['DATES'] = pd.to_datetime(df['DATES'], errors='coerce')
        df.dropna(subset=['DATES', 'year', 'month'], inplace=True)
        current_year = datetime.now().year
        start_year = current_year - years_back
        df = df[df['year'] >= start_year].copy()
    else:
        current_year = datetime.now().year
        start_year = current_year - years_back
        df = df[df['year'] >= start_year].copy()
        df.dropna(subset=['year', 'month'], inplace=True)

    actual_price_columns_in_df = [col for col in close_price_columns if col in df.columns]
    if not actual_price_columns_in_df:
        return pd.DataFrame()

    for col in actual_price_columns_in_df:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=actual_price_columns_in_df, how='all')

    groupby_cols = ['country', 'adm1_name', 'year', 'month']
    if not all(col in df_clean.columns for col in groupby_cols):
        missing_cols = [col for col in groupby_cols if col not in df_clean.columns]
        st.error(f"Essential grouping columns missing after food price data fetch/clean: {missing_cols}. Check API fields and data schema.")
        return pd.DataFrame()

    df_avg = df_clean.groupby(groupby_cols)[actual_price_columns_in_df].mean().reset_index()
    df_avg['unit'] = '100 KG'
    df_avg.rename(columns={col: col[2:].capitalize() for col in actual_price_columns_in_df}, inplace=True)
    df_avg.rename(columns={'year': 'Year', 'month': 'Month', 'unit': 'Unit'}, inplace=True)

    if 'country' in df_avg.columns:
        df_avg.drop('country', axis=1, inplace=True)

    id_vars_for_melt = ['adm1_name', 'Year', 'Month', 'Unit']
    if not all(col in df_avg.columns for col in id_vars_for_melt):
        missing_cols = [col for col in id_vars_for_melt if col not in df_avg.columns]
        st.error(f"Columns for melting (id_vars) missing in food prices: {missing_cols}. Check previous renames or API fields.")
        return pd.DataFrame()

    df_long = pd.melt(
        df_avg,
        id_vars=id_vars_for_melt,
        var_name='Food_Item',
        value_name='Price'
    )
    df_long.rename(columns={'adm1_name': 'State'}, inplace=True)
    df_long = df_long[['State', 'Year', 'Month', 'Food_Item', 'Unit', 'Price']]
    df_long = df_long[df_long['State'] != 'Market Average']
    df_long.dropna(subset=['Price'], inplace=True)
    df_long.sort_values(by=['State', 'Year', 'Month', 'Food_Item'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)
    
    return df_long

@st.cache_data(ttl=3600 * 24) # Cache the result of this file load for 24 hours
def load_inflation_data_from_file(filepath):
    """Loads inflation data from an Excel file and returns a DataFrame."""
    try:
        if not os.path.exists(filepath):
            st.error(f"Inflation data file not found at: {filepath}. Please ensure it's in the app's directory.")
            return pd.DataFrame()
        df_inflation = pd.read_excel(filepath)
        df_inflation = df_inflation[['tyear', 'tmonth', 'foodYearOn']]
        df_inflation.rename(columns={'tyear': 'Year', 'tmonth': 'Month'}, inplace=True)
        df_inflation.dropna(subset=['Year', 'Month', 'foodYearOn'], inplace=True)
        return df_inflation
    except Exception as e:
        st.error(f"Error loading inflation data from Excel: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600 * 24) # Cache the result of this API call for 24 hours
def fetch_rainfall_data_from_api(state_coords, years_back=10):
    """Fetches rainfall data from Open-Meteo API for specified states and returns a DataFrame."""
    today = datetime.now()
    start_year_rainfall = today.year - years_back
    start_date_str = f"{start_year_rainfall}-01-01"
    end_date_str = (today - timedelta(days=1)).strftime("%Y-%m-%d") # Up to yesterday

    all_rainfall_data = []
    
    total_states = len(state_coords)
    progress_bar = st.progress(0)
    state_count = 0

    for state, coords in state_coords.items():
        lat, lon = coords
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date_str}&end_date={end_date_str}"
            f"&daily=rain_sum"
            f"&timezone=Africa%2FLagos"
        )
        try:
            response = requests.get(url, timeout=60) # Increased timeout
            response.raise_for_status()
            data = response.json()

            if 'daily' in data and 'rain_sum' in data['daily']:
                df = pd.DataFrame({
                    "date": data['daily']['time'],
                    "rain": data['daily']['rain_sum'],
                })
                df['date'] = pd.to_datetime(df['date'])
                df['Year'] = df['date'].dt.year
                df['Month'] = df['date'].dt.month
                
                monthly_summary = (
                    df
                    .groupby(['Year', 'Month'])[['rain']]
                    .sum()
                    .reset_index()
                )
                monthly_summary['state'] = state
                all_rainfall_data.append(monthly_summary)
            else:
                pass # Keep pass to avoid breaking flow if no data
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch rainfall data for {state}: {e}")
        except json.JSONDecodeError as e:
            st.error(f"Failed to decode JSON for rainfall data for {state}: {e}")
        
        state_count += 1
        progress_bar.progress(state_count / total_states)
    
    progress_bar.empty()
    if all_rainfall_data:
        df_rainfall = pd.concat(all_rainfall_data, ignore_index=True)
        return df_rainfall
    else:
        return pd.DataFrame()

@st.cache_data(ttl=3600 * 24) # Cache for 24 hours
def load_geojson():
    """Loads the GeoJSON file for Nigeria states."""
    try:
        if os.path.exists("nga/ngs.json"):
            filepath = "nga/ngs.json"
        elif os.path.exists("ngs.json"):
            filepath = "ngs.json"
        else:
            st.error("GeoJSON file 'ngs.json' not found. Please ensure it's in 'nga/' or the root directory.")
            return None

        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return None

def get_historical_average_rainfall(state, df_rainfall_history):
    """
    Calculates the multi-year monthly average rainfall for a given state.
    This function is used for pseudo-forecasting future rainfall values
    by assuming future months will have rainfall similar to their historical averages.
    """
    state_rain = df_rainfall_history[df_rainfall_history['state'] == state]
    if not state_rain.empty:
        return state_rain.groupby('Month')['rain'].mean().reset_index(name='avg_rain')
    return pd.DataFrame()


@st.cache_data(ttl=3600 * 24) # Cache the final merged dataset for 24 hours
def load_and_merge_all_data_directly(selected_food_items_lower, years_back):
    """
    Fetches food price, inflation, and rainfall data directly from external sources
    and merges them. This function is cached.
    """
    # Consolidated spinner for data loading with the fun message
    with st.spinner("Data ready for exploration! Dive in! 🎉"):
        df_food_prices = fetch_food_prices_from_api(API_URL, TARGET_FOOD_ITEMS, 'Nigeria', years_back)
        if df_food_prices.empty:
            st.error("Failed to load food price data. Please check API connectivity and data availability.")
            return None, None

        df_inflation = load_inflation_data_from_file(INFLATION_FILEPATH)
        if df_inflation.empty:
            st.error("Failed to load inflation data. Please ensure 'Inflation_Data_in_Excel.xlsx' exists and is valid.")
            return None, None

        df_rainfall = fetch_rainfall_data_from_api(state_coords, years_back)
        if df_rainfall.empty:
            # Create an empty DataFrame with expected columns to prevent merge errors
            df_rainfall = pd.DataFrame(columns=['state', 'Year', 'Month', 'rain'])

        df_merged = df_food_prices.merge(df_inflation, on=['Year', 'Month'], how='left')
        df_final_merged = df_merged.merge(
            df_rainfall,
            left_on=['State', 'Year', 'Month'],
            right_on=['state', 'Year', 'Month'],
            how='left'
        )
        if 'state' in df_final_merged.columns:
            df_final_merged.drop(columns=['state'], inplace=True)
        
        if df_final_merged.empty:
            st.error("Merged dataset is empty. Check individual data sources.")
            return None, None
        
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
    # These are illustrative hardcoded future CPI values.
    # In a real scenario, you'd fetch or estimate these from a reliable source.
    known_future_cpi_data = {
        '2025-06-01': 24, # Next month (Current time is May 2025)
        '2025-09-01': 24.5,
        '2025-12-01': 25,
        '2026-03-01': 25.5
    }
    known_future_cpi_series = pd.Series(known_future_cpi_data).astype(float)
    known_future_cpi_series.index = pd.to_datetime(known_future_cpi_series.index)

    future_cpi_filled = pd.Series(index=future_dates, dtype=float)
    for date, value in known_future_cpi_series.items():
        if date in future_cpi_filled.index:
            future_cpi_filled.loc[date] = value
    
    if not known_future_cpi_series.empty:
        last_known_cpi_value = known_future_cpi_series.iloc[-1]
    else:
        last_known_cpi_value = exog['foodYearOn'].iloc[-1] if 'foodYearOn' in exog.columns and not exog['foodYearOn'].empty else 0.0
    
    future_cpi_filled = future_cpi_filled.fillna(method='bfill').fillna(method='ffill').fillna(last_known_cpi_value)
    future_cpi = future_cpi_filled.values

    # --- Handling future Rainfall: Using multi-year monthly average (Pseudo-forecast) ---
    # This section implements the pseudo-forecasting logic for rainfall as requested.
    df_rainfall_history = df_full_merged[['State', 'Year', 'Month', 'rain']].drop_duplicates().rename(columns={'State': 'state'}).dropna(subset=['rain'])
    monthly_avg_rain = get_historical_average_rainfall(selected_state, df_rainfall_history)

    future_rain = np.full(steps, np.nan)

    if not monthly_avg_rain.empty:
        for i, date in enumerate(future_dates):
            month_match = monthly_avg_rain[monthly_avg_rain['Month'] == date.month]
            if not month_match.empty:
                future_rain[i] = month_match['avg_rain'].iloc[0]
            
        if np.isnan(future_rain).any():
            if 'rain' in exog.columns and not exog['rain'].empty:
                last_historical_rain = exog['rain'].iloc[-1]
                future_rain = np.nan_to_num(future_rain, nan=last_historical_rain)
            else:
                future_rain = np.nan_to_num(future_rain, nan=0.0)
    else:
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

    # Ensure all required columns exist before attempting to fillna
    for col in ['Price', 'foodYearOn', 'rain']:
        if col not in series_data.columns:
            series_data[col] = np.nan # Add column with NaN if missing

    series_data['Price'] = series_data['Price'].fillna(method='ffill').fillna(method='bfill')
    series_data['foodYearOn'] = series_data['foodYearOn'].fillna(method='ffill').fillna(method='bfill')
    series_data['rain'] = series_data['rain'].fillna(method='ffill').fillna(method='bfill')

    series_data.dropna(subset=['Price', 'foodYearOn', 'rain'], inplace=True)

    if series_data.empty:
        st.warning(f"No complete data available for {food_item} in {state_name} after handling missing values.")
        return pd.Series(dtype='float64'), pd.DataFrame(), pd.DataFrame()

    ts = series_data['Price']
    exog = series_data[['foodYearOn', 'rain']]
    
    return ts, exog, series_data

# --- SARIMAX Forecasting Function ---
@st.cache_resource(ttl=3600) # Cache the fitted model for 1 hour
def forecast_food_prices_sarimax(ts, exog, food_item, state_name, forecast_steps, df_full_merged):
    """
    Fits SARIMAX on full series with exogenous variables and forecasts
    the next forecast_steps months.
    Uses auto_arima to find the best SARIMAX order.
    """
    with st.spinner(f"Fetching data and training model for {food_item} in {state_name}... please wait"):
        try:
            model_auto = auto_arima(ts, exogenous=exog, seasonal=True, m=12, trace=False,
                                    suppress_warnings=True, error_action='ignore', stepwise=True,
                                    max_p=5, max_d=2, max_q=5, max_P=2, max_D=1, max_Q=2)
            order = model_auto.order
            seasonal_order = model_auto.seasonal_order
            st.info(f"Optimal SARIMAX order: {order}, Seasonal order: {seasonal_order}")

            model = SARIMAX(ts, exog=exog, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)

        except Exception as e:
            st.error(f"Error during SARIMAX model training: {e}")
            return pd.Series(dtype='float64')

        future_exog = extend_exog_for_forecast(exog, forecast_steps, state_name, df_full_merged)

        try:
            forecast = model_fit.forecast(steps=forecast_steps, exog=future_exog)
            return forecast
        except Exception as e:
            st.error(f"Error during SARIMAX forecast: {e}")
            return pd.Series(dtype='float64')

# --- Streamlit App Setup ---
st.sidebar.title("🧊 Filter Options")
# Assign unique keys to multiselect and slider in the sidebar
selected_food_items_explorer = st.sidebar.multiselect("Select Food Items:", CAPITALIZED_FOOD_ITEMS, default=['Maize'], key="explorer_food_select")
years_back_explorer = st.sidebar.slider("No. of years:", min_value=1, max_value=10, value=5, key="explorer_years_slider")

# Initialize session state variables using .get() for robustness
# No need to explicitly initialize if 'Load and Analyze Data' is the primary trigger
if 'df_full_merged' not in st.session_state:
    st.session_state.df_full_merged = None
if 'df_food_prices_for_explorer' not in st.session_state:
    st.session_state.df_food_prices_for_explorer = None

# Add the "Load and Analyze Data" button to the sidebar
with st.sidebar:
    if st.button("Load and Analyze Data", key="load_analyze_button"):
        # Pre-load all data by calling the direct fetching and merging function
        all_food_items_lower = [item.lower() for item in TARGET_FOOD_ITEMS]
        st.session_state.df_full_merged, st.session_state.df_food_prices_for_explorer = load_and_merge_all_data_directly(
            all_food_items_lower, years_back=10 # Load a good range for both explorer and predictor
        )
        # After loading, the app will rerun, and the data will be available.

# --- Main Page UI ---
st.title("🥦 Nigerian Food Price Data Explorer & Predictor")
st.markdown("""
Welcome to the interactive dashboard to explore food price trends across Nigerian states and predict future prices.
""")

# Always display tabs
tab1, tab2 = st.tabs(["📊 Data Explorer", "🧠 Predictor"])

# --- Data Explorer Tab ---
with tab1:
    st.markdown("Historical price data is pulled from the World Bank Monthly food price estimates API")
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
            nigeria_geojson = load_geojson()
            if nigeria_geojson:
                try:
                    available_map_items = food_data_explorer_filtered['Food_Item'].unique()
                    if selected_food_items_explorer and available_map_items.size > 0:
                        initial_map_item = next((item for item in selected_food_items_explorer if item in available_map_items), available_map_items[0])
                        map_item = st.selectbox("Select food item to map:", available_map_items, index=list(available_map_items).index(initial_map_item), key="map_food_select")
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

            # --- National Average Price Trends by Food Item (Previously "Price Trends Over Time") ---
            st.markdown("#### 📉 National Average Price Trends by Food Item")
            st.markdown("Shows average monthly trends across all states for each selected food item.")

            # Get unique food items from the filtered data
            unique_food_items_for_national_trend = sorted(food_data_explorer_filtered['Food_Item'].dropna().unique())

            if unique_food_items_for_national_trend:
                # Allow selecting a single food item for display in this chart
                selected_food_item_national_trend = st.selectbox(
                    "Select Food Item to View National Average Trend:", 
                    unique_food_items_for_national_trend, 
                    key="national_food_trend_select"
                )

                # Calculate national average for the selected food item
                df_national_avg_item = food_data_explorer_filtered[
                    food_data_explorer_filtered['Food_Item'] == selected_food_item_national_trend
                ].groupby('Date')['Price'].mean().reset_index()

                if not df_national_avg_item.empty:
                    fig_national_avg = px.line(
                        df_national_avg_item, x='Date', y='Price', 
                        title=f"National Average Price of {selected_food_item_national_trend} Over Time",
                        labels={"Price": "Price (₦ per 100 KG)", "Date": "Date"}
                    )
                    st.plotly_chart(fig_national_avg, use_container_width=True)
                else:
                    st.info(f"No national average data available for {selected_food_item_national_trend} for the selected period.")
            else:
                st.info("No food items available for national trend analysis.")


            # --- Food Price Trends Within a Selected State (Previously "Trend in Avg Price Across States for a Selected Food Item") ---
            st.markdown("#### 📌 Food Price Trends Within a Selected State")
            st.markdown("Compare the price movement of all selected food items within a specific state.")
            
            unique_states_for_multi_food_trend = sorted(food_data_explorer_filtered['State'].dropna().unique())
            if unique_states_for_multi_food_trend:
                selected_state_multi_food_trend = st.selectbox(
                    "Select State to Compare Food Items Within:", 
                    unique_states_for_multi_food_trend, 
                    key="multi_food_state_trend"
                )

                # Filter data for the selected state and selected food items (from sidebar)
                df_state_multi_food = food_data_explorer_filtered[
                    (food_data_explorer_filtered['State'] == selected_state_multi_food_trend) &
                    (food_data_explorer_filtered['Food_Item'].isin(selected_food_items_explorer))
                ].copy()

                if not df_state_multi_food.empty:
                    fig_multi_food_state = px.line(
                        df_state_multi_food, x='Date', y='Price', color='Food_Item',
                        title=f"Price Trends of Selected Food Items in {selected_state_multi_food_trend}",
                        labels={"Price": "Price (₦ per 100 KG)", "Date": "Date", "Food_Item": "Food Item"}
                    )
                    fig_multi_food_state.update_layout(legend_title_text="Food Item")
                    st.plotly_chart(fig_multi_food_state, use_container_width=True)
                else:
                    st.info(f"No data for selected food items in {selected_state_multi_food_trend} for the selected period.")
            else:
                st.info("No states available for multi-food trend comparison after data fetch.")

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
                data=food_data_explorer_filtered.to_csv(index)
