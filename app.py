import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import json
import joblib # for loading models
import numpy as np
import warnings
import os
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Nigerian Food Price Dashboard")

# Suppress specific warnings from statsmodels and pandas
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Global Configurations / Data Sources ---
API_URL = "https://microdata.worldbank.org/index.php/api/tables/data/fcv/wld_2021_rtfp_v02_m"
TARGET_FOOD_ITEMS = ['gari', 'groundnuts', 'maize', 'millet', 'sorghum', "cassava_meal"]
CAPITALIZED_FOOD_ITEMS = [item.capitalize() for item in TARGET_FOOD_ITEMS]
INFLATION_FILEPATH = 'Inflation_Data_in_Excel.xlsx' # Ensure this file is in your app's directory
BASE_MODEL_DIR = "saved_models" # Directory where pre-trained models are stored

# Coordinates for rainfall API calls
state_coords = {
    "Kaduna": (10.609319, 7.429504), "Kebbi": (11.6600, 4.0900), "Katsina": (12.9881, 7.6000),
    "Borno": (11.8333, 13.1500), "Kano": (12.0000, 8.5167), "Abia": (5.5000, 7.5000),
    "Adamawa": (9.3265, 12.3984), "Zamfara": (12.1833, 6.2333), "Lagos": (6.5244, 3.3792),
    "Oyo": (7.8500, 3.9333), "Gombe": (10.2904, 11.1700), "Jigawa": (12.2280, 9.5616),
    "Yobe": (12.0000, 11.5000),
}

# --- Functions to Fetch Data from External Sources (APIs, Files) ---

@st.cache_data(ttl=3600 * 24) # Cache the result of this API call for 24 hours
def fetch_food_prices_from_api(api_url, food_items, country='Nigeria', years_back=10):
    limit = 10000
    offset = 0
    all_records = []
    expected_api_columns = ['country', 'adm1_name', 'year', 'month', 'DATES'] + [f'c_{item}' for item in food_items]
    fields_param = ','.join(expected_api_columns)
    while True:
        params = {'limit': limit, 'offset': offset, 'country': country, 'fields': fields_param}
        try:
            response = requests.get(api_url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            if 'data' in data:
                records = data['data']
                if not records: break
                all_records.extend(records)
                offset += limit
            else: break
        except requests.exceptions.RequestException as e: st.error(f"Failed to fetch data from World Bank API: {e}"); break
        except json.JSONDecodeError as e: st.error(f"Failed to decode JSON from API response for food prices: {e}"); break
    df = pd.DataFrame(all_records)
    if df.empty: return pd.DataFrame()
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
    actual_price_columns_in_df = [col for col in [f'c_{item}' for item in food_items] if col in df.columns]
    if not actual_price_columns_in_df: st.warning("No relevant food price columns found in the fetched World Bank data."); return pd.DataFrame()
    for col in actual_price_columns_in_df: df[col] = pd.to_numeric(df[col], errors='coerce')
    df_clean = df.dropna(subset=actual_price_columns_in_df, how='all')
    groupby_cols = ['country', 'adm1_name', 'year', 'month']
    if not all(col in df_clean.columns for col in groupby_cols): st.error(f"Essential grouping columns missing after food price data fetch/clean: {missing_cols}. Check API fields and data schema."); return pd.DataFrame()
    df_avg = df_clean.groupby(groupby_cols)[actual_price_columns_in_df].mean().reset_index()
    df_avg['unit'] = '100 KG'
    df_avg.rename(columns={col: col[2:].capitalize() for col in actual_price_columns_in_df}, inplace=True)
    df_avg.rename(columns={'year': 'Year', 'month': 'Month', 'unit': 'Unit'}, inplace=True)
    if 'country' in df_avg.columns: df_avg.drop('country', axis=1, inplace=True)
    id_vars_for_melt = ['adm1_name', 'Year', 'Month', 'Unit']
    if not all(col in df_avg.columns for col in id_vars_for_melt): st.error(f"Columns for melting (id_vars) missing in food prices: {missing_cols}. Check previous renames or API fields."); return pd.DataFrame()
    df_long = pd.melt(df_avg, id_vars=id_vars_for_melt, var_name='Food_Item', value_name='Price')
    df_long.rename(columns={'adm1_name': 'State'}, inplace=True)
    df_long = df_long[['State', 'Year', 'Month', 'Food_Item', 'Unit', 'Price']]
    df_long = df_long[df_long['State'] != 'Market Average']
    df_long.dropna(subset=['Price'], inplace=True)
    df_long.sort_values(by=['State', 'Year', 'Month', 'Food_Item'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)
    return df_long

@st.cache_data(ttl=3600 * 24) # Cache the result of this file load for 24 hours
def load_inflation_data_from_file(filepath):
    try:
        if not os.path.exists(filepath): st.error(f"Inflation data file not found at: {filepath}. Please ensure it's in the app's directory."); return pd.DataFrame()
        df_inflation = pd.read_excel(filepath)
        df_inflation = df_inflation[['tyear', 'tmonth', 'foodYearOn']]
        df_inflation.rename(columns={'tyear': 'Year', 'tmonth': 'Month'}, inplace=True)
        df_inflation.dropna(subset=['Year', 'Month', 'foodYearOn'], inplace=True)
        return df_inflation
    except Exception as e: st.error(f"Error loading inflation data from Excel: {e}"); return pd.DataFrame()

@st.cache_data(ttl=3600 * 24) # Cache the result of this API call for 24 hours
def fetch_rainfall_data_from_api(state_coords, years_back=10):
    today = datetime.now()
    start_year_rainfall = today.year - years_back
    start_date_str = f"{start_year_rainfall}-01-01"
    end_date_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    all_rainfall_data = []
    for state, coords in state_coords.items():
        lat, lon = coords
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={start_date_str}&end_date={end_date_str}&daily=rain_sum&timezone=Africa%2FLagos")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()
            if 'daily' in data and 'rain_sum' in data['daily']:
                df = pd.DataFrame({"date": data['daily']['time'], "rain": data['daily']['rain_sum'],})
                df['date'] = pd.to_datetime(df['date'])
                df['Year'] = df['date'].dt.year
                df['Month'] = df['date'].dt.month
                monthly_summary = (df.groupby(['Year', 'Month'])[['rain']].sum().reset_index())
                monthly_summary['state'] = state
                all_rainfall_data.append(monthly_summary)
        except requests.exceptions.RequestException as e: st.error(f"Failed to fetch rainfall data for {state}: {e}")
        except json.JSONDecodeError as e: st.error(f"Failed to decode JSON for rainfall data for {state}: {e}")
    if all_rainfall_data: return pd.concat(all_rainfall_data, ignore_index=True)
    else: return pd.DataFrame()

@st.cache_data(ttl=3600 * 24) # Cache for 24 hours
def load_geojson():
    try:
        filepath = "ngs.json"
        if not os.path.exists(filepath): st.error("GeoJSON file 'ngs.json' not found. Please ensure it's in the root directory."); return None
        with open(filepath, "r") as f: return json.load(f)
    except Exception as e: st.error(f"Error loading GeoJSON: {e}"); return None

def get_historical_average_rainfall(state, df_rainfall_history):
    state_rain = df_rainfall_history[df_rainfall_history['state'] == state]
    if not state_rain.empty: return state_rain.groupby('Month')['rain'].mean().reset_index(name='avg_rain')
    return pd.DataFrame()

@st.cache_data(ttl=3600 * 24) # Cache the final merged dataset for 24 hours
def load_and_merge_all_data_directly(target_food_items_lower, years_back):
    with st.spinner("Loading and preparing data... this might take a moment. 🎉"):
        df_food_prices = fetch_food_prices_from_api(API_URL, target_food_items_lower, 'Nigeria', years_back)
        if df_food_prices.empty: st.error("Failed to load food price data. Please check API connectivity and data availability."); return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_inflation = load_inflation_data_from_file(INFLATION_FILEPATH)
        if df_inflation.empty: st.error("Failed to load inflation data. Please ensure 'Inflation_Data_in_Excel.xlsx' exists and is valid."); return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_rainfall = fetch_rainfall_data_from_api(state_coords, years_back)
        if df_rainfall.empty:
            df_rainfall = pd.DataFrame(columns=['state', 'Year', 'Month', 'rain'])
            st.warning("Failed to load rainfall data. Rainfall data might be missing in predictions.")

        df_merged = df_food_prices.merge(df_inflation, on=['Year', 'Month'], how='left')
        df_final_merged = df_merged.merge(df_rainfall, left_on=['State', 'Year', 'Month'], right_on=['state', 'Year', 'Month'], how='left')
        if 'state' in df_final_merged.columns: df_final_merged.drop(columns=['state'], inplace=True)
        df_final_merged['Date'] = pd.to_datetime(df_final_merged['Year'].astype(str) + '-' + df_final_merged['Month'].astype(str) + '-01')

        if df_final_merged.empty: st.error("Merged dataset is empty. Check individual data sources."); return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_rainfall_history_for_forecast = df_rainfall[['state', 'Year', 'Month', 'rain']].drop_duplicates().dropna(subset=['rain'])
    return df_final_merged, df_food_prices, df_rainfall_history_for_forecast

def extend_exog_for_forecast(exog, steps, selected_state, df_rainfall_history_for_forecast):
    if not isinstance(exog.index, pd.DatetimeIndex): raise ValueError("Exogenous variable index must be a DatetimeIndex.")
    future_dates = pd.date_range(start=exog.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')

    current_year = datetime.now().year
    known_future_cpi_data = {
        f'{current_year}-06-01': 24.0, f'{current_year}-09-01': 24.5, f'{current_year}-12-01': 25.0,
        f'{current_year+1}-03-01': 25.5, f'{current_year+1}-06-01': 26.0, f'{current_year+1}-09-01': 26.5,
        f'{current_year+1}-12-01': 27.0,
    }
    known_future_cpi_series = pd.Series(known_future_cpi_data).astype(float)
    known_future_cpi_series.index = pd.to_datetime(known_future_cpi_series.index)
    future_cpi_filled = pd.Series(index=future_dates, dtype=float)

    for date, value in known_future_cpi_series.items():
        if date in future_cpi_filled.index: future_cpi_filled.loc[date] = value
            
    if not known_future_cpi_series.empty:
        combined_index = future_dates.union(known_future_cpi_series.index)
        temp_cpi_series = known_future_cpi_series.reindex(combined_index)
        temp_cpi_series = temp_cpi_series.fillna(method='ffill').fillna(method='bfill')
        future_cpi_filled = temp_cpi_series.reindex(future_dates)
            
    if future_cpi_filled.isnull().any():
        last_known_cpi_value = exog['foodYearOn'].iloc[-1] if 'foodYearOn' in exog.columns and not exog['foodYearOn'].empty else 0.0
        future_cpi_filled = future_cpi_filled.fillna(last_known_cpi_value)
    future_cpi = future_cpi_filled.values

    monthly_avg_rain = get_historical_average_rainfall(selected_state, df_rainfall_history_for_forecast)
    future_rain = np.full(steps, np.nan)
    if not monthly_avg_rain.empty:
        for i, date in enumerate(future_dates):
            month_match = monthly_avg_rain[monthly_avg_rain['Month'] == date.month]
            if not month_match.empty: future_rain[i] = month_match['avg_rain'].iloc[0]
            
        if np.isnan(future_rain).any():
            if 'rain' in exog.columns and not exog['rain'].empty:
                last_historical_rain = exog['rain'].iloc[-1]
                future_rain = np.nan_to_num(future_rain, nan=last_historical_rain)
            else: future_rain = np.nan_to_num(future_rain, nan=0.0)
    else:
        last_historical_rain_fallback = exog['rain'].iloc[-1] if 'rain' in exog.columns and not exog['rain'].empty else 0.0
        future_rain = np.full(steps, last_historical_rain_fallback)

    future_exog = pd.DataFrame({'foodYearOn': future_cpi, 'rain': future_rain}, index=future_dates)
    future_exog = future_exog[exog.columns]
    return future_exog

# --- Prepare time series and exogenous variables for forecasting ---
def prepare_time_series_with_exog(df, state_name, food_item):
    series_data = df[(df['State'] == state_name) & (df['Food_Item'] == food_item)].copy()
    if series_data.empty: return pd.Series(dtype='float64'), pd.DataFrame(), pd.DataFrame()
    series_data.set_index('Date', inplace=True)
    series_data.sort_index(inplace=True)
    for col in ['Price', 'foodYearOn', 'rain']:
        if col not in series_data.columns: series_data[col] = np.nan
    series_data['Price'] = series_data['Price'].fillna(method='ffill').fillna(method='bfill')
    series_data['foodYearOn'] = series_data['foodYearOn'].fillna(method='ffill').fillna(method='bfill')
    series_data['rain'] = series_data['rain'].fillna(method='ffill').fillna(method='bfill')
    series_data.dropna(subset=['Price', 'foodYearOn', 'rain'], inplace=True)
    if series_data.empty: st.warning(f"No complete data available for {food_item} in {state_name} after handling missing values."); return pd.Series(dtype='float64'), pd.DataFrame(), pd.DataFrame()
    ts = series_data['Price']
    exog = series_data[['foodYearOn', 'rain']]
    return ts, exog, series_data

# --- ARIMA Forecasting Function (Loads pre-trained model) ---
@st.cache_resource(ttl=3600) # Cache the loaded model for 1 hour
def load_and_forecast_arima_model(food_item_lower, state_name, ts_tail_hash, exog_tail_hash, forecast_steps, future_exog_hash):
    """
    Loads a pre-trained ARIMA model and generates a forecast.
    Uses hashes of recent data to ensure caching works effectively for model loading and prediction.
    """
    # Reconstruct ts and exog from st.session_state if needed (or pass directly if small)
    ts = st.session_state[f'ts_{food_item_lower}_{state_name}_current']
    exog = st.session_state[f'exog_{food_item_lower}_{state_name}_current']
    future_exog = st.session_state[f'future_exog_{food_item_lower}_{state_name}_current']

    model_path = os.path.join(BASE_MODEL_DIR, food_item_lower.replace(" ", "_"), state_name.replace(" ", "_"), "model.pkl")

    if not os.path.exists(model_path):
        st.error(f"Pre-trained model not found for {food_item_lower.capitalize()} in {state_name}. Please run the training script first: {model_path}")
        return pd.Series(dtype='float64'), None, None # Return empty series and None for conf_int

    with st.spinner(f"Loading and generating forecast for {food_item_lower.capitalize()} in {state_name}..."):
        try:
            model = joblib.load(model_path)
            
            # The model was trained on log_series, so prediction will also be on log scale.
            # We need to supply the last known log_series point for prediction start, and future_exog.
            # model.predict() takes `n_periods` and `X` (exog for forecast)
            # The model automatically determines where to start prediction based on the last observed point
            # it was fitted on, and then uses `n_periods` and `X` for the forecast.
            
            # Since the model was fitted on log(ts), the prediction will be on log scale.
            forecast_log, conf_int_log = model.predict(
                n_periods=forecast_steps,
                X=future_exog, # Exogenous variables for the forecast horizon
                return_conf_int=True
            )
            
            # Convert back from log scale
            forecast = np.exp(forecast_log)
            conf_int_exp = np.exp(conf_int_log)

            # Create forecast index for plotting
            last_date = ts.index[-1]
            forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
            
            forecast_series = pd.Series(forecast, index=forecast_index)
            conf_int_df = pd.DataFrame(conf_int_exp, index=forecast_index, columns=['lower', 'upper'])
            
            return forecast_series, conf_int_df

        except Exception as e:
            st.error(f"Error loading or predicting with ARIMA model for {food_item_lower.capitalize()} in {state_name}: {e}")
            return pd.Series(dtype='float64'), None, None

# --- Streamlit App Setup ---
st.sidebar.title("🧊 Filter Options")
selected_food_items_explorer = st.sidebar.multiselect("Select Food Items:", CAPITALIZED_FOOD_ITEMS, default=['Maize'], key="explorer_food_select")
years_back_explorer = st.sidebar.slider("No. of years:", min_value=1, max_value=10, value=5, key="explorer_years_slider")

if 'df_full_merged' not in st.session_state: st.session_state.df_full_merged = pd.DataFrame()
if 'df_food_prices_raw' not in st.session_state: st.session_state.df_food_prices_raw = pd.DataFrame()
if 'df_rainfall_history_for_forecast' not in st.session_state: st.session_state.df_rainfall_history_for_forecast = pd.DataFrame()
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False

with st.sidebar:
    if st.button("Load All Data", key="load_analyze_button") or not st.session_state.data_loaded:
        all_food_items_lower = [item.lower() for item in TARGET_FOOD_ITEMS]
        st.session_state.df_full_merged, st.session_state.df_food_prices_raw, st.session_state.df_rainfall_history_for_forecast = load_and_merge_all_data_directly(
            all_food_items_lower, years_back=10
        )
        if not st.session_state.df_full_merged.empty:
            st.session_state.data_loaded = True
            st.success("Data loaded successfully! You can now explore and predict.")
        else:
            st.error("Failed to load data. Please check your internet connection or file paths.")

st.title("🥦 Nigerian Food Price Data Explorer & Predictor")
st.markdown("""
Welcome to the interactive dashboard to explore food price trends across Nigerian states and predict future prices.
""")

tab1, tab2 = st.tabs(["📊 Data Explorer", "🧠 Predictor"])

with tab1:
    st.markdown("Historical price data is pulled from the World Bank Monthly food price estimates API")
    st.markdown("This tab lets you analyze food price trends, map data, and download cleaned datasets.")
    if st.session_state.data_loaded and not st.session_state.df_food_prices_raw.empty:
        food_data_explorer_filtered = st.session_state.df_food_prices_raw[
            (st.session_state.df_food_prices_raw['Food_Item'].isin(selected_food_items_explorer)) &
            (st.session_state.df_food_prices_raw['Year'] >= (datetime.now().year - years_back_explorer))
        ].copy()
        food_data_explorer_filtered['Date'] = pd.to_datetime(food_data_explorer_filtered['Year'].astype(str) + '-' + food_data_explorer_filtered['Month'].astype(str) + '-01')

        if food_data_explorer_filtered.empty:
            st.info("No data available for the selected food items and years in the explorer. Try adjusting filters.")
        else:
            st.markdown("#### 📊 Data Quality Check")
            st.markdown("This section helps you assess the completeness and reliability of the dataset.")
            total = len(food_data_explorer_filtered)
            missing_price = food_data_explorer_filtered['Price'].isna().sum()
            zero_price = (food_data_explorer_filtered['Price'] == 0).sum()
            st.info(f"Missing prices: {missing_price} | Zero prices: {zero_price} | Total entries: {total}")

            st.markdown("#### 🗺️ Choropleth Map")
            st.markdown("Visualize average food prices by state using a color-coded map.")
            nigeria_geojson = load_geojson()
            if nigeria_geojson:
                try:
                    available_map_items = food_data_explorer_filtered['Food_Item'].unique()
                    if selected_food_items_explorer: # Only show the map if a food item is selected
                        selected_food_for_map = st.selectbox(
                            "Select Food Item for Map:",
                            available_map_items,
                            index=0 if available_map_items.size > 0 else None
                        )
                        if selected_food_for_map:
                            df_map_data = food_data_explorer_filtered[food_data_explorer_filtered['Food_Item'] == selected_food_for_map].groupby('State')['Price'].mean().reset_index()
                            fig_map = px.choropleth_mapbox(
                                df_map_data,
                                geojson=nigeria_geojson,
                                locations='State',
                                featureidkey="properties.NAME_1",
                                color='Price',
                                color_continuous_scale="Viridis",
                                mapbox_style="carto-positron",
                                zoom=5, center={"lat": 9.0820, "lon": 8.6753},
                                opacity=0.7,
                                hover_name='State',
                                hover_data={'Price': True},
                                title=f'Average Price of {selected_food_for_map} by State'
                            )
                            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                            st.plotly_chart(fig_map, use_container_width=True)
                        else:
                            st.info("No food item selected for map visualization.")
                    else:
                        st.info("Please select at least one food item in the sidebar to view the map.")
                except Exception as e:
                    st.error(f"Error generating choropleth map: {e}")
            else:
                st.warning("Cannot display map: GeoJSON data not loaded.")

            # Time Series Line Chart
            st.markdown("#### 📈 Price Trends Over Time")
            st.markdown("Observe how prices of selected food items have changed over months/years.")
            fig_line = px.line(
                food_data_explorer_filtered,
                x='Date',
                y='Price',
                color='Food_Item',
                line_dash='State', # Use line_dash to distinguish states for the same food item
                title='Food Prices Over Time (Naira)',
                labels={'Price': 'Price (Naira)', 'Date': 'Date'},
                hover_data={'State': True, 'Food_Item': True, 'Price': ':.2f'}
            )
            fig_line.update_layout(hovermode="x unified")
            st.plotly_chart(fig_line, use_container_width=True)

            # Raw Data Display and Download
            st.markdown("#### ⬇️ Raw Data & Download")
            st.markdown("View the raw data used for analysis and download it.")
            st.dataframe(food_data_explorer_filtered)

            csv_data = food_data_explorer_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv_data,
                file_name="nigerian_food_prices_explorer_data.csv",
                mime="text/csv",
            )
    elif st.session_state.data_loaded and st.session_state.df_food_prices_raw.empty:
        st.info("Food price data is empty. Please check the API source or filters.")
    else:
        st.info("Please click 'Load All Data' in the sidebar to begin exploring.")

# --- Predictor Tab ---
with tab2:
    st.markdown("This tab allows you to predict future food prices using pre-trained ARIMA models.")
    if st.session_state.data_loaded and not st.session_state.df_full_merged.empty:
        # Get unique states and food items from the loaded data for prediction
        available_states = sorted(st.session_state.df_full_merged['State'].unique().tolist())
        available_food_items = sorted(st.session_state.df_full_merged['Food_Item'].unique().tolist())

        col1_pred, col2_pred, col3_pred = st.columns([1, 1, 1])

        with col1_pred:
            selected_state_pred = st.selectbox("Select State:", available_states, key="predictor_state_select")
        with col2_pred:
            selected_food_item_pred = st.selectbox("Select Food Item:", available_food_items, key="predictor_food_select")
        with col3_pred:
            forecast_months = st.slider("Forecast Months:", min_value=1, max_value=24, value=6, key="forecast_months_slider")

        if st.button("Generate Forecast", key="generate_forecast_button"):
            food_item_lower = selected_food_item_pred.lower()
            
            # Prepare time series and exogenous variables from the full merged data
            ts, exog, series_data_for_pred = prepare_time_series_with_exog(
                st.session_state.df_full_merged, selected_state_pred, selected_food_item_pred
            )
            
            if ts.empty or exog.empty:
                st.warning(f"Not enough data to generate forecast for {selected_food_item_pred} in {selected_state_pred}.")
            else:
                # Store ts, exog, and future_exog in session state for caching
                st.session_state[f'ts_{food_item_lower}_{selected_state_pred}_current'] = ts
                st.session_state[f'exog_{food_item_lower}_{selected_state_pred}_current'] = exog

                # Extend exogenous variables for the forecast horizon
                future_exog = extend_exog_for_forecast(exog, forecast_months, selected_state_pred, st.session_state.df_rainfall_history_for_forecast)
                st.session_state[f'future_exog_{food_item_lower}_{selected_state_pred}_current'] = future_exog

                # Dummy hashes for cache key - actual objects are in session_state
                ts_hash = hash(ts.to_json()) if not ts.empty else 0
                exog_hash = hash(exog.to_json()) if not exog.empty else 0
                future_exog_hash = hash(future_exog.to_json()) if not future_exog.empty else 0

                # Load and forecast using the cached function
                forecast_prices, conf_int_df = load_and_forecast_arima_model(
                    food_item_lower, selected_state_pred, ts_hash, exog_hash, forecast_months, future_exog_hash
                )

                if not forecast_prices.empty:
                    st.markdown(f"#### 🔮 Forecast for {selected_food_item_pred} in {selected_state_pred}")

                    # Plotting the forecast
                    # Combine historical and forecasted data for plotting
                    # Ensure historical data is also in log scale for consistency if the model was trained on log
                    
                    # For plotting, use original prices, so if model predicts log, convert back
                    # The `prepare_time_series_with_exog` provides original prices.
                    historical_prices = ts
                    
                    fig = px.line()
                    fig.add_scatter(x=historical_prices.index, y=historical_prices.values, mode='lines', name='Historical Prices')
                    fig.add_scatter(x=forecast_prices.index, y=forecast_prices.values, mode='lines', name='Forecasted Prices', line=dict(color='red'))
                    
                    if conf_int_df is not None and not conf_int_df.empty:
                        fig.add_trace(px.scatter(x=conf_int_df.index, y=conf_int_df['lower']).data[0].update(
                            mode='lines', line=dict(width=0), showlegend=False
                        ))
                        fig.add_trace(px.scatter(x=conf_int_df.index, y=conf_int_df['upper']).data[0].update(
                            mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='Confidence Interval'
                        ))

                    fig.update_layout(
                        title=f'Food Price Forecast for {selected_food_item_pred} in {selected_state_pred}',
                        xaxis_title='Date',
                        yaxis_title='Price (Naira)',
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### 📈 Forecast Details")
                    forecast_df = pd.DataFrame({
                        'Date': forecast_prices.index.strftime('%Y-%m'),
                        'Forecasted Price (Naira)': forecast_prices.values.round(2)
                    })
                    if conf_int_df is not None:
                        forecast_df['Lower Bound (Naira)'] = conf_int_df['lower'].values.round(2)
                        forecast_df['Upper Bound (Naira)'] = conf_int_df['upper'].values.round(2)
                    st.dataframe(forecast_df)

                    csv_forecast_data = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Forecast as CSV",
                        data=csv_forecast_data,
                        file_name=f"{selected_food_item_pred}_{selected_state_pred}_forecast.csv",
                        mime="text/csv",
                    )
                else:
                    st.info(f"Could not generate a forecast for {selected_food_item_pred} in {selected_state_pred}. Check data availability and model training status.")
    else:
        st.info("Please click 'Load All Data' in the sidebar to enable the predictor tab.")
