import pandas as pd
import requests
import plotly.express as px
import json
import numpy as np
import warnings
import os
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

st.set_page_config(layout="wide", page_title="Nigerian Food Price Dashboard")

# Suppress specific warnings from statsmodels and pandas
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Global Configurations / Data Sources ---
API_URL = "https://microdata.worldbank.org/index.php/api/tables/data/fcv/wld_2021_rtfp_v02_m"
TARGET_FOOD_ITEMS = ['gari', 'groundnuts', 'maize', 'millet', 'sorghum', "cassava_meal"]
CAPITALIZED_FOOD_ITEMS = [item.capitalize() for item in TARGET_FOOD_ITEMS]
BEST_MODEL_DIR = 'best_food_price_models' # Directory where .kera models are saved

# --- Functions to Fetch Data from External Sources (APIs, Files) ---

@st.cache_data(ttl=3600 * 24)
def fetch_food_prices_from_api(api_url, food_items, country='Nigeria', years_back=10):
    """Fetches food price data from the World Bank API and returns a DataFrame."""
    limit = 10000
    offset = 0
    all_records = []

    # Ensure 'DATES' is explicitly requested if it's the primary date column
    expected_api_columns = ['country', 'adm1_name', 'year', 'month', 'DATES'] + [f'c_{item}' for item in food_items]
    fields_param = ','.join(expected_api_columns)

    st.info("Attempting to fetch data from World Bank API...")
    while True:
        params = {
            'limit': limit,
            'offset': offset,
            'country': country,
            'fields': fields_param
        }
        try:
            response = requests.get(api_url, params=params, timeout=60)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            if 'data' in data:
                records = data['data']
                if not records:
                    break # No more records to fetch
                all_records.extend(records)
                offset += limit
            else:
                st.warning("API response does not contain 'data' key or is empty. Stopping fetch.")
                break
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please check your internet connection or try again later.")
            break
        except requests.exceptions.ConnectionError:
            st.error("Network connection error. Please check your internet connection.")
            break
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data from World Bank API: {e}")
            break
        except json.JSONDecodeError as e:
            st.error(f"Failed to decode JSON from API response for food prices: {e}")
            break

    df = pd.DataFrame(all_records)
    if df.empty:
        st.warning("No data fetched from the API.")
        return pd.DataFrame()

    # Convert 'year' and 'month' to numeric, coerce errors to NaN
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['month'] = pd.to_numeric(df['month'], errors='coerce')

    # Handle 'DATES' column or create from 'year' and 'month'
    if 'DATES' in df.columns:
        df['DATES'] = pd.to_datetime(df['DATES'], errors='coerce')
        df.dropna(subset=['DATES', 'year', 'month'], inplace=True)
    else:
        # Create 'DATES' from 'year' and 'month' if it's missing
        df['DATES'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01', errors='coerce')
        df.dropna(subset=['DATES', 'year', 'month'], inplace=True)
        st.info("Created 'DATES' column from 'year' and 'month'.")

    # Filter by years_back
    current_year = datetime.now().year
    start_year = current_year - years_back
    df = df[df['year'] >= start_year].copy()

    # Identify and convert price columns
    actual_price_columns_in_df = [col for col in [f'c_{item}' for item in food_items] if col in df.columns]
    if not actual_price_columns_in_df:
        st.warning("No relevant food price columns found in the fetched World Bank data. Please check `TARGET_FOOD_ITEMS`.")
        return pd.DataFrame()

    for col in actual_price_columns_in_df:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # FIX: Use 'actual_price_columns_in_df' instead of 'actual_price_columns_in_all_records'
    df_clean = df.dropna(subset=actual_price_columns_in_df, how='all')

    groupby_cols = ['country', 'adm1_name', 'year', 'month']
    if not all(col in df_clean.columns for col in groupby_cols):
        missing_cols = [col for col in groupby_cols if col not in df_clean.columns]
        st.error(f"Essential grouping columns missing after food price data fetch/clean: {missing_cols}. Check API fields and data schema.")
        return pd.DataFrame()

    df_avg = df_clean.groupby(groupby_cols)[actual_price_columns_in_df].mean().reset_index()
    df_avg['unit'] = '100 KG' # Assuming unit is consistent

    # Create a mapping for renaming, ensuring no clash with 'Price'
    rename_mapping = {col: col[2:].capitalize() for col in actual_price_columns_in_df}
    df_avg.rename(columns=rename_mapping, inplace=True)
    df_avg.rename(columns={'year': 'Year', 'month': 'Month', 'unit': 'Unit'}, inplace=True)

    if 'country' in df_avg.columns:
        df_avg.drop('country', axis=1, inplace=True)

    id_vars_for_melt = ['adm1_name', 'Year', 'Month', 'Unit']
    if not all(col in df_avg.columns for col in id_vars_for_melt):
        missing_cols = [col for col in id_vars_for_melt if col not in df_avg.columns]
        st.error(f"Columns for melting (id_vars) missing in food prices: {missing_cols}. Check previous renames or API fields.")
        return pd.DataFrame()

    # FIX: Explicitly define value_vars for pd.melt to avoid conflicts
    # The columns that contain the actual price values, which have now been renamed
    value_vars_for_melt = [rename_mapping[col] for col in actual_price_columns_in_df]

    df_long = pd.melt(
        df_avg,
        id_vars=id_vars_for_melt,
        value_vars=value_vars_for_melt, # Specify which columns to unpivot
        var_name='Food_Item',
        value_name='Price'
    )
    df_long.rename(columns={'adm1_name': 'State'}, inplace=True)
    df_long = df_long[['State', 'Year', 'Month', 'Food_Item', 'Unit', 'Price']]
    df_long = df_long[df_long['State'] != 'Market Average'] # Exclude 'Market Average' if it's not a state
    df_long.dropna(subset=['Price'], inplace=True)
    df_long.sort_values(by=['State', 'Year', 'Month', 'Food_Item'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)
    
    return df_long

@st.cache_data(ttl=3600 * 24)
def load_geojson():
    """Loads the GeoJSON file for Nigeria states."""
    try:
        filepath = "ngs.json"
        if not os.path.exists(filepath):
            st.error("GeoJSON file 'ngs.json' not found. Please ensure it's in the root directory.")
            return None

        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return None

@st.cache_data(ttl=3600 * 24)
def load_and_prepare_data_for_app(food_items_list_lower, years_back=10):
    """
    Orchestrates fetching food price data and prepares it for both explorer and predictor.
    Returns df_food_prices_raw (for explorer) and df_national_avg_prices (for predictor).
    """
    st.info("Loading food price data. This may take a moment...")
    
    df_food_prices = fetch_food_prices_from_api(API_URL, food_items_list_lower, years_back=years_back)

    if df_food_prices.empty:
        st.error("Failed to load food price data.")
        return pd.DataFrame(), pd.DataFrame()

    df_food_prices['Date'] = pd.to_datetime(df_food_prices['Year'].astype(str) + '-' + df_food_prices['Month'].astype(str) + '-01')
    df_food_prices.sort_values(by=['State', 'Food_Item', 'Date'], inplace=True)
    df_food_prices.reset_index(drop=True, inplace=True)

    return df_food_prices, df_food_prices # Returning the same DataFrame for now, can be split later if needed

# --- Prepare time series for LSTM (univariate national average) ---
def prepare_national_average_time_series(df, food_item):
    """
    Prepares the national average time series (Price) for a given food item for univariate LSTM.
    Returns:
    - ts_scaled: Pandas Series of historical national average prices, scaled and indexed by Date.
    - scaler_price: MinMaxScaler fitted on the price data.
    """
    df_filtered_item = df[df['Food_Item'] == food_item].copy()
    
    if df_filtered_item.empty:
        st.warning(f"No data available for {food_item} to calculate national average.")
        return pd.Series(dtype='float64'), None

    # Calculate national average and set frequency
    series = df_filtered_item.groupby('Date').Price.mean().asfreq('MS')

    # Interpolate missing values using 'ffill' and then 'bfill' to handle leading NaNs
    series = series.fillna(method='ffill').fillna(method='bfill')
    series = series.clip(lower=0.01)       # Avoid zeros or negatives for scaling

    if series.empty or len(series) < 2:
        st.warning(f"Insufficient national average data for {food_item} to prepare time series for LSTM. Found {len(series)} data points.")
        return pd.Series(dtype='float64'), None

    scaler_price = MinMaxScaler(feature_range=(0, 1))
    ts_scaled = pd.Series(scaler_price.fit_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)
    
    st.session_state[f'scaler_price_{food_item}'] = scaler_price

    return ts_scaled, scaler_price

@st.cache_resource(hash_funcs={tf.keras.Model: lambda _: None})
def load_lstm_model(model_filepath):
    """Loads a pre-trained LSTM model from a .kera file."""
    try:
        model = load_model(model_filepath)
        return model
    except Exception as e:
        st.error(f"Error loading LSTM model from {model_filepath}: {e}")
        return None

def forecast_national_average_food_prices_lstm(ts_scaled, food_item, forecast_steps):
    """
    Loads the appropriate LSTM model and generates univariate forecasts for national average prices.
    Expects scaled time series.
    """
    model_filename = f"{food_item.capitalize()}_LSTM_model.keras"
    model_filepath = os.path.join(BEST_MODEL_DIR, model_filename)

    if not os.path.exists(model_filepath):
        st.warning(f"LSTM model for national average {food_item} not found at {model_filepath}. Cannot generate forecast.")
        return pd.Series(dtype='float64'), None

    model = load_lstm_model(model_filepath)
    if model is None:
        return pd.Series(dtype='float64'), None
    
    st.info(f"Using LSTM model: {model_filename}")
    
    scaler_price = st.session_state.get(f'scaler_price_{food_item}')
    if scaler_price is None:
        st.error("Price scaler not found in session state. Cannot inverse transform forecast. Please ensure data is loaded and prepared first.")
        return pd.Series(dtype='float64'), None

    # Determine sequence_length from the loaded model's input shape
    # LSTM input shape: (batch_size, sequence_length, num_features)
    sequence_length = 1 # Default
    if model.input_shape and len(model.input_shape) >= 2:
        sequence_length = model.input_shape[1]
    
    if ts_scaled.empty or len(ts_scaled) < sequence_length:
        st.warning(f"Insufficient historical data ({len(ts_scaled)} points) for national average {food_item} to generate an LSTM forecast with model's sequence length {sequence_length}. At least {sequence_length} points are required.")
        return pd.Series(dtype='float64'), None

    # Use the last `sequence_length` data points for the initial prediction
    current_sequence = ts_scaled.iloc[-sequence_length:].values

    forecast_values = []
    last_date = ts_scaled.index[-1]

    for i in range(forecast_steps):
        # Reshape for LSTM input: (1, sequence_length, 1)
        input_sequence = current_sequence.reshape(1, sequence_length, 1)
        
        next_scaled_price = model.predict(input_sequence, verbose=0)[0][0]
        forecast_values.append(next_scaled_price)

        # Update the sequence for the next prediction
        current_sequence = np.append(current_sequence[1:], next_scaled_price)

    forecast_index = [last_date + pd.DateOffset(months=i) for i in range(1, forecast_steps + 1)]
    forecast_series_scaled = pd.Series(forecast_values, index=forecast_index)
    forecast_series_unscaled = pd.Series(scaler_price.inverse_transform(forecast_series_scaled.values.reshape(-1, 1)).flatten(), index=forecast_series_scaled.index)

    return forecast_series_unscaled, model_filename

# --- Streamlit App Setup ---
st.sidebar.title("🧊 Filter Options")
selected_food_items_explorer = st.sidebar.multiselect("Select Food Items:", CAPITALIZED_FOOD_ITEMS, default=['Maize'], key="explorer_food_select")
years_back_explorer = st.sidebar.slider("No. of years:", min_value=1, max_value=10, value=5, key="explorer_years_slider")

# Initialize session state variables
if 'df_full_merged' not in st.session_state:
    st.session_state.df_full_merged = pd.DataFrame()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

with st.sidebar:
    # Use a consistent key for the button to avoid re-rendering issues
    if st.button("Load All Data", key="load_data_button") or not st.session_state.data_loaded:
        all_food_items_lower = [item.lower() for item in TARGET_FOOD_ITEMS]
        # Only fetch data if not already loaded or explicitly requested
        if not st.session_state.data_loaded or st.button("Reload Data", key="reload_data_button_hidden"): # Hidden reload button for programmatic use
            st.session_state.df_full_merged, _ = load_and_prepare_data_for_app(
                all_food_items_lower, years_back=10
            )
            if not st.session_state.df_full_merged.empty:
                st.session_state.data_loaded = True
                st.success("Data loaded successfully! You can now explore and predict.")
            else:
                st.error("Failed to load data. Please check your internet connection or file paths.")
        else:
            st.info("Data already loaded. Click 'Reload Data' if you want to refetch.")

# --- Main Page UI ---
st.title("🥦 Nigerian Food Price Data Explorer & Predictor")
st.markdown("""
Welcome to the interactive dashboard to explore food price trends across Nigerian states and predict future national average prices.
""")

tab1, tab2 = st.tabs(["📊 Data Explorer", "🧠 Predictor"])

# --- Data Explorer Tab ---
with tab1:
    st.markdown("Historical price data is pulled from the World Bank Monthly food price estimates API")
    st.markdown("This tab lets you analyze food price trends, map data, and download cleaned datasets.")

    if st.session_state.data_loaded and not st.session_state.df_full_merged.empty:
        food_data_explorer_filtered = st.session_state.df_full_merged[
            (st.session_state.df_full_merged['Food_Item'].isin(selected_food_items_explorer)) &
            (st.session_state.df_full_merged['Year'] >= (datetime.now().year - years_back_explorer))
        ].copy()
        
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
                    if selected_food_items_explorer and available_map_items.size > 0:
                        # Ensure default selection for map is one of the available items
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

            st.markdown("#### 📉 National Average Price Trends by Food Item")
            st.markdown("Shows average monthly trends across all states for each selected food item.")

            unique_food_items_for_national_trend = sorted(food_data_explorer_filtered['Food_Item'].dropna().unique())

            if unique_food_items_for_national_trend:
                selected_food_item_national_trend = st.selectbox(
                    "Select Food Item to View National Average Trend:", 
                    unique_food_items_for_national_trend, 
                    key="national_food_trend_select"
                )

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

            st.markdown("#### 📌 Food Price Trends Within a Selected State")
            st.markdown("Compare the price movement of all selected food items within a specific state.")
            
            unique_states_for_multi_food_trend = sorted(food_data_explorer_filtered['State'].dropna().unique())

            if unique_states_for_multi_food_trend:
                selected_state_for_multi_food_trend = st.selectbox(
                    "Select State to View Food Item Trends:", 
                    unique_states_for_multi_food_trend, 
                    key="state_food_trend_select"
                )

                df_state_trends = food_data_explorer_filtered[
                    (food_data_explorer_filtered['State'] == selected_state_for_multi_food_trend)
                ]
                
                if not df_state_trends.empty:
                    fig_state_trends = px.line(
                        df_state_trends, x='Date', y='Price', color='Food_Item',
                        title=f"Food Price Trends in {selected_state_for_multi_food_trend}",
                        labels={"Price": "Price (₦ per 100 KG)", "Date": "Date", "Food_Item": "Food Item"}
                    )
                    st.plotly_chart(fig_state_trends, use_container_width=True)
                else:
                    st.info(f"No data available for food item trends in {selected_state_for_multi_food_trend} for the selected period.")
            else:
                st.info("No states available for food price trend analysis.")

            st.markdown("#### ⬇️ Raw Data Table and Download")
            st.markdown("View the filtered raw data and download it as a CSV.")
            
            st.dataframe(food_data_explorer_filtered)
            
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(food_data_explorer_filtered)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"nigerian_food_prices_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_explorer_data"
            )
    else:
        st.info("Please click 'Load All Data' in the sidebar to start exploring.")

# --- Predictor Tab ---
with tab2:
    st.markdown("This tab allows you to forecast future national average food prices using pre-trained LSTM models.")

    if st.session_state.data_loaded and not st.session_state.df_full_merged.empty:
        df_for_prediction_base = st.session_state.df_full_merged.copy()

        available_food_items = sorted(df_for_prediction_base['Food_Item'].unique())

        if not available_food_items:
            st.warning("No sufficient data available to make predictions. Please check the loaded data.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                selected_food_item_predictor = st.selectbox(
                    "Select Food Item for National Average Prediction:",
                    available_food_items,
                    key="predictor_food_item_select"
                )
            with col2:
                forecast_months = st.slider("Forecast for (months):", min_value=1, max_value=24, value=6, key="forecast_months_slider")

            if st.button("Generate National Average Forecast", key="generate_forecast_button"):
                if selected_food_item_predictor:
                    st.write(f"Generating national average forecast for **{selected_food_item_predictor}** for **{forecast_months}** months...")

                    # Prepare national average time series for scaling
                    ts_scaled, scaler_price = prepare_national_average_time_series(
                        df_for_prediction_base,
                        selected_food_item_predictor
                    )

                    if ts_scaled.empty or scaler_price is None:
                        st.error(f"Could not prepare sufficient national average data for {selected_food_item_predictor} for prediction.")
                    else:
                        # Call the national average forecast function
                        forecast_unscaled, model_used = forecast_national_average_food_prices_lstm(
                            ts_scaled,
                            selected_food_item_predictor,
                            forecast_months
                        )

                        if not forecast_unscaled.empty:
                            st.subheader("📊 Forecast Results")
                            st.write(f"National Average Forecast for {selected_food_item_predictor} (using {model_used}):")
                            st.dataframe(forecast_unscaled.to_frame(name='Predicted National Avg. Price (₦)'))

                            # Combine historical data and forecast for plotting
                            historical_data_national_avg = df_for_prediction_base[
                                df_for_prediction_base['Food_Item'] == selected_food_item_predictor
                            ].groupby('Date').Price.mean().asfreq('MS').fillna(method='ffill').clip(lower=0.01)

                            # Create DataFrames with a 'Data Type' column
                            df_historical = historical_data_national_avg.to_frame(name='Price')
                            df_historical['Data Type'] = 'Historical National Avg. Price'
                            df_historical.reset_index(inplace=True)

                            df_forecast = forecast_unscaled.to_frame(name='Price')
                            df_forecast['Data Type'] = 'Predicted National Avg. Price'
                            df_forecast.reset_index(inplace=True)
                            
                            # Concatenate the two DataFrames
                            combined_plot_data = pd.concat([df_historical, df_forecast])
                            combined_plot_data.rename(columns={'index': 'Date'}, inplace=True) # Rename 'index' if it exists from reset_index

                            fig_forecast = px.line(
                                combined_plot_data,
                                x='Date',
                                y='Price',
                                color='Data Type',
                                title=f"Historical and Predicted National Average Price of {selected_food_item_predictor}",
                                labels={'Price': 'Price (₦ per 100 KG)', 'Date': 'Date'}
                            )
                            fig_forecast.add_vline(x=historical_data_national_avg.index.max(), line_dash="dash", line_color="gray", annotation_text="Forecast Start")
                            st.plotly_chart(fig_forecast, use_container_width=True)
                        else:
                            st.warning("No forecast generated. Check data and model availability.")
                else:
                    st.warning("Please select a food item to generate a national average forecast.")
    else:
        st.info("Please click 'Load All Data' in the sidebar to enable the predictor.")
