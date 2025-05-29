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
        st.warning("No relevant food price columns found in the fetched World Bank data. Please check TARGET_FOOD_ITEMS.")
        return pd.DataFrame()

    for col in actual_price_columns_in_df:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Use 'actual_price_columns_in_df' for dropna subset
    df_clean = df.dropna(subset=actual_price_columns_in_df, how='all')

    groupby_cols = ['country', 'adm1_name', 'year', 'month']
    if not all(col in df_clean.columns for col in groupby_cols):
        missing_cols = [col for col in groupby_cols if col not in df_clean.columns]
        st.error(f"Essential grouping columns missing after food price data fetch/clean: {missing_cols}. Check API fields and data schema.")
        return pd.DataFrame()

    df_avg = df_clean.groupby(groupby_cols)[actual_price_columns_in_df].mean().reset_index()
    df_avg['unit'] = '100 KG' # Assuming unit is consistent

    # Create a mapping for renaming. This will be used to create the value_vars list for melt.
    rename_for_melt = {col: col[2:].capitalize() for col in actual_price_columns_in_df}
    
    # Rename the columns in df_avg BEFORE melting
    df_avg.rename(columns=rename_for_melt, inplace=True)
    df_avg.rename(columns={'year': 'Year', 'month': 'Month', 'unit': 'Unit'}, inplace=True)

    if 'country' in df_avg.columns:
        df_avg.drop('country', axis=1, inplace=True)

    id_vars_for_melt = ['adm1_name', 'Year', 'Month', 'Unit']
    if not all(col in df_avg.columns for col in id_vars_for_melt):
        missing_cols = [col for col in id_vars_for_melt if col not in df_avg.columns]
        st.error(f"Columns for melting (id_vars) missing in food prices: {missing_cols}. Check previous renames or API fields.")
        return pd.DataFrame()

    # CRITICAL FIX: Ensure 'value_vars' explicitly targets the *renamed* food item columns.
    # These are the columns that will be unpivoted into the 'Price' column.
    # Filter for columns that are present in df_avg AND are part of the renamed food items.
    value_vars_for_melt = [col for col in rename_for_melt.values() if col in df_avg.columns]

    if not value_vars_for_melt:
        st.error("No valid food price columns found to melt. Check the renaming logic.")
        return pd.DataFrame()

    df_long = pd.melt(
        df_avg,
        id_vars=id_vars_for_melt,
        value_vars=value_vars_for_melt, # Explicitly specify which columns to unpivot
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

    # For the predictor, we need national average prices
    df_national_avg_prices = df_food_prices.groupby(['Date', 'Food_Item'])['Price'].mean().reset_index()
    df_national_avg_prices.sort_values(by=['Food_Item', 'Date'], inplace=True)

    return df_food_prices, df_national_avg_prices

# --- Prediction Functions ---

@st.cache_resource
def load_prediction_model(food_item_capitalized):
    """Loads the pre-trained Keras model for the given food item."""
    model_name = f"{food_item_capitalized}_LSTM_model.keras"
    model_path = os.path.join(BEST_MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        st.error(f"Model file not found for {food_item_capitalized}: {model_path}. Please ensure models are trained and saved correctly.")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

def create_sequences(data, sequence_length):
    """Creates sequences for LSTM input."""
    xs = []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length), 0]
        xs.append(x)
    return np.array(xs)

def forecast_prices(model, scaler, historical_prices_scaled, num_months_to_forecast, sequence_length=12):
    """
    Forecasts future prices using the loaded LSTM model.
    """
    if historical_prices_scaled.shape[0] < sequence_length:
        st.warning(f"Not enough historical data ({historical_prices_scaled.shape[0]} months) to create a sequence of length {sequence_length}. Cannot forecast.")
        return np.array([]), np.array([])

    current_sequence = historical_prices_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    
    predicted_prices_scaled = []
    for _ in range(num_months_to_forecast):
        next_price_scaled = model.predict(current_sequence, verbose=0)[0, 0]
        predicted_prices_scaled.append(next_price_scaled)
        # Update the sequence: remove the first element, add the predicted element
        current_sequence = np.append(current_sequence[:, 1:, :], [[next_price_scaled]], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))
    return predicted_prices.flatten()


# --- Streamlit App Setup ---
st.sidebar.title("🧊 Filter Options")
selected_food_items_explorer = st.sidebar.multiselect("Select Food Items:", CAPITALIZED_FOOD_ITEMS, default=['Maize'], key="explorer_food_select")
years_back_explorer = st.sidebar.slider("No. of years:", min_value=1, max_value=10, value=5, key="explorer_years_slider")

# Initialize session state variables
if 'df_full_merged' not in st.session_state:
    st.session_state.df_full_merged = pd.DataFrame()
if 'df_national_avg' not in st.session_state:
    st.session_state.df_national_avg = pd.DataFrame()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Automatically load data on first run if not already loaded
if not st.session_state.data_loaded:
    all_food_items_lower = [item.lower() for item in TARGET_FOOD_ITEMS]
    with st.sidebar: # Messages should be in the sidebar
        st.session_state.df_full_merged, st.session_state.df_national_avg = load_and_prepare_data_for_app(
            all_food_items_lower, years_back=10
        )
        if not st.session_state.df_full_merged.empty and not st.session_state.df_national_avg.empty:
            st.session_state.data_loaded = True
            st.success("Data loaded successfully! You can now explore.")
        else:
            st.error("Failed to load data. Please check your internet connection or file paths.")

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
            (st.session_state['df_full_merged']['Year'] >= (datetime.now().year - years_back_explorer))
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
        st.info("Data is being loaded. Please wait or check the messages in the sidebar.")

# --- Predictor Tab ---
with tab2:
    st.markdown("Forecast future national average food prices using pre-trained LSTM models.")

    if st.session_state.data_loaded and not st.session_state.df_national_avg.empty:
        # User input for prediction
        food_item_to_forecast = st.selectbox(
            "Select Food Item to Forecast:",
            CAPITALIZED_FOOD_ITEMS,
            key="predictor_food_select"
        )
        num_months_to_forecast = st.slider(
            "Number of Months to Forecast:",
            min_value=1,
            max_value=12,
            value=3,
            key="predictor_months_slider"
        )

        if st.button("Generate Forecast", key="generate_forecast_button"):
            # Load the model
            model = load_prediction_model(food_item_to_forecast)

            if model:
                # Prepare data for prediction
                df_item_national_avg = st.session_state.df_national_avg[
                    st.session_state.df_national_avg['Food_Item'] == food_item_to_forecast
                ].copy()

                if df_item_national_avg.empty:
                    st.warning(f"No historical national average data found for {food_item_to_forecast}. Cannot make predictions.")
                else:
                    # Sort by date to ensure correct sequence
                    df_item_national_avg = df_item_national_avg.sort_values(by='Date')
                    
                    # Use a MinMaxScaler fit on the historical data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    
                    # Reshape for scaler (needs 2D array)
                    historical_prices = df_item_national_avg['Price'].values.reshape(-1, 1)
                    
                    # Fit and transform the historical data
                    historical_prices_scaled = scaler.fit_transform(historical_prices)

                    # Determine sequence length from the model's input shape
                    # The input shape of the LSTM is typically (batch_size, sequence_length, features)
                    # We need the sequence_length (the second dimension)
                    try:
                        sequence_length = model.input_shape[1]
                    except IndexError:
                        st.error("Could not determine sequence length from model input shape. Ensure your model is correctly loaded.")
                        sequence_length = 12 # Default to 12 if shape can't be inferred

                    if historical_prices_scaled.shape[0] < sequence_length:
                        st.warning(f"Not enough historical data ({historical_prices_scaled.shape[0]} months) to create a sequence of length {sequence_length} for {food_item_to_forecast}. Need at least {sequence_length} months of data to start forecasting.")
                    else:
                        # Generate forecasts
                        with st.spinner(f"Forecasting prices for {food_item_to_forecast}..."):
                            predicted_prices = forecast_prices(
                                model, scaler, historical_prices_scaled, num_months_to_forecast, sequence_length
                            )
                        
                        # Create future dates for the forecast
                        last_historical_date = df_item_national_avg['Date'].max()
                        future_dates = [last_historical_date + timedelta(days=30 * i) for i in range(1, num_months_to_forecast + 1)]
                        
                        df_predictions = pd.DataFrame({
                            'Date': future_dates,
                            'Food_Item': food_item_to_forecast,
                            'Price': predicted_prices,
                            'Type': 'Forecast'
                        })

                        df_historical = df_item_national_avg.copy()
                        df_historical['Type'] = 'Historical'

                        df_combined = pd.concat([df_historical, df_predictions])

                        fig_forecast = px.line(
                            df_combined,
                            x='Date',
                            y='Price',
                            color='Type',
                            title=f"National Average Price Forecast for {food_item_to_forecast}",
                            labels={"Price": "Price (₦ per 100 KG)", "Date": "Date", "Type": "Data Type"},
                            line_dash='Type' # Differentiate historical and forecast lines
                        )
                        fig_forecast.update_traces(
                            selector=dict(name='Forecast'),
                            line=dict(color='red', dash='dot') # Make forecast line red and dotted
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)

                        st.markdown("#### 📈 Forecasted Prices")
                        st.dataframe(df_predictions[['Date', 'Price']].style.format({"Price": "₦{:,.2f}"}))
            else:
                st.warning("Please select a valid food item and ensure its model is available.")
    else:
        st.info("Data is being loaded. Please wait or check the messages in the sidebar.")
