import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import json
import joblib  # for loading models
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
BASE_MODEL_DIR = "saved_models"  # Directory where pre-trained models are stored

# --- Functions to Fetch Data from External Sources (APIs, Files) ---

@st.cache_data(ttl=3600 * 24)  # Cache the result of this API call for 24 hours
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

@st.cache_data(ttl=3600 * 24)  # Cache for 24 hours
def load_geojson():
    try:
        filepath = "ngs.json"
        if not os.path.exists(filepath): st.error("GeoJSON file 'ngs.json' not found. Please ensure it's in the root directory."); return None
        with open(filepath, "r") as f: return json.load(f)
    except Exception as e: st.error(f"Error loading GeoJSON: {e}"); return None

@st.cache_data(ttl=3600 * 24)  # Cache the final merged dataset for 24 hours
def load_and_merge_all_data_directly(target_food_items_lower, years_back):
    with st.spinner("Loading and preparing data... this might take a moment. 🎉"):
        df_food_prices = fetch_food_prices_from_api(API_URL, target_food_items_lower, 'Nigeria', years_back)
        if df_food_prices.empty: st.error("Failed to load food price data. Please check API connectivity and data availability."); return pd.DataFrame(), pd.DataFrame()

        df_merged = df_food_prices.copy()
        df_merged['Date'] = pd.to_datetime(df_merged['Year'].astype(str) + '-' + df_merged['Month'].astype(str) + '-01')

        if df_merged.empty: st.error("Merged dataset is empty. Check individual data sources."); return pd.DataFrame(), pd.DataFrame()

    return df_merged, df_food_prices

# --- Prepare time series for ARIMA forecasting ---
def prepare_time_series_for_arima(df, food_item):
    # For ARIMA, we don't need state specific data, as the model is trained per food item
    # We'll take the global average price for the food item.
    series_data = df[df['Food_Item'] == food_item].copy()
    if series_data.empty:
        return pd.Series(dtype='float64')

    # Group by month and year to get a single time series
    series = series_data.groupby(pd.to_datetime(series_data[['Year', 'Month']].assign(DAY=1))).Price.mean()
    series = series.asfreq('MS') # Ensure monthly frequency

    if (series <= 0).any():
        series = series.clip(lower=0.01) # Match preprocessing during training

    # ARIMA models were trained on log of series
    log_series = np.log(series)

    # For prediction, auto_arima needs the last few observed points for `y`
    # and will predict `n_periods` into the future.
    return log_series # Return the log series, as the model expects this for prediction.

# --- ARIMA Forecasting Function (Loads pre-trained model) ---
@st.cache_resource(ttl=3600)  # Cache the loaded model for 1 hour
def load_and_forecast_arima_model(food_item_lower, ts_log_series_hash, forecast_steps):
    """
    Loads a pre-trained ARIMA model and generates a forecast.
    Uses a hash of the recent log-transformed time series data to ensure caching works effectively.
    """
    # Retrieve the actual log series from session state
    ts_log_series = st.session_state[f'ts_log_{food_item_lower}_current']

    # THIS IS THE KEY CHANGE for loading the model:
    model_filename = f"{food_item_lower.replace(' ', '_')}.pkl"
    model_path = os.path.join(BASE_MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        st.error(f"Pre-trained model not found for {food_item_lower.capitalize()}. Please ensure you've trained and saved the models correctly: `{model_path}`")
        return pd.Series(dtype='float64'), None

    with st.spinner(f"Loading and generating forecast for {food_item_lower.capitalize()}..."):
        try:
            model = joblib.load(model_path)

            # Generate forecast on log scale
            forecast_log, conf_int_log = model.predict(
                n_periods=forecast_steps,
                return_conf_int=True
            )

            # Convert back from log scale to original price scale
            forecast = np.exp(forecast_log)
            conf_int_exp = np.exp(conf_int_log)

            # Create forecast index for plotting
            last_historical_date = ts_log_series.index[-1]
            forecast_index = pd.date_range(
                start=last_historical_date + pd.DateOffset(months=1),
                periods=forecast_steps,
                freq='MS' # Monthly start frequency
            )

            forecast_series = pd.Series(forecast, index=forecast_index)
            conf_int_df = pd.DataFrame(conf_int_exp, index=forecast_index, columns=['lower', 'upper'])

            return forecast_series, conf_int_df

        except Exception as e:
            st.error(f"Error loading or predicting with ARIMA model for {food_item_lower.capitalize()}: {e}")
            return pd.Series(dtype='float64'), None

# --- Streamlit App Setup ---
st.sidebar.title("🧊 Filter Options")
selected_food_items_explorer = st.sidebar.multiselect("Select Food Items:", CAPITALIZED_FOOD_ITEMS, default=['Maize'], key="explorer_food_select")
years_back_explorer = st.sidebar.slider("No. of years:", min_value=1, max_value=10, value=5, key="explorer_years_slider")

if 'df_full_merged' not in st.session_state: st.session_state.df_full_merged = pd.DataFrame()
if 'df_food_prices_raw' not in st.session_state: st.session_state.df_food_prices_raw = pd.DataFrame()
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False

with st.sidebar:
    if st.button("Load All Data", key="load_analyze_button") or not st.session_state.data_loaded:
        all_food_items_lower = [item.lower() for item in TARGET_FOOD_ITEMS]
        st.session_state.df_full_merged, st.session_state.df_food_prices_raw = load_and_merge_all_data_directly(
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
                    if selected_food_items_explorer:  # Only show the map if a food item is selected
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
                            fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
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
        # Get unique food items from the loaded data for prediction
        available_food_items_pred = sorted(st.session_state.df_full_merged['Food_Item'].unique().tolist())

        col1_pred, col2_pred = st.columns([1, 1])

        with col1_pred:
            selected_food_item_pred = st.selectbox("Select Food Item:", available_food_items_pred, key="predictor_food_select")
        with col2_pred:
            # Prompt user for years to forecast
            forecast_years = st.slider("Forecast Years:", min_value=1, max_value=5, value=1, key="forecast_years_slider")
            forecast_months = forecast_years * 12 # Convert years to months

        if st.button("Generate Forecast", key="generate_forecast_button"):
            food_item_lower = selected_food_item_pred.lower()

            # Prepare the historical time series (log-transformed) for the selected food item (global average)
            ts_log_series = prepare_time_series_for_arima(
                st.session_state.df_full_merged, selected_food_item_pred
            )

            if ts_log_series.empty:
                st.warning(f"Not enough historical data to generate a forecast for **{selected_food_item_pred}**. Please ensure sufficient data exists for this item.")
            else:
                # Store the log series in session state for the cached function to retrieve
                st.session_state[f'ts_log_{food_item_lower}_current'] = ts_log_series

                # Create a hash of the current log series for the cache key. This ensures the cached function
                # only re-runs if the underlying data changes, not just on every button click.
                ts_log_series_hash = hash(ts_log_series.to_json()) if not ts_log_series.empty else 0

                # Load and forecast using the cached function
                forecast_prices, conf_int_df = load_and_forecast_arima_model(
                    food_item_lower, ts_log_series_hash, forecast_months
                )

                if not forecast_prices.empty:
                    st.markdown(f"#### 🔮 Forecast for {selected_food_item_pred} Prices")

                    # Get original historical prices for plotting
                    historical_prices_original = st.session_state.df_full_merged[
                        st.session_state.df_full_merged['Food_Item'] == selected_food_item_pred
                    ].groupby('Date')['Price'].mean().asfreq('MS')

                    fig = px.line()
                    fig.add_scatter(x=historical_prices_original.index, y=historical_prices_original.values, mode='lines', name='Historical Prices')
                    fig.add_scatter(x=forecast_prices.index, y=forecast_prices.values, mode='lines', name='Forecasted Prices', line=dict(color='red'))

                    if conf_int_df is not None and not conf_int_df.empty:
                        fig.add_trace(px.scatter(x=conf_int_df.index, y=conf_int_df['lower']).data[0].update(
                            mode='lines', line=dict(width=0), showlegend=False
                        ))
                        fig.add_trace(px.scatter(x=conf_int_df.index, y=conf_int_df['upper']).data[0].update(
                            mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='Confidence Interval'
                        ))

                    fig.update_layout(
                        title=f'Food Price Forecast for {selected_food_item_pred}',
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
                        file_name=f"{selected_food_item_pred}_forecast.csv",
                        mime="text/csv",
                    )
                else:
                    st.info(f"Could not generate a forecast for **{selected_food_item_pred}**. This could be due to a missing or corrupted model file, or insufficient historical data.")
    else:
        st.info("Please click **'Load All Data'** in the sidebar to enable the predictor tab.")
