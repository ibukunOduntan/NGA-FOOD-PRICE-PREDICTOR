import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import json
import joblib  # for loading models
import numpy as np
import warnings
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Nigerian Food Price Dashboard")

# Suppress specific warnings from statsmodels and pandas
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Global Configurations / Data Sources ---
API_URL = "https://microdata.worldbank.org/index.php/api/tables/data/fcv/wld_2021_rtfp_v02_m"
# TARGET_FOOD_ITEMS will be dynamically populated
BASE_MODEL_DIR = "models"  # Directory where pre-trained models are stored

# --- Functions to Fetch Data from External Sources (APIs, Files) ---

@st.cache_data(ttl=3600 * 24)  # Cache the result of this API call for 24 hours
def fetch_food_prices_from_api(api_url, country='Nigeria', years_back=10):
    limit = 10000
    offset = 0
    all_records = []
    
    # We'll fetch all available fields to dynamically determine food items and units
    params_initial = {'limit': 1, 'country': country} # Fetch 1 record to get column names
    try:
        response_initial = requests.get(api_url, params=params_initial, timeout=60)
        response_initial.raise_for_status()
        data_initial = response_initial.json()
        if 'data' in data_initial and data_initial['data']:
            # Get all column names from a sample record
            all_api_columns = list(data_initial['data'][0].keys())
        else:
            st.error("Failed to retrieve column names from API. Data might be empty or in an unexpected format.")
            return pd.DataFrame(), [], {} # Return empty DataFrame, list, and dict
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to World Bank API to get column names: {e}")
        return pd.DataFrame(), [], {}
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON from initial API response: {e}")
        return pd.DataFrame(), [], {}

    # Identify food item columns (starting with 'c_') and units
    food_price_columns_raw = [col for col in all_api_columns if col.startswith('c_') and '_unit' not in col]
    # Units are typically in columns like 'c_fooditem_unit'
    food_unit_columns_raw = {col.split('_unit')[0]: col for col in all_api_columns if col.endswith('_unit')}
    
    # Construct expected API columns including units
    expected_api_columns = ['country', 'adm1_name', 'year', 'month', 'DATES'] + food_price_columns_raw + list(food_unit_columns_raw.values())
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
    if df.empty: return pd.DataFrame(), [], {}

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
    
    actual_price_columns_in_df = [col for col in food_price_columns_raw if col in df.columns]
    
    if not actual_price_columns_in_df: 
        st.warning("No relevant food price columns found in the fetched World Bank data."); 
        return pd.DataFrame(), [], {}
    
    for col in actual_price_columns_in_df: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df_clean = df.dropna(subset=actual_price_columns_in_df, how='all')
    groupby_cols = ['country', 'adm1_name', 'year', 'month']
    df_avg = df_clean.groupby(groupby_cols)[actual_price_columns_in_df + list(food_unit_columns_raw.values())].mean(numeric_only=False).reset_index() # Keep units as object type

    # Extract dynamic food items and their units
    dynamic_food_items_lower = [col[2:] for col in actual_price_columns_in_df]
    food_item_units = {}
    for food_col in actual_price_columns_in_df:
        food_name_lower = food_col[2:]
        unit_col = f"{food_col}_unit"
        if unit_col in df.columns and not df[unit_col].empty:
            # Get the most common unit for each food item
            mode_unit = df[unit_col].mode()
            if not mode_unit.empty:
                food_item_units[food_name_lower.capitalize()] = mode_unit[0]
            else:
                food_item_units[food_name_lower.capitalize()] = "Unit N/A"
        else:
            food_item_units[food_name_lower.capitalize()] = "Unit N/A"


    # Rename price columns
    df_avg.rename(columns={col: col[2:].capitalize() for col in actual_price_columns_in_df}, inplace=True)
    df_avg.rename(columns={'year': 'Year', 'month': 'Month'}, inplace=True)
    
    if 'country' in df_avg.columns: df_avg.drop('country', axis=1, inplace=True)
    
    # Identify and separate Food Price Index
    food_price_index_col = "Food_price_index" # Assuming this will be capitalized after rename
    
    # Separate FPI data before melting
    df_fpi = pd.DataFrame()
    if food_price_index_col in df_avg.columns:
        df_fpi = df_avg[['adm1_name', 'Year', 'Month', food_price_index_col]].copy()
        df_fpi.rename(columns={food_price_index_col: 'Price'}, inplace=True)
        df_fpi['Food_Item'] = 'Food Price Index'
        df_avg.drop(columns=[food_price_index_col], inplace=True)
        # Also remove from dynamic food items list
        if 'food_price_index' in dynamic_food_items_lower:
            dynamic_food_items_lower.remove('food_price_index')
            food_item_units.pop('Food_price_index', None) # Remove from units as well if it exists

    # Prepare for melt: dynamically include unit columns if they exist and are distinct per food item
    id_vars_for_melt = ['adm1_name', 'Year', 'Month']
    
    # Collect actual unit columns present in df_avg *before* dropping them for melt
    unit_cols_for_melt = [f"{food_col[2:].capitalize()}_unit" for food_col in actual_price_columns_in_df if f"{food_col}_unit" in df_clean.columns]
    
    # Create a mapping for units for each specific Food_Item after melting
    # This approach captures the unit directly with the food item in the melted data
    df_long_list = []
    for food_col_raw in actual_price_columns_in_df:
        food_name_capitalized = food_col_raw[2:].capitalize()
        
        # Skip if it's the Food Price Index which has already been handled
        if food_name_capitalized == "Food_price_index":
            continue

        temp_df = df_avg[id_vars_for_melt + [food_name_capitalized]].copy()
        temp_df.rename(columns={food_name_capitalized: 'Price'}, inplace=True)
        temp_df['Food_Item'] = food_name_capitalized
        temp_df['Unit'] = food_item_units.get(food_name_capitalized, "Unit N/A") # Assign the unit

        df_long_list.append(temp_df)
    
    if df_long_list:
        df_long = pd.concat(df_long_list, ignore_index=True)
    else:
        df_long = pd.DataFrame(columns=['adm1_name', 'Year', 'Month', 'Price', 'Food_Item', 'Unit'])

    df_long.rename(columns={'adm1_name': 'State'}, inplace=True)
    df_long = df_long[df_long['State'] != 'Market Average']
    df_long.dropna(subset=['Price'], inplace=True)
    df_long.sort_values(by=['State', 'Year', 'Month', 'Food_Item'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    return df_long, dynamic_food_items_lower, food_item_units, df_fpi

@st.cache_data(ttl=3600 * 24)  # Cache for 24 hours
def load_geojson():
    try:
        filepath = "ngs.json"
        if not os.path.exists(filepath): st.error("GeoJSON file 'ngs.json' not found. Please ensure it's in the root directory."); return None
        with open(filepath, "r") as f: return json.load(f)
    except Exception as e: st.error(f"Error loading GeoJSON: {e}"); return None

@st.cache_data(ttl=3600 * 24)  # Cache the final merged dataset for 24 hours
def load_and_merge_all_data_directly(years_back):
    with st.spinner("Loading and preparing data... this might take a moment. 🎉"):
        df_food_prices, dynamic_food_items_lower, food_item_units, df_fpi = fetch_food_prices_from_api(API_URL, 'Nigeria', years_back)
        
        if df_food_prices.empty and df_fpi.empty: 
            st.error("Failed to load any data. Please check API connectivity and data availability."); 
            return pd.DataFrame(), pd.DataFrame(), [], {}

        df_merged = df_food_prices.copy()
        df_merged['Date'] = pd.to_datetime(df_merged['Year'].astype(str) + '-' + df_merged['Month'].astype(str) + '-01')

        if not df_fpi.empty:
            df_fpi['Date'] = pd.to_datetime(df_fpi['Year'].astype(str) + '-' + df_fpi['Month'].astype(str) + '-01')


    return df_merged, df_food_prices, dynamic_food_items_lower, food_item_units, df_fpi

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

    # For prediction, auto_arima needs the last few observed points for y
    # and will predict n_periods into the future.
    return log_series # Return the log series, as the model expects this for prediction.

# --- ARIMA Forecasting Function (Loads pre-trained model) ---
@st.cache_resource(ttl=3600)  # Cache the loaded model for 1 hour
def load_and_forecast_arima_model(food_item_lower, ts_log_series_hash, forecast_steps):
    """
    Loads a pre-trained ARIMA model and generates a forecast.
    Uses a hash of the recent log-transformed time series data to ensure caching works effectively.
    """
    model_filename = f"{food_item_lower.replace(' ', '_')}_model.pkl" # Use the new naming convention
    model_path = os.path.join(BASE_MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        st.error(f"Pre-trained model not found for {food_item_lower.capitalize()}. Please ensure you've trained and saved the models correctly: {model_path}")
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

            full_historical_series = st.session_state.df_full_merged[
                st.session_state.df_full_merged['Food_Item'] == food_item_lower.capitalize()
            ].groupby('Date')['Price'].mean().asfreq('MS')

            if full_historical_series.empty:
                   st.error(f"No historical data found for {food_item_lower.capitalize()} to determine last date for forecasting index.")
                   return pd.Series(dtype='float64'), None

            last_historical_date = full_historical_series.index[-1]

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

# Initialize session state variables if they don't exist
if 'df_full_merged' not in st.session_state: st.session_state.df_full_merged = pd.DataFrame()
if 'df_food_prices_raw' not in st.session_state: st.session_state.df_food_prices_raw = pd.DataFrame()
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
if 'dynamic_food_items_lower' not in st.session_state: st.session_state.dynamic_food_items_lower = []
if 'capitalized_food_items' not in st.session_state: st.session_state.capitalized_food_items = []
if 'food_item_units' not in st.session_state: st.session_state.food_item_units = {}
if 'df_fpi' not in st.session_state: st.session_state.df_fpi = pd.DataFrame()


with st.sidebar:
    if st.button("Load All Data", key="load_analyze_button") or not st.session_state.data_loaded:
        st.session_state.df_full_merged, st.session_state.df_food_prices_raw, \
        st.session_state.dynamic_food_items_lower, st.session_state.food_item_units, \
        st.session_state.df_fpi = load_and_merge_all_data_directly(years_back=10)
        
        # Capitalize the dynamically fetched food items for display
        st.session_state.capitalized_food_items = [item.capitalize() for item in st.session_state.dynamic_food_items_lower]

        if not st.session_state.df_full_merged.empty or not st.session_state.df_fpi.empty:
            st.session_state.data_loaded = True
            st.success("Data loaded successfully! You can now explore and predict.")
        else:
            st.error("Failed to load data. Please check your internet connection or file paths.")

# After loading data, populate the multiselect with dynamic food items
selected_food_items_explorer = st.sidebar.multiselect(
    "Select Food Items:", 
    st.session_state.capitalized_food_items, 
    default=['Maize'] if 'Maize' in st.session_state.capitalized_food_items else (st.session_state.capitalized_food_items[0:1] if st.session_state.capitalized_food_items else []), 
    key="explorer_food_select"
)
years_back_explorer = st.sidebar.slider("No. of years:", min_value=1, max_value=10, value=5, key="explorer_years_slider")


st.title("🥦 Nigerian Food Price Dashboard")
st.markdown("""
Welcome to the interactive dashboard to explore food price trends across Nigerian states and predict future prices.
""")

tab1, tab2 = st.tabs(["📊 Data Explorer", "📈 Food Price Index Prediction"])

with tab1:
    st.markdown("Historical price data is pulled from the World Bank Monthly food price estimates API")
    st.markdown("This tab lets you analyze food price trends, map data, and download cleaned datasets.")
    
    if st.session_state.data_loaded:
        
        # Display food item units
        st.markdown("#### 📏 Food Item Units (from World Bank Data)")
        if st.session_state.food_item_units:
            units_display = " | ".join([f"**{item}**: {unit}" for item, unit in st.session_state.food_item_units.items()])
            st.info(f"Units found in the dataset: {units_display}")
        else:
            st.info("No specific units found for food items in the dataset.")
        
        # Filter food price data (excluding FPI)
        food_data_explorer_filtered = st.session_state.df_food_prices_raw[
            (st.session_state.df_food_prices_raw['Food_Item'].isin(selected_food_items_explorer)) &
            (st.session_state.df_food_prices_raw['Year'] >= (datetime.now().year - years_back_explorer))
        ].copy()
        food_data_explorer_filtered['Date'] = pd.to_datetime(food_data_explorer_filtered['Year'].astype(str) + '-' + food_data_explorer_filtered['Month'].astype(str) + '-01')

        if food_data_explorer_filtered.empty and st.session_state.df_fpi.empty:
            st.info("No data available for the selected food items and years in the explorer. Try adjusting filters.")
        else:
            st.markdown("---") # Separator

            st.markdown("#### 📊 Data Quality Check (Food Prices)")
            st.markdown("This section helps you assess the completeness and reliability of the food price dataset.")
            total = len(food_data_explorer_filtered)
            missing_price = food_data_explorer_filtered['Price'].isna().sum()
            zero_price = (food_data_explorer_filtered['Price'] == 0).sum()
            st.info(f"Missing prices: {missing_price} | Zero prices: {zero_price} | Total entries: {total}")

            st.markdown("#### 🗺️ Choropleth Map (Average Food Prices)")
            st.markdown("Visualize average food prices by state using a color-coded map.")
            nigeria_geojson = load_geojson()
            if nigeria_geojson:
                try:
                    available_map_items = food_data_explorer_filtered['Food_Item'].unique()
                    if selected_food_items_explorer:  # Only show the map if a food item is selected
                        selected_food_for_map = st.selectbox(
                            "Select Food Item for Map:",
                            available_map_items,
                            index=0 if available_map_items.size > 0 else None,
                            key="map_food_select"
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

            # New: Price Trend Over Time for a Selected Food Item (Average Across States)
            st.markdown("---") # Separator
            st.markdown("#### 📈 Average Price Trend Over Time for a Food Item (Across All States)")
            st.markdown("Select a food item to view its average price trend across all states for the set time period.")

            # Dropdown for selecting a single food item
            food_item_for_avg_trend = st.selectbox(
                "Select Food Item to view average trend:",
                st.session_state.capitalized_food_items,
                key="food_item_avg_trend_select"
            )

            if food_item_for_avg_trend:
                df_avg_food_price_trend = food_data_explorer_filtered[
                    food_data_explorer_filtered['Food_Item'] == food_item_for_avg_trend
                ].groupby('Date')['Price'].mean().reset_index()

                if not df_avg_food_price_trend.empty:
                    fig_avg_trend = px.line(
                        df_avg_food_price_trend,
                        x='Date',
                        y='Price',
                        title=f'Average Price of {food_item_for_avg_trend} Over Time (Across All States)',
                        labels={'Price': f'Average Price (Naira / {st.session_state.food_item_units.get(food_item_for_avg_trend, "Unit N/A")})', 'Date': 'Date'},
                        hover_data={'Price': ':.2f'}
                    )
                    fig_avg_trend.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_avg_trend, use_container_width=True)
                else:
                    st.info(f"No data available for {food_item_for_avg_trend} to show average trend.")
            else:
                st.info("Please select a food item to view its average price trend.")

            # New: Average Food Price Trend Across User Set Time Period for Each State
            st.markdown("---") # Separator
            st.markdown("#### 📊 Average Food Price Trend for Each State (All Food Items)")
            st.markdown("Select a state to view the price trends of all food items within that state over the set time period.")

            # Dropdown for selecting a single state
            available_states = food_data_explorer_filtered['State'].unique().tolist()
            state_for_multi_line_trend = st.selectbox(
                "Select a State to view multi-line trend:",
                available_states,
                key="state_multi_line_trend_select"
            )

            if state_for_multi_line_trend:
                df_state_food_prices = food_data_explorer_filtered[
                    food_data_explorer_filtered['State'] == state_for_multi_line_trend
                ]

                if not df_state_food_prices.empty:
                    fig_state_trend = px.line(
                        df_state_food_prices,
                        x='Date',
                        y='Price',
                        color='Food_Item',
                        title=f'Food Price Trends in {state_for_multi_line_trend} Over Time',
                        labels={'Price': 'Price (Naira)', 'Date': 'Date'},
                        hover_data={'Food_Item': True, 'Price': ':.2f', 'Unit': True}
                    )
                    fig_state_trend.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_state_trend, use_container_width=True)
                else:
                    st.info(f"No data available for {state_for_multi_line_trend} to show food price trends.")
            else:
                st.info("Please select a state to view its food price trends.")

            # New: Correlation Plot of Food Prices
            st.markdown("---") # Separator
            st.markdown("#### 🤝 Food Price Correlation Plot")
            st.markdown("Understand how the average price *changes* of different food items (across all states) correlate with each other. A higher correlation (closer to 1 or -1) indicates a stronger relationship in their proportional movements.")

            # Prepare data for correlation calculation using average price percentage change
            df_correlation_prep = food_data_explorer_filtered.copy()

            # Ensure 'Date' column is present and correct for time-series operations
            df_correlation_prep['Date'] = pd.to_datetime(df_correlation_prep['Year'].astype(str) + '-' + df_correlation_prep['Month'].astype(str) + '-01')

            # Calculate the monthly average price for each food item across all states
            df_avg_prices = df_correlation_prep.groupby(['Date', 'Food_Item'])['Price'].mean().reset_index()

            # Pivot table to get average prices of each food item as columns, indexed by Date
            df_wide_avg_prices = df_avg_prices.pivot_table(
                index='Date',
                columns='Food_Item',
                values='Price'
            )

            # Calculate percentage change (returns) for these average prices and drop any NaNs
            df_returns_avg = df_wide_avg_prices.pct_change().dropna()

            # Check if all targeted food items are present in the 'returns' DataFrame
            # before attempting to compute correlation, to avoid errors and show meaningful plot
            required_columns_for_correlation = set(selected_food_items_explorer)
            current_columns_in_returns = set(df_returns_avg.columns)

            if not df_returns_avg.empty and len(df_returns_avg.columns) > 1 and required_columns_for_correlation.issubset(current_columns_in_returns):
                return_corr_matrix = df_returns_avg.corr()

                fig_corr = px.imshow(
                    return_corr_matrix,
                    text_auto=True,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='Average Price Change Correlation Between Food Items'
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                st.markdown("##### Smart Insights on Correlation:")
                # Analyze correlation matrix for insights
                np.fill_diagonal(return_corr_matrix.values, np.nan)  # Ignore self-correlation

                threshold = 0.75
                max_pairs = 2

                # Extract top correlations above threshold
                most_correlated = return_corr_matrix.stack().nlargest(20).index.tolist()  # get more to filter later

                top_pairs = []
                seen_pairs = set()

                for idx in most_correlated:
                    item1, item2 = idx
                    if item1 != item2 and frozenset({item1, item2}) not in seen_pairs:
                        corr_val = return_corr_matrix.loc[item1, item2]
                        if corr_val >= threshold:
                            top_pairs.append((item1, item2, corr_val))
                            seen_pairs.add(frozenset({item1, item2}))
                        if len(top_pairs) >= max_pairs:
                            break

                if top_pairs:
                    for item1, item2, corr_val in top_pairs:
                        st.info(f"**Most Correlated:** **{item1}** and **{item2}** show a strong positive correlation of **{corr_val:.2f}**. This suggests their average prices across states tend to move in the same direction, possibly due to shared market factors, supply chain, or substitutability.")
                else:
                    st.info(f"No strong positive correlations (>={threshold}) found between distinct food items in their average price changes.")

                # Least correlated (closest to 0 or negative)
                least_correlated = return_corr_matrix.stack().nsmallest(2).index.tolist()
                if least_correlated:
                    bottom_pairs = []
                    seen_pairs_bottom = set()
                    for idx in least_correlated:
                        item1, item2 = idx
                        if item1 != item2 and frozenset({item1, item2}) not in seen_pairs_bottom:
                            bottom_pairs.append((item1, item2, return_corr_matrix.loc[item1, item2]))
                            seen_pairs_bottom.add(frozenset({item1, item2}))
                        if len(bottom_pairs) >= 1: # Just show the bottom one meaningful pair
                            break
                    if bottom_pairs:
                        item1, item2, corr_val = bottom_pairs[0]
                        st.info(f"**Least Correlated (or Negatively Correlated):** **{item1}** and **{item2}** have a correlation of **{corr_val:.2f}**. A value close to zero or negative indicates little to no linear relationship, or an inverse relationship, suggesting their average price changes are largely independent.")
                    else:
                        st.info("No distinct food items found with very low or negative correlations in their average price changes.")
                else:
                    st.info("Not enough distinct food items selected to determine least correlation.")
            else:
                if df_returns_avg.empty:
                    st.info("Not enough data with sufficient history to calculate meaningful average price change correlations. Please ensure you have selected enough years and food items.")
                else:
                    missing_items = required_columns_for_correlation - current_columns_in_returns
                    st.info(f"Correlation plot for *all* target food items is shown only when data for every selected item is available. Missing: {', '.join(missing_items)}")

            st.markdown("---") # Separator
            st.markdown("#### 📈 Food Price Index Trend")
            st.markdown("This chart shows the trend of the Food Price Index over time across different states.")
            
            if not st.session_state.df_fpi.empty:
                # Filter FPI data by years_back_explorer
                df_fpi_filtered = st.session_state.df_fpi[
                    (st.session_state.df_fpi['Year'] >= (datetime.now().year - years_back_explorer))
                ].copy()

                if not df_fpi_filtered.empty:
                    fig_fpi = px.line(
                        df_fpi_filtered,
                        x='Date',
                        y='Price',
                        color='State',
                        title='Food Price Index Over Time by State',
                        labels={'Price': 'Food Price Index', 'Date': 'Date'},
                        hover_data={'State': True, 'Price': ':.2f'}
                    )
                    fig_fpi.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_fpi, use_container_width=True)
                else:
                    st.info("No Food Price Index data available for the selected time period.")
            else:
                st.info("No Food Price Index data found in the dataset.")

            # Raw Data Display and Download
            st.markdown("---") # Separator
            st.markdown("#### ⬇️ Raw Data & Download")
            st.markdown("View the raw data used for analysis and download it.")

            csv_data = food_data_explorer_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Food Prices Data as CSV",
                data=csv_data,
                file_name="nigerian_food_prices_explorer_data.csv",
                mime="text/csv",
            )
            
            if not st.session_state.df_fpi.empty:
                csv_fpi_data = st.session_state.df_fpi.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Food Price Index Data as CSV",
                    data=csv_fpi_data,
                    file_name="nigerian_food_price_index_data.csv",
                    mime="text/csv",
                )

    elif st.session_state.data_loaded and st.session_state.df_food_prices_raw.empty and st.session_state.df_fpi.empty:
        st.info("No data (food prices or FPI) loaded. Please check the API source or filters.")
    else:
        st.info("Please click 'Load All Data' in the sidebar to begin exploring.")

with tab2:
    st.markdown("### 📈 Food Price Index Prediction")
    st.markdown("This tab will host the functionality for predicting the Food Price Index using an ARIMA model.")
    st.info("Feature under development. Stay tuned!")
