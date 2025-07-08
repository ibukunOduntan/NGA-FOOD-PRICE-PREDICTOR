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
# TARGET_FOOD_ITEMS will be dynamically populated from API response for Nigeria
BASE_MODEL_DIR = "models"  # Directory where pre-trained models are stored

# Static information about typical WFP units for Nigerian food prices
WFP_UNITS_INFO = {
    'Gari': '100 KG',
    'Groundnuts': '100 KG',
    'Maize': '100 KG',
    'Sorghum': '100 KG',
    'Cassava_meal': '100 KG', # Note: API might return 'cassava_meal'
    'Beans': '100 KG',
    'Rice': '50 KG',
    'Millet': '100 KG',
    'Yam': '1 KG',
    'Fish': 'KG', # General, specific type might vary
    'Oil (Palm)': '750 ML',
    'Salt': '250 G',
    'Sugar': '1.3 KG',
    'Tomatoes': '0.5 KG',
    'Milk': '20 G (or pcs)',
    'Eggs': '30 pcs',
    'Bananas': '1.3 KG',
    'Cowpeas': '100 KG',
    'Food_price_index': 'Index points'
}


# --- Functions to Fetch Data from External Sources (APIs, Files) ---

@st.cache_data(ttl=3600 * 24)
def fetch_food_prices_from_api(api_url, country='Nigeria', years_back=10):
    limit, offset = 10000, 0
    all_records = []

    # Initial fetch to get structure
    response_initial = requests.get(api_url, params={'limit': 1, 'country': country})
    response_initial.raise_for_status()
    data_initial = response_initial.json()

    if 'data' not in data_initial or not data_initial['data']:
        st.error("Initial API response is empty.")
        return pd.DataFrame(), [], pd.DataFrame()

    sample = pd.DataFrame(data_initial['data'])
    price_fields = [col for col in sample.columns if col.startswith('c_') and '_unit' not in col and col != 'c_food_price_index']
    fpi_column = 'c_food_price_index' if 'c_food_price_index' in sample.columns else None

    fields_to_fetch = ['country', 'adm1_name', 'year', 'month', 'DATES'] + price_fields
    if fpi_column:
        fields_to_fetch.append(fpi_column)

    # Paginated fetch
    while True:
        response = requests.get(api_url, params={
            'limit': limit,
            'offset': offset,
            'country': country,
            'fields': ','.join(fields_to_fetch)
        })
        response.raise_for_status()
        data = response.json().get('data', [])
        if not data:
            break
        all_records.extend(data)
        offset += limit

    df = pd.DataFrame(all_records)
    if df.empty:
        return pd.DataFrame(), [], pd.DataFrame()

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['month'] = pd.to_numeric(df['month'], errors='coerce')

    if 'DATES' in df.columns:
        df['DATES'] = pd.to_datetime(df['DATES'], errors='coerce')
        df.dropna(subset=['DATES', 'year', 'month'], inplace=True)
    else:
        df.dropna(subset=['year', 'month'], inplace=True)

    df = df[df['year'] >= datetime.now().year - years_back]

    # Convert all price fields to numeric
    # Convert all price fields to numeric
    for col in price_fields:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any price field that has even a single NaN
    price_fields = [col for col in price_fields if col in df.columns and df[col].notna().all()]
    
    # Rebuild df to only keep necessary fields + surviving price columns
    keep_cols = ['country', 'adm1_name', 'year', 'month', 'DATES'] + price_fields
    if fpi_column:
        keep_cols.append(fpi_column)
    
    df = df[keep_cols]



    if fpi_column:
        df[fpi_column] = pd.to_numeric(df[fpi_column], errors='coerce')
        if df[fpi_column].isnull().any():
            fpi_column = None

    # Drop rows where all remaining food prices are missing (shouldn’t happen now)
    df_clean = df.dropna(subset=price_fields, how='all').copy()

    if not price_fields and not fpi_column:
        st.warning("No valid food price columns found after NaN filtering.")
        return pd.DataFrame(), [], pd.DataFrame()

    # Group and average
    group_cols = ['country', 'adm1_name', 'year', 'month']
    avg_fields = price_fields + ([fpi_column] if fpi_column else [])
    df_avg = df_clean.groupby(group_cols)[avg_fields].mean().reset_index()

    # Rename columns
    df_avg.rename(columns={col: col[2:].capitalize() for col in price_fields}, inplace=True)
    if fpi_column:
        df_avg.rename(columns={fpi_column: 'Food_price_index'}, inplace=True)

    df_avg.rename(columns={'year': 'Year', 'month': 'Month'}, inplace=True)
    df_avg.drop(columns='country', inplace=True)

    # Extract FPI separately if available
    df_fpi = pd.DataFrame()
    if fpi_column and 'Food_price_index' in df_avg.columns:
        df_fpi = df_avg[['adm1_name', 'Year', 'Month', 'Food_price_index']].copy()
        df_fpi.rename(columns={'adm1_name': 'State', 'Food_price_index': 'Price'}, inplace=True)
        df_fpi['Food_Item'] = 'Food Price Index'
        df_avg.drop(columns=['Food_price_index'], inplace=True)

    # Melt to long format
    df_long = pd.melt(df_avg, id_vars=['adm1_name', 'Year', 'Month'], var_name='Food_Item', value_name='Price')
    df_long.rename(columns={'adm1_name': 'State'}, inplace=True)
    df_long.dropna(subset=['Price'], inplace=True)
    df_long.sort_values(by=['State', 'Year', 'Month', 'Food_Item'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    dynamic_food_items_lower = [col[2:].lower() for col in price_fields]

    return df_long, dynamic_food_items_lower, df_fpi


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
        df_food_prices, dynamic_food_items_lower, df_fpi = fetch_food_prices_from_api(API_URL, 'Nigeria', years_back)
        
        if df_food_prices.empty and df_fpi.empty: 
            st.error("Failed to load any data. Please check API connectivity and data availability."); 
            return pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame()

        df_merged = df_food_prices.copy()
        df_merged['Date'] = pd.to_datetime(df_merged['Year'].astype(str) + '-' + df_merged['Month'].astype(str) + '-01')

        if not df_fpi.empty:
            df_fpi['Date'] = pd.to_datetime(df_fpi['Year'].astype(str) + '-' + df_fpi['Month'].astype(str) + '-01')


    return df_merged, df_food_prices, dynamic_food_items_lower, df_fpi

# --- Prepare time series for ARIMA forecasting ---
def prepare_time_series_for_arima(df, food_item):
    series_data = df[df['Food_Item'] == food_item].copy()
    if series_data.empty:
        return pd.Series(dtype='float64')

    series = series_data.groupby(pd.to_datetime(series_data[['Year', 'Month']].assign(DAY=1))).Price.mean()
    series = series.asfreq('MS') # Ensure monthly frequency

    if (series <= 0).any():
        series = series.clip(lower=0.01) # Match preprocessing during training

    log_series = np.log(series)

    return log_series 

# --- ARIMA Forecasting Function (Loads pre-trained model) ---
@st.cache_resource(ttl=3600)  # Cache the loaded model for 1 hour
def load_and_forecast_arima_model(food_item_lower, ts_log_series_hash, forecast_steps):
    """
    Loads a pre-trained ARIMA model and generates a forecast.
    Uses a hash of the recent log-transformed time series data to ensure caching works effectively.
    """
    model_filename = f"{food_item_lower.replace(' ', '_')}_model.pkl" 
    model_path = os.path.join(BASE_MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        st.error(f"Pre-trained model not found for {food_item_lower.capitalize()}. Please ensure you've trained and saved the models correctly: {model_path}")
        return pd.Series(dtype='float64'), None

    with st.spinner(f"Loading and generating forecast for {food_item_lower.capitalize()}..."):
        try:
            model = joblib.load(model_path)

            forecast_log, conf_int_log = model.predict(
                n_periods=forecast_steps,
                return_conf_int=True
            )

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
                freq='MS' 
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
if 'df_fpi' not in st.session_state: st.session_state.df_fpi = pd.DataFrame()


with st.sidebar:
    if st.button("Load All Data", key="load_analyze_button") or not st.session_state.data_loaded:
        st.session_state.df_full_merged, st.session_state.df_food_prices_raw, \
        st.session_state.dynamic_food_items_lower, \
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
    default=(['Maize'] if 'Maize' in st.session_state.capitalized_food_items else 
             (st.session_state.capitalized_food_items[0:1] if st.session_state.capitalized_food_items else [])), 
    key="explorer_food_select"
)
years_back_explorer = st.sidebar.slider("No. of years:", min_value=1, max_value=10, value=5, key="explorer_years_slider")


st.title("🥦 Nigerian Food Price Dashboard")
st.markdown("""
Welcome to the interactive dashboard to explore food price trends across Nigerian states and predict future prices.
""")

# Display the static unit information
st.info(
    "**Note on Units:** The World Bank Food Price Monitoring and Analysis (FPMA) tool typically reports "
    "food prices for Nigeria in the following approximate units (though specific units might vary by item and dataset version):\n"
    "- **Gari, Groundnuts, Maize, Sorghum, Cowpeas, Millet**: ~100 KG\n"
    "- **Rice**: ~50 KG\n"
    "- **Cassava Meal (Gari, Yellow)**: ~100 KG\n"
    "- **Yam**: ~1 KG\n"
    "- **Fish**: ~KG\n"
    "- **Oil (Palm)**: ~750 ML\n"
    "- **Salt**: ~250 G\n"
    "- **Sugar**: ~1.3 KG\n"
    "- **Tomatoes**: ~0.5 KG\n"
    "- **Milk**: ~20 G (or pcs)\n"
    "- **Eggs**: ~30 pcs\n"
    "- **Bananasss**: ~1.3 KG\n"
    "Prices are in Nigerian Naira (NGN)."
)
st.markdown("---")


tab1, tab2 = st.tabs(["📊 Data Explorer", "📈 Food Price Index Prediction"])

with tab1:
    st.markdown("Historical price data is pulled from the World Bank Monthly food price estimates API")
    st.markdown("This tab lets you analyze food price trends, map data, and download cleaned datasets.")
    
    if st.session_state.data_loaded:
        
        # Filter food price data (excluding FPI)
        food_data_explorer_filtered = st.session_state.df_food_prices_raw[
            (st.session_state.df_food_prices_raw['Food_Item'].isin(selected_food_items_explorer)) &
            (st.session_state.df_food_prices_raw['Year'] >= (datetime.now().year - years_back_explorer))
        ].copy()
        food_data_explorer_filtered['Date'] = pd.to_datetime(food_data_explorer_filtered['Year'].astype(str) + '-' + food_data_explorer_filtered['Month'].astype(str) + '-01')

        if food_data_explorer_filtered.empty and st.session_state.df_fpi.empty:
            st.info("No data available for the selected food items and years in the explorer. Try adjusting filters or loading data.")
        else:
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
                    if selected_food_items_explorer and available_map_items.size > 0:  
                        selected_food_for_map = st.selectbox(
                            "Select Food Item for Map:",
                            available_map_items,
                            index=0,
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
                                hover_data={'Price': ':.2f'},
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
            st.markdown("---") 
            st.markdown("#### 📈 Average Price Trend Over Time for a Food Item (Across All States)")
            st.markdown("Select a food item to view its average price trend across all states for the set time period.")

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
                    # Dynamically adjust Y-axis label based on the selected food item's general unit
                    unit_for_display = WFP_UNITS_INFO.get(food_item_for_avg_trend, "Unit N/A").replace("~", "") # Remove approx symbol
                    y_axis_label = f'Average Price (Naira / {unit_for_display})' if unit_for_display != "Unit N/A" else 'Average Price (Naira)'

                    fig_avg_trend = px.line(
                        df_avg_food_price_trend,
                        x='Date',
                        y='Price',
                        title=f'Average Price of {food_item_for_avg_trend} Over Time (Across All States)',
                        labels={'Price': y_axis_label, 'Date': 'Date'},
                        hover_data={'Price': ':.2f'}
                    )
                    fig_avg_trend.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_avg_trend, use_container_width=True)
                else:
                    st.info(f"No data available for {food_item_for_avg_trend} to show average trend.")
            else:
                st.info("Please select a food item to view its average price trend.")

            # New: Average Food Price Trend Across User Set Time Period for Each State
            st.markdown("---") 
            st.markdown("#### 📊 Average Food Price Trend for Each State (All Food Items)")
            st.markdown("Select a state to view the price trends of all food items within that state over the set time period.")

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
                        hover_data={'Food_Item': True, 'Price': ':.2f'} # Removed 'Unit' as it's no longer in df_long
                    )
                    fig_state_trend.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_state_trend, use_container_width=True)
                else:
                    st.info(f"No data available for {state_for_multi_line_trend} to show food price trends.")
            else:
                st.info("Please select a state to view its food price trends.")

            # New: Correlation Plot of Food Prices
            st.markdown("---") 
            st.markdown("#### 🤝 Food Price Correlation Plot")
            st.markdown("Understand how the average price *changes* of different food items (across all states) correlate with each other. A higher correlation (closer to 1 or -1) indicates a stronger relationship in their proportional movements.")

            df_correlation_prep = food_data_explorer_filtered.copy()
            df_correlation_prep['Date'] = pd.to_datetime(df_correlation_prep['Year'].astype(str) + '-' + df_correlation_prep['Month'].astype(str) + '-01')
            df_avg_prices = df_correlation_prep.groupby(['Date', 'Food_Item'])['Price'].mean().reset_index()
            df_wide_avg_prices = df_avg_prices.pivot_table(
                index='Date',
                columns='Food_Item',
                values='Price'
            )
            df_returns_avg = df_wide_avg_prices.pct_change().dropna()

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
                np.fill_diagonal(return_corr_matrix.values, np.nan)  

                threshold = 0.75
                max_pairs = 2

                most_correlated = return_corr_matrix.stack().nlargest(20).index.tolist()  

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

                least_correlated = return_corr_matrix.stack().nsmallest(2).index.tolist()
                if least_correlated:
                    bottom_pairs = []
                    seen_pairs_bottom = set()
                    for idx in least_correlated:
                        item1, item2 = idx
                        if item1 != item2 and frozenset({item1, item2}) not in seen_pairs_bottom:
                            bottom_pairs.append((item1, item2, return_corr_matrix.loc[item1, item2]))
                            seen_pairs_bottom.add(frozenset({item1, item2}))
                        if len(bottom_pairs) >= 1: 
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

            st.markdown("---") 
            st.markdown("#### 📈 Food Price Index Trend")
            st.markdown("This chart shows the trend of the Food Price Index over time across different states.")
            
            if not st.session_state.df_fpi.empty:
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
            st.markdown("---") 
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
