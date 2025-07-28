import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import json
import joblib # for loading models
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
BASE_MODEL_DIR = "models" # Directory where pre-trained models is stored

# Static information about typical WFP units for Nigerian food prices
WFP_UNITS_INFO = {
    'Gari': '100 KG',
    'Groundnuts': '100 KG',
    'Maize': '100 KG',
    'Sorghum': '100 KG',
    'Cassava_meal': '100 KG', # Note: API might return 'cassava_meal'
    'Beans (white)': '2.5 KG',
    'Rice (imported)': '2.8 KG',
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
    price_fields_initial = [col for col in sample.columns if col.startswith('c_') and '_unit' not in col and col != 'c_food_price_index']
    fpi_column = 'c_food_price_index' if 'c_food_price_index' in sample.columns else None

    fields_to_fetch = ['country', 'adm1_name', 'year', 'month', 'DATES'] + price_fields_initial
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

    # Convert all price fields to numeric and identify those with NaNs
    price_fields = []
    for col in price_fields_initial:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Only keep columns that do NOT have any NaN values
            if df[col].notna().all():
                price_fields.append(col)
            else:
                pass
    # Rebuild df to only keep necessary fields + surviving price columns
    keep_cols = ['country', 'adm1_name', 'year', 'month', 'DATES'] + price_fields
    if fpi_column:
        keep_cols.append(fpi_column)
    
    # Ensure all columns in keep_cols actually exist in df before selection
    df = df[[col for col in keep_cols if col in df.columns]]


    if fpi_column and fpi_column in df.columns:
        df[fpi_column] = pd.to_numeric(df[fpi_column], errors='coerce')
        # If FPI itself has NaNs, we might choose to drop it or handle it differently.
        # For now, if it has any NaNs, we will treat it as if it wasn't available.
        if df[fpi_column].isnull().any():
            fpi_column = None
    else:
        fpi_column = None


    # Drop rows where all remaining food prices are missing
    # This acts as a final safeguard after column-level NaN removal
    if price_fields: # Only try to dropna if there are still price fields left
        df_clean = df.dropna(subset=price_fields, how='all').copy()
    else:
        df_clean = df.copy() # If no price fields, no need to drop rows based on them


    if not price_fields and not fpi_column:
        st.warning("No valid food price or FPI columns found after NaN filtering.")
        return pd.DataFrame(), [], pd.DataFrame()

    # Group and average
    group_cols = ['country', 'adm1_name', 'year', 'month']
    avg_fields = price_fields + ([fpi_column] if fpi_column else [])
    
    # Filter avg_fields to only include columns that are actually present in df_clean
    avg_fields = [col for col in avg_fields if col in df_clean.columns]

    if not avg_fields: # If no columns to average after filtering, return empty
        return pd.DataFrame(), [], df_fpi

    df_avg = df_clean.groupby(group_cols)[avg_fields].mean().reset_index()

    # Rename columns
    df_avg.rename(columns={col: col[2:].capitalize() for col in price_fields}, inplace=True)
    if fpi_column and fpi_column in df_avg.columns: # Check if FPI column is still there after grouping
        df_avg.rename(columns={fpi_column: 'Food_price_index'}, inplace=True)

    df_avg.rename(columns={'year': 'Year', 'month': 'Month'}, inplace=True)
    df_avg.drop(columns='country', inplace=True)

    # Extract FPI separately if available
    df_fpi = pd.DataFrame()
    if 'Food_price_index' in df_avg.columns:
        df_fpi = df_avg[['adm1_name', 'Year', 'Month', 'Food_price_index']].copy()
        df_fpi.rename(columns={'adm1_name': 'State', 'Food_price_index': 'Price'}, inplace=True)
        df_fpi['Food_Item'] = 'Food Price Index'
        df_avg.drop(columns=['Food_price_index'], inplace=True)

    # Melt to long format
    # Only include columns that are actually present in df_avg for melting
    columns_to_melt = [col for col in df_avg.columns if col not in ['adm1_name', 'Year', 'Month']]
    
    if not columns_to_melt: # If nothing to melt, return empty
        return pd.DataFrame(), [], df_fpi

    df_long = pd.melt(df_avg, id_vars=['adm1_name', 'Year', 'Month'], var_name='Food_Item', value_name='Price', value_vars=columns_to_melt)
    df_long.rename(columns={'adm1_name': 'State'}, inplace=True)
    df_long.dropna(subset=['Price'], inplace=True) # Drop rows where price became NaN after melting
    df_long.sort_values(by=['State', 'Year', 'Month', 'Food_Item'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    df_long = df_long[df_long['State'] != 'Market Average']
    if not df_fpi.empty:
        df_fpi = df_fpi[df_fpi['State'] != 'Market Average']


    # Get dynamic food items from the (potentially filtered) df_long
    all_dynamic_food_items_lower = [item.lower() for item in df_long['Food_Item'].unique()]

    # Filter to only the first 8 items after full processing
    # This ensures consistency for display and subsequent filtering
    selected_dynamic_food_items_lower = sorted(all_dynamic_food_items_lower)[:8]
    df_long_filtered_to_8 = df_long[df_long['Food_Item'].isin([item.capitalize() for item in selected_dynamic_food_items_lower])]

    return df_long_filtered_to_8, selected_dynamic_food_items_lower, df_fpi


@st.cache_data(ttl=3600 * 24) # Cache for 24 hours
def load_geojson():
    try:
        filepath = "ngs.json"
        if not os.path.exists(filepath): st.error("GeoJSON file 'ngs.json' not found. Please ensure it's in the root directory."); return None
        with open(filepath, "r") as f: return json.load(f)
    except Exception as e: st.error(f"Error loading GeoJSON: {e}"); return None

@st.cache_data(ttl=3600 * 24) # Cache the final merged dataset for 24 hours
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
        # This list now *already* contains only the top 8
        st.session_state.capitalized_food_items = [item.capitalize() for item in st.session_state.dynamic_food_items_lower]

        if not st.session_state.df_full_merged.empty or not st.session_state.df_fpi.empty:
            st.session_state.data_loaded = True
            st.success("Data loaded successfully! You can now explore and predict.")
        else:
            st.error("Failed to load data. Please check your internet connection or file paths.")

# After loading data, populate the multiselect with dynamic food items
# The list st.session_state.capitalized_food_items now *only* contains the first 8
selected_food_items_explorer = st.sidebar.multiselect(
    "Select Food Items (8 Items Filtered):",
    st.session_state.capitalized_food_items, # Options are already limited to 8
    default=st.session_state.capitalized_food_items, # Default to selecting all 8
    key="explorer_food_select"
)

years_back_explorer = st.sidebar.slider("No. of years:", min_value=1, max_value=10, value=5, key="explorer_years_slider")


st.title("🥦 Nigerian Food Price Dashboard")
st.markdown("""
Welcome to the interactive dashboard to explore food price trends across Nigerian states.
""")

# Display the static unit information
st.info(
    "**Note on Units:** The World Bank Food Price Monitoring and Analysis (FPMA) tool typically reports "
    "food prices for Nigeria in the following approximate units (though specific units might vary by item and dataset version):\n"
    "- **Gari, Groundnuts, Maize, Sorghum, Cowpeas**: ~100 KG\n"
    "- **Millet**: ~2.6 KG\n"
    "- **Rice**: ~2.8 KG\n"
    "- **Beans (white)**: ~2.5 KG\n"
    "- **Cassava Meal (Gari, Yellow)**: ~100 KG\n"
    "- **Maize flour**: ~2.1 KG\n"
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


tab1 = st.tabs(["📊 Data Explorer"])[0]

with tab1:
    st.markdown("Historical price data is pulled from the World Bank Monthly food price estimates API")
    st.markdown("This tab lets you analyze food price trends and map data.")
    
    if st.session_state.data_loaded:
        
        food_data_explorer_filtered = st.session_state.df_food_prices_raw[
            (st.session_state.df_food_prices_raw['Food_Item'].isin(selected_food_items_explorer)) &
            (st.session_state.df_food_prices_raw['Year'] >= (datetime.now().year - years_back_explorer))
        ].copy()
        food_data_explorer_filtered['Date'] = pd.to_datetime(food_data_explorer_filtered['Year'].astype(str) + '-' + food_data_explorer_filtered['Month'].astype(str) + '-01')

        if food_data_explorer_filtered.empty and st.session_state.df_fpi.empty:
            st.info("No data available for the selected food items and years in the explorer. Try adjusting filters or loading data.")
        else:
            # 1. Choropleth Map of Latest Prices by State
            st.markdown("---")
            st.markdown("#### ✅ 1. Choropleth Map of Latest Prices by State")
            st.markdown("Shows the **price of a selected food item across states for the most recent month**. This helps identify regional price disparities at a glance.")
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
                            df_selected_food = food_data_explorer_filtered[food_data_explorer_filtered['Food_Item'] == selected_food_for_map]
                            
                            if not df_selected_food.empty:
                                latest_date = df_selected_food['Date'].max()
                                df_map_data_current = df_selected_food[df_selected_food['Date'] == latest_date]
                                df_map_data_final = df_map_data_current.groupby('State')['Price'].mean().reset_index()

                                if not df_map_data_final.empty:
                                    fig_map = px.choropleth_mapbox(
                                        df_map_data_final,
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
                                        title=f'Price of {selected_food_for_map} by State ({latest_date.strftime("%B %Y")})'
                                    )
                                    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                                    st.plotly_chart(fig_map, use_container_width=True)
                                else:
                                    st.info(f"No price data available for {selected_food_for_map} for the latest month.")
                            else:
                                st.info(f"No data available for the selected food item '{selected_food_for_map}'.")
                        else:
                            st.info("No food item selected for map visualization.")
                    else:
                        st.info("Please select at least one food item in the sidebar to view the map.")
                except Exception as e:
                    st.error(f"Error generating choropleth map: {e}")
            else:
                st.warning("Cannot display map: GeoJSON data not loaded.")

            # 2. Line Chart of Monthly Price Trends (Per State or Nationwide)
            st.markdown("---")
            st.markdown("#### ✅ 2. Line Chart of Monthly Price Trends (Per State or Nationwide)")
            st.markdown("Shows **how prices for a selected food item change over time**. Users can select a state to compare with the national average or other states.")
            
            trend_food_item = st.selectbox(
                "Select Food Item for Trend Analysis:",
                st.session_state.capitalized_food_items,
                key="trend_food_item_select"
            )

            trend_states = ['Nationwide Average'] + sorted(food_data_explorer_filtered['State'].unique().tolist())
            selected_trend_state = st.selectbox(
                "Select State for Trend Analysis:",
                trend_states,
                key="trend_state_select"
            )

            if trend_food_item:
                df_trend = food_data_explorer_filtered[food_data_explorer_filtered['Food_Item'] == trend_food_item].copy()
                
                if selected_trend_state == 'Nationwide Average':
                    df_plot = df_trend.groupby('Date')['Price'].mean().reset_index()
                    title = f'National Average Price Trend for {trend_food_item} Over Time'
                else:
                    df_plot = df_trend[df_trend['State'] == selected_trend_state].groupby('Date')['Price'].mean().reset_index()
                    title = f'Price Trend for {trend_food_item} in {selected_trend_state} Over Time'
                
                if not df_plot.empty:
                    unit_for_display = WFP_UNITS_INFO.get(trend_food_item, "Unit N/A").replace("~", "")
                    y_axis_label = f'Price (Naira / {unit_for_display})' if unit_for_display != "Unit N/A" else 'Price (Naira)'
                    
                    fig_trend = px.line(
                        df_plot,
                        x='Date',
                        y='Price',
                        title=title,
                        labels={'Price': y_axis_label, 'Date': 'Date'},
                        hover_data={'Price': ':.2f'}
                    )
                    fig_trend.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info(f"No data available for {trend_food_item} in {selected_trend_state}.")
            else:
                st.info("Please select a food item to view its trend.")

            # 3. Bar Chart: Top 10 Most Expensive States (Latest Month)
            st.markdown("---")
            st.markdown("#### ✅ 3. Bar Chart: Top 10 Most Expensive States (Latest Month)")
            st.markdown("Shows **which states have the highest prices for a selected item** for the latest available month. This is good for comparing at a glance without the map.")

            bar_chart_food_item = st.selectbox(
                "Select Food Item for Top States Bar Chart:",
                st.session_state.capitalized_food_items,
                key="bar_food_item_select"
            )

            if bar_chart_food_item:
                df_bar_data = food_data_explorer_filtered[food_data_explorer_filtered['Food_Item'] == bar_chart_food_item].copy()
                if not df_bar_data.empty:
                    latest_date_bar = df_bar_data['Date'].max()
                    df_bar_latest_month = df_bar_data[df_bar_data['Date'] == latest_date_bar].copy()
                    
                    # Group by state and get the average price for the latest month
                    df_bar_grouped = df_bar_latest_month.groupby('State')['Price'].mean().reset_index()
                    df_top_10_expensive = df_bar_grouped.nlargest(10, 'Price')

                    if not df_top_10_expensive.empty:
                        unit_for_display = WFP_UNITS_INFO.get(bar_chart_food_item, "Unit N/A").replace("~", "")
                        x_axis_label = f'Price (Naira / {unit_for_display})' if unit_for_display != "Unit N/A" else 'Price (Naira)'
                        
                        fig_bar = px.bar(
                            df_top_10_expensive,
                            x='Price',
                            y='State',
                            orientation='h',
                            title=f'Top 10 Most Expensive States for {bar_chart_food_item} ({latest_date_bar.strftime("%B %Y")})',
                            labels={'Price': x_axis_label, 'State': 'State'},
                            color='Price',
                            color_continuous_scale="Viridis"
                        )
                        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info(f"No states found for {bar_chart_food_item} for the latest month.")
                else:
                    st.info(f"No data available for {bar_chart_food_item} to show top expensive states.")
            else:
                st.info("Please select a food item for the bar chart.")

            # 4. Choropleth Map of Monthly Price Change (% Change)
            st.markdown("---")
            st.markdown("#### ✅ 4. Choropleth Map of Monthly Price Change (% Change)")
            st.markdown("Shows **month-over-month or year-over-year percentage change in price**, color-coded to highlight where prices are rising or falling fastest.")

            change_food_item = st.selectbox(
                "Select Food Item for Price Change Map:",
                st.session_state.capitalized_food_items,
                key="change_map_food_select"
            )

            change_type = st.radio(
                "Select Change Type:",
                ('Month-over-Month', 'Year-over-Year'),
                key="change_type_radio"
            )

            if change_food_item and nigeria_geojson:
                df_change_data = food_data_explorer_filtered[food_data_explorer_filtered['Food_Item'] == change_food_item].copy()
                df_change_data = df_change_data.sort_values(by=['State', 'Date'])
                
                if not df_change_data.empty:
                    df_change_data['Price_Lagged'] = df_change_data.groupby('State')['Price'].shift(1 if change_type == 'Month-over-Month' else 12)
                    df_change_data['Price_Change_Pct'] = ((df_change_data['Price'] - df_change_data['Price_Lagged']) / df_change_data['Price_Lagged']) * 100
                    
                    # Get the latest date for which change can be calculated
                    latest_date_with_change = df_change_data.dropna(subset=['Price_Change_Pct'])['Date'].max()

                    if pd.isna(latest_date_with_change):
                        st.info(f"Not enough historical data to calculate {change_type} price change for {change_food_item}.")
                    else:
                        df_map_change_current = df_change_data[df_change_data['Date'] == latest_date_with_change].copy()
                        df_map_change_final = df_map_change_current.groupby('State')['Price_Change_Pct'].mean().reset_index()

                        if not df_map_change_final.empty:
                            fig_change_map = px.choropleth_mapbox(
                                df_map_change_final,
                                geojson=nigeria_geojson,
                                locations='State',
                                featureidkey="properties.NAME_1",
                                color='Price_Change_Pct',
                                color_continuous_scale="RdBu", # Red for increase, Blue for decrease
                                mapbox_style="carto-positron",
                                zoom=5, center={"lat": 9.0820, "lon": 8.6753},
                                opacity=0.7,
                                hover_name='State',
                                hover_data={'Price_Change_Pct': ':.2f%'},
                                title=f'{change_type} Price Change for {change_food_item} ({latest_date_with_change.strftime("%B %Y")})'
                            )
                            fig_change_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                            st.plotly_chart(fig_change_map, use_container_width=True)
                        else:
                            st.info(f"No price change data available for {change_food_item} for the selected period.")
                else:
                    st.info(f"No data available for {change_food_item} to calculate price change.")
            elif not nigeria_geojson:
                st.warning("Cannot display map: GeoJSON data not loaded.")
            else:
                st.info("Please select a food item for the price change map.")

            # 5. Multi-line Chart: Compare Price Trends Across Multiple States
            st.markdown("---")
            st.markdown("#### ✅ 5. Multi-line Chart: Compare Price Trends Across Multiple States")
            st.markdown("Shows **lines for 2-4 user-selected states for a selected food item**. This helps track how different regions are experiencing inflation differently.")

            multi_state_food_item = st.selectbox(
                "Select Food Item to Compare Across States:",
                st.session_state.capitalized_food_items,
                key="multi_state_food_select"
            )

            available_states_for_compare = sorted(food_data_explorer_filtered['State'].unique().tolist())
            selected_states_to_compare = st.multiselect(
                "Select 2-4 States to Compare:",
                available_states_for_compare,
                default=available_states_for_compare[:2] if len(available_states_for_compare) >= 2 else [],
                max_selections=4,
                key="multi_state_select"
            )

            if multi_state_food_item and len(selected_states_to_compare) >= 2:
                df_compare = food_data_explorer_filtered[
                    (food_data_explorer_filtered['Food_Item'] == multi_state_food_item) &
                    (food_data_explorer_filtered['State'].isin(selected_states_to_compare))
                ].copy()

                if not df_compare.empty:
                    df_compare_plot = df_compare.groupby(['Date', 'State'])['Price'].mean().reset_index()
                    
                    unit_for_display = WFP_UNITS_INFO.get(multi_state_food_item, "Unit N/A").replace("~", "")
                    y_axis_label = f'Price (Naira / {unit_for_display})' if unit_for_display != "Unit N/A" else 'Price (Naira)'

                    fig_multi_line = px.line(
                        df_compare_plot,
                        x='Date',
                        y='Price',
                        color='State',
                        title=f'Price Trends for {multi_state_food_item} Across Selected States',
                        labels={'Price': y_axis_label, 'Date': 'Date'},
                        hover_data={'Price': ':.2f', 'State': True}
                    )
                    fig_multi_line.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_multi_line, use_container_width=True)
                else:
                    st.info(f"No data available for {multi_state_food_item} in the selected states.")
            elif multi_state_food_item and len(selected_states_to_compare) < 2:
                st.info("Please select at least 2 states to compare their price trends.")
            else:
                st.info("Please select a food item and states to compare.")
