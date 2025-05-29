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
