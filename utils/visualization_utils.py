"""
Visualization utility functions for the Pediatric Clinical Trials app.
This module contains reusable visualization functions for creating charts and maps.
"""
import unicodedata
import re
import pandas as pd
from typing import Tuple, Optional, List, Dict
import plotly.express as px
import streamlit as st
from pandas.api.types import CategoricalDtype


# Global dictionary for common misspellings and aliases
SPELLING_CORRECTIONS = {
    # Montreal variations
    'montréal': 'montreal',
    'montr al': 'montreal',
    'montral': 'montreal',
    'montr al,': 'montreal',
    'mtl': 'montreal',
    'mont-royal': 'montreal',
    
    # Vancouver variations
    'west vancouver': 'vancouver',
    'vancouver bc': 'vancouver',
    'van': 'vancouver',
    
    # Toronto variations
    'tor': 'toronto',
    'gta': 'toronto',

    # Hamilton variations
    'hamitlon': 'hamilton',
    
    # Quebec variations
    'qc': 'quebec',
    'qc city': 'quebec city',

    # Sherbrooke variations
    'sherbrook': 'sherbrooke',

    # Edmonton variations
    'edmonton ab': 'edmonton',
    
    # Winnipeg variations
    'winnepeg': 'winnipeg',

    # Halifax variations
    'dalhousie': 'halifax',

    # St. John's variations
    "saint johns": "st john",
    "st johns": "st john",
    
    # Other variations: if the value is None, it means to ignore province names
    'ontario': None,
    'bc': None,
    'alberta': None,

    # For "quebec", if used in a province context, force the accented version for city lookup
    'quebec': 'québec'
}

# Global dictionary for city coordinates
CITY_COORDS = {
    # Ontario
    'toronto': {'lat': 43.651070, 'lon': -79.347015},
    'ottawa': {'lat': 45.424721, 'lon': -75.695000},
    'mississauga': {'lat': 43.589045, 'lon': -79.644119},
    'hamilton': {'lat': 43.255722, 'lon': -79.871101},
    'london': {'lat': 42.983612, 'lon': -81.249725},
    'kitchener': {'lat': 43.451290, 'lon': -80.492763},
    'windsor': {'lat': 42.3149, 'lon': -83.0364},
    'ajax': {'lat': 43.8509, 'lon': -79.0205},
    'markham': {'lat': 43.8561, 'lon': -79.3370},
    'waterloo': {'lat': 43.4643, 'lon': -80.5204},
    'richmond hill': {'lat': 43.8828, 'lon': -79.4403},
    'barrie': {'lat': 44.3894, 'lon': -79.6903},
    'newmarket': {'lat': 44.0582, 'lon': -79.4622},
    'oshawa': {'lat': 43.8971, 'lon': -78.8658},
    'oakville': {'lat': 43.4675, 'lon': -79.6877},
    'burlington': {'lat': 43.3255, 'lon': -79.7990},
    'niagara falls': {'lat': 43.0896, 'lon': -79.0849},
    'etobicoke': {'lat': 43.6205, 'lon': -79.5132},
    'thornhill': {'lat': 43.8161, 'lon': -79.4269},
    'brampton': {'lat': 43.7315, 'lon': -79.7624},
    'peterborough': {'lat': 44.3091, 'lon': -78.3197},
    'north bay': {'lat': 46.3091, 'lon': -79.4608},
    'guelph': {'lat': 43.5448, 'lon': -80.2482},
    'scarborough': {'lat': 43.7764, 'lon': -79.2318},
    'st. catharines': {'lat': 43.1594, 'lon': -79.2469},
    'st catharines': {'lat': 43.1594, 'lon': -79.2469},
    'thunder bay': {'lat': 48.3809, 'lon': -89.2477},
    'north york': {'lat': 43.7615, 'lon': -79.4111},
    'sarnia': {'lat': 42.9745, 'lon': -82.4066},
    'east york': {'lat': 43.6909, 'lon': -79.3352},
    'vaughan': {'lat': 43.8563, 'lon': -79.5085},
    'cobourg': {'lat': 43.9593, 'lon': -78.1677},
    'whitby': {'lat': 43.8975, 'lon': -78.9428},
    'kingston': {'lat': 44.230687, 'lon': -76.481323},
    'orillia': {'lat': 44.6087, 'lon': -79.4207},
    'woodbridge': {'lat': 43.7758, 'lon': -79.5992},
    'brantford': {'lat': 43.1394, 'lon': -80.2644},
    'caledon': {'lat': 43.8358, 'lon': -79.8661},
    'west hamilton': {'lat': 43.2555, 'lon': -79.9100},
    
    # Quebec
    'montreal': {'lat': 45.508888, 'lon': -73.561668},
    'quebec city': {'lat': 46.813878, 'lon': -71.207981},
    'quebec': {'lat': 46.813878, 'lon': -71.207981},
    'gatineau': {'lat': 45.476545, 'lon': -75.701271},
    'sherbrooke': {'lat': 45.404242, 'lon': -71.894917},
    'laval': {'lat': 45.606649, 'lon': -73.712409},
    'chicoutimi': {'lat': 48.4283, 'lon': -71.0582},
    'rimouski': {'lat': 48.4488, 'lon': -68.5236},
    'st-jérôme': {'lat': 45.7809, 'lon': -74.0036},
    'trois-rivières': {'lat': 46.3432, 'lon': -72.5430},
    'ste-foy': {'lat': 46.7805, 'lon': -71.2872},
    'saint-jean-sur-richelieu': {'lat': 45.3007, 'lon': -73.2574},
    'baie-saint-paul': {'lat': 47.4417, 'lon': -70.5042},
    'pointe-claire': {'lat': 45.4490, 'lon': -73.8166},
    'saint-eustache': {'lat': 45.5641, 'lon': -73.9035},
    'saguenay': {'lat': 48.4260, 'lon': -71.0725},
    'la malbaie': {'lat': 47.6549, 'lon': -70.1522},
    
    # British Columbia
    'vancouver': {'lat': 49.246292, 'lon': -123.116226},
    'victoria': {'lat': 48.428329, 'lon': -123.365868},
    'burnaby': {'lat': 49.248809, 'lon': -122.980507},
    'surrey': {'lat': 49.104431, 'lon': -122.801094},
    'kelowna': {'lat': 49.887952, 'lon': -119.496010},
    'new westminster': {'lat': 49.2057, 'lon': -122.9110},
    'penticton': {'lat': 49.4991, 'lon': -119.5937},
    'west vancouver': {'lat': 49.3690, 'lon': -123.1715},
    'whistler': {'lat': 50.1162, 'lon': -122.9535},
    'abbotsford': {'lat': 49.0504, 'lon': -122.3045},
    'nanaimo': {'lat': 49.1659, 'lon': -123.9401},
    'kamloops': {'lat': 50.6745, 'lon': -120.3273},
    'cranbrook': {'lat': 49.5097, 'lon': -115.7663},
    'richmond': {'lat': 49.1666, 'lon': -123.1336},
    'fort st. james': {'lat': 54.4438, 'lon': -124.2542},
    'grand forks': {'lat': 49.0313, 'lon': -118.4447},
    'canal flats': {'lat': 50.1513, 'lon': -115.8319},
    'chetwynd': {'lat': 55.6978, 'lon': -121.6298},
    'cariboo': {'lat': 52.9283, 'lon': -122.4461},
    'lytton': {'lat': 50.2337, 'lon': -121.5820},
    
    # Alberta
    'calgary': {'lat': 51.044270, 'lon': -114.062019},
    'edmonton': {'lat': 53.544388, 'lon': -113.490929},
    'sherwood park': {'lat': 53.5413, 'lon': -113.2958},
    'red deer': {'lat': 52.269, 'lon': -113.811},
    'lethbridge': {'lat': 49.6956, 'lon': -112.8451},
    'canmore': {'lat': 51.0884, 'lon': -115.3475},
    'st. albert': {'lat': 53.6322, 'lon': -113.6275},
    'grande prairie': {'lat': 55.1707, 'lon': -118.7947},
    'banff': {'lat': 51.1784, 'lon': -115.5708},
    'camrose': {'lat': 53.0216, 'lon': -112.8335},
    'medicine hat': {'lat': 50.0405, 'lon': -110.6768},
    
    # Manitoba
    'winnipeg': {'lat': 49.895077, 'lon': -97.138451},
    
    # Saskatchewan
    'saskatoon': {'lat': 52.131802, 'lon': -106.655808},
    'regina': {'lat': 50.445210, 'lon': -104.618896},
    
    # Nova Scotia
    'halifax': {'lat': 44.648618, 'lon': -63.586002},
    
    # New Brunswick
    'saint john': {'lat': 45.273220, 'lon': -66.063308},
    'st. john': {'lat': 45.273220, 'lon': -66.063308},
    'moncton': {'lat': 46.090946, 'lon': -64.790306},
    'fredericton': {'lat': 45.9636, 'lon': -66.6431},
    
    # Newfoundland and Labrador
    "st. john's": {'lat': 47.561510, 'lon': -52.712577},
    
    # Other locations/special cases
    'windermere': {'lat': 43.6631, 'lon': -79.4749},
    'winchester': {'lat': 45.0842, 'lon': -75.3500},
    'alliston': {'lat': 44.1502, 'lon': -79.8682},
    'coburg': {'lat': 43.9593, 'lon': -78.1677},
}

def normalize_city_name(city_name: str) -> str:
    """
    Normalize a city name by lower-casing, removing accents and punctuation,
    and then applying spelling corrections. Also, remove trailing 's' from
    names like 'johns' so that variants such as "saint john's" and "st john's"
    become a common form.
    """
    if not city_name:
        return ""
    
    # Lowercase and trim spaces
    city = city_name.lower().strip()
    
    # Remove accents
    city = unicodedata.normalize('NFD', city)
    city = ''.join(c for c in city if not unicodedata.combining(c))
    
    # Remove punctuation (e.g. apostrophes) and extra spaces
    city = re.sub(r'[^\w\s]', '', city)
    city = re.sub(r'\s+', ' ', city).strip()
    
    # Apply spelling corrections based on normalized (punctuation-free) key.
    # For example, ensure that both "saint johns" and "st johns" are mapped
    # to a common form.
    if city in SPELLING_CORRECTIONS:
        correction = SPELLING_CORRECTIONS[city]
        if correction:
            city = correction
    
    # Optionally, remove a trailing 's' from names like 'johns' if that seems appropriate.
    # For example, turn "saint johns" or "st johns" into "st john".
    # (You can adjust the regex if you have other specific rules.)
    city = re.sub(r'(?<=\bjohn)s\b', '', city).strip()
    
    # If the city name starts with common prefixes like "st", "st.", or "saint",
    # check if the corresponding name exists in the city coordinates.
    # (Assumes CITY_COORDS contains the preferred normalized version.)
    for prefix in ['saint', 'st', 'st']:
        if city.startswith(prefix + ' '):
            base = city.split(' ', 1)[1]
            if f"saint {base}" in CITY_COORDS:
                return f"saint {base}"
            if f"st {base}" in CITY_COORDS:
                return f"st {base}"
    return city


def plot_geographic_map(facilities_df: pd.DataFrame) -> Tuple[Optional[px.scatter_mapbox], List[str]]:
    """
    Create a map visualization showing the distribution of trials across Canada.
    
    Args:
        facilities_df: DataFrame containing facility information with a 'city' column 
                       and 'nct_id' for trial IDs.
        
    Returns:
        Tuple of (plotly scatter_mapbox figure, list of normalized cities missing coordinates)
    """
    # Validate required columns
    if facilities_df.empty or 'city' not in facilities_df.columns or 'nct_id' not in facilities_df.columns:
        return None, []
    
    # Build a mapping of normalized city names to coordinates using the global CITY_COORDS
    normalized_city_coords = {}
    for city_name, coords in CITY_COORDS.items():
        norm_name = normalize_city_name(city_name)
        if norm_name:
            normalized_city_coords[norm_name] = coords

    # Add a normalized city column to the facilities data
    facilities_df = facilities_df.copy()
    facilities_df['norm_city'] = facilities_df['city'].apply(lambda x: normalize_city_name(x))
    
    # Aggregate counts by normalized city name
    city_counts = facilities_df.groupby('norm_city')['nct_id'].nunique().reset_index()
    city_counts.columns = ['norm_city', 'trial_count']
    city_counts = city_counts.sort_values('trial_count', ascending=False)
    
    cities_without_coords = []
    cities_with_coords = []
    
    def get_coords(norm_city: str) -> Tuple[Dict[str, float], bool]:
        if norm_city in normalized_city_coords:
            return normalized_city_coords[norm_city], True
        # Try checking against the spelling corrections (if part of the key)
        for variant, corrected in SPELLING_CORRECTIONS.items():
            if variant in norm_city or norm_city in variant:
                if corrected and corrected in CITY_COORDS:
                    return CITY_COORDS[corrected], True
        cities_without_coords.append(norm_city)
        # Default coordinate: center of Canada
        return {'lat': 56.130366, 'lon': -106.346771}, False
    
    for _, row in city_counts.iterrows():
        norm_city = row['norm_city']
        coords, found = get_coords(norm_city)
        cities_with_coords.append({
            'city': norm_city,
            'trial_count': row['trial_count'],
            'lat': coords['lat'],
            'lon': coords['lon'],
            'found': found
        })
    
    map_df = pd.DataFrame(cities_with_coords)
    map_df_for_viz = map_df[map_df['found']].copy()
    
    fig_map = px.scatter_mapbox(
        map_df_for_viz,
        lat="lat", 
        lon="lon",
        size="trial_count",
        color="trial_count",
        color_continuous_scale="Viridis",
        hover_name="city",
        hover_data={"lat": False, "lon": False, "trial_count": True},
        title="Canadian Cities with Pediatric Clinical Trials",
        size_max=35,
        zoom=3,
    )
    fig_map.update_layout(
        height=600,
        mapbox_style="carto-positron",
    )
    
    return fig_map, sorted(list(set(cities_without_coords)))

def render_city_visualization(filtered_df: pd.DataFrame, conn: any, facilities_df: Optional[pd.DataFrame] = None):
    """
    Render the city visualization section with a map and a bar chart.
    
    Args:
        filtered_df: DataFrame with filtered trial data.
        conn: Database connection for fetching additional data if needed.
        facilities_df: Optional pre-fetched facilities data.
    """
    # Import the database utility function to avoid circular imports
    from utils.database_utils import fetch_facilities_for_trials
    
    st.subheader("Geographic Distribution")
    
    # Get the list of filtered trial IDs - these are the trials that match our current filters
    filtered_nct_ids = filtered_df['nct_id'].tolist()
    
    if not filtered_nct_ids:
        st.warning("No trials match the current filters.")
        return
        
    # FIX: Always fetch fresh facility data for the FILTERED trials 
    # instead of using the pre-fetched facilities_df which might contain ALL trials
    with st.spinner("Loading geographic data..."):
        filtered_facilities_df = fetch_facilities_for_trials(conn, filtered_nct_ids)
    
    if filtered_facilities_df is not None and not filtered_facilities_df.empty:
        # Create map visualization using normalized city names
        fig_map, missing_cities = plot_geographic_map(filtered_facilities_df)
        
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
            st.info("Cities without precise coordinates are excluded from the map. The size of each point represents the number of trials in that city.")
            
            # For the bar chart, aggregate trials by normalized city names
            filtered_facilities_df = filtered_facilities_df.copy()
            filtered_facilities_df['norm_city'] = filtered_facilities_df['city'].apply(lambda x: normalize_city_name(x))
            
            # FIX: Debug information to verify counts
            with st.expander("Debug Information"):
                total_unique_trials = filtered_facilities_df['nct_id'].nunique()
                st.write(f"Total unique trials in filtered set: {total_unique_trials}")
                
                # Get count for Edmonton specifically
                edmonton_mask = filtered_facilities_df['norm_city'] == 'edmonton'
                edmonton_trial_count = filtered_facilities_df[edmonton_mask]['nct_id'].nunique()
                st.write(f"Edmonton unique trial count: {edmonton_trial_count}")
                
                # Show sample of the data
                st.write("Sample of filtered facilities data:")
                st.dataframe(filtered_facilities_df.head(10))
            
            # Continue with the visualization
            city_counts = filtered_facilities_df.groupby('norm_city')['nct_id'].nunique().reset_index()
            city_counts.columns = ['city', 'trial_count']
            city_counts = city_counts.sort_values('trial_count', ascending=True)
            
            top_n = min(10, len(city_counts))
            top_cities = city_counts.tail(top_n)
            
            fig_cities = px.bar(
                top_cities,
                y='city',
                x='trial_count',
                orientation='h',
                title=f'Top {top_n} Canadian Cities by Number of Pediatric Clinical Trials',
                labels={'city': 'City', 'trial_count': 'Number of Trials'},
                color='trial_count',
                color_continuous_scale='Viridis',
            )
            
            fig_cities.update_layout(
                height=max(400, top_n * 25),
                xaxis_title="Number of Trials",
                yaxis_title="City",
                yaxis={'categoryorder':'total ascending'},
                template="plotly_white"
            )
            
            st.plotly_chart(fig_cities, use_container_width=True)
            
            if missing_cities:
                with st.expander(f"Cities Without Coordinates ({len(missing_cities)})", expanded=False):
                    cols = st.columns(3)
                    chunk_size = (len(missing_cities) + 2) // 3
                    for i, city_chunk in enumerate([missing_cities[i:i + chunk_size] for i in range(0, len(missing_cities), chunk_size)]):
                        if i < len(cols):
                            with cols[i]:
                                for city in city_chunk:
                                    st.write(f"• {city}")
        else:
            st.warning("Unable to create map visualization.")
    else:
        st.warning("No geographic data available for the selected trials.")

def plot_geographic_map_alberta(facilities_df: pd.DataFrame) -> Tuple[Optional[px.scatter_mapbox], List[str]]:
    """
    Create a map visualization showing the distribution of trials 
    across Alberta (filtered by known Alberta city names).
    
    Args:
        facilities_df: DataFrame with 'city' and 'nct_id' columns.
        
    Returns:
        Tuple of (plotly scatter_mapbox figure, list of missing city coords).
    """
    # Validate required columns
    if facilities_df.empty or 'city' not in facilities_df.columns or 'nct_id' not in facilities_df.columns:
        return None, []
    
    # A set (or list) of normalized city names in Alberta
    # Make sure these match the normalized keys you have in CITY_COORDS.
    alberta_cities = {
        "calgary",
        "edmonton",
        "red deer",
        "lethbridge",
        "canmore",
        "st. albert",
        "grande prairie",
        "banff",
        "camrose",
        "medicine hat",
    }
    
    # Build a mapping of normalized city names to coordinates
    normalized_city_coords = {}
    for city_name, coords in CITY_COORDS.items():
        norm_name = normalize_city_name(city_name)
        if norm_name:
            normalized_city_coords[norm_name] = coords

    # Copy DataFrame to avoid mutating original
    facilities_df = facilities_df.copy()
    # Normalize city names
    facilities_df['norm_city'] = facilities_df['city'].apply(lambda x: normalize_city_name(x))

    # **Filter** for only Alberta cities
    alberta_df = facilities_df[facilities_df['norm_city'].isin(alberta_cities)]
    if alberta_df.empty:
        return None, []

    # Count how many trials per city
    city_counts = alberta_df.groupby('norm_city')['nct_id'].nunique().reset_index()
    city_counts.columns = ['norm_city', 'trial_count']
    city_counts = city_counts.sort_values('trial_count', ascending=False)
    
    # Prepare to track missing coords
    missing_cities = []
    cities_with_coords = []
    
    def get_coords(norm_city: str) -> Tuple[Dict[str, float], bool]:
        if norm_city in normalized_city_coords:
            return normalized_city_coords[norm_city], True
        # If we have a partial match in SPELLING_CORRECTIONS, try that
        for variant, corrected in SPELLING_CORRECTIONS.items():
            if variant in norm_city or norm_city in variant:
                if corrected and corrected in CITY_COORDS:
                    return CITY_COORDS[corrected], True
        # Otherwise, record as missing
        missing_cities.append(norm_city)
        # Default to center of Canada (or pick a center of Alberta if you prefer)
        return {'lat': 56.130366, 'lon': -106.346771}, False
    
    # Collect city coords
    for _, row in city_counts.iterrows():
        norm_city = row['norm_city']
        coords, found = get_coords(norm_city)
        cities_with_coords.append({
            'city': norm_city,
            'trial_count': row['trial_count'],
            'lat': coords['lat'],
            'lon': coords['lon'],
            'found': found
        })

    # Build a DataFrame of the Alberta cities with coords
    map_df = pd.DataFrame(cities_with_coords)
    map_df_for_viz = map_df[map_df['found']].copy()
    
    # Create the Plotly scatter_mapbox
    # Center near the middle of Alberta, with a higher zoom for a closer view.
    fig_map = px.scatter_mapbox(
        map_df_for_viz,
        lat="lat", 
        lon="lon",
        size="trial_count",
        color="trial_count",
        color_continuous_scale="Viridis",
        hover_name="city",
        hover_data={"lat": False, "lon": False, "trial_count": True},
        title="Alberta Cities with Pediatric Clinical Trials",
        size_max=35,
        zoom=5,  # Tighter zoom for Alberta
    )
    fig_map.update_layout(
        height=600,
        mapbox_style="carto-positron",
        mapbox_center={"lat": 53.9333, "lon": -116.5765},  # Rough center of Alberta
    )
    
    return fig_map, sorted(list(set(missing_cities)))

def render_city_visualization_alberta(filtered_df: pd.DataFrame, conn: any, facilities_df: Optional[pd.DataFrame] = None):
    """
    Render the Alberta-specific city visualization with a map and bar chart.
    
    Args:
        filtered_df: DataFrame with filtered trial data.
        conn: Database connection for fetching additional data if needed.
        facilities_df: Optional pre-fetched facilities data.
    """
    # Import the database utility function to avoid circular imports
    from utils.database_utils import fetch_facilities_for_trials
    
    st.subheader("Alberta Geographic Distribution")
    
    # Get the list of filtered trial IDs
    filtered_nct_ids = filtered_df['nct_id'].tolist()
    
    if not filtered_nct_ids:
        st.warning("No trials match the current filters.")
        return
    
    # FIX: Always fetch fresh facility data for the FILTERED trials
    with st.spinner("Loading geographic data for Alberta..."):
        filtered_facilities_df = fetch_facilities_for_trials(conn, filtered_nct_ids)
    
    if filtered_facilities_df is not None and not filtered_facilities_df.empty:
        # Define Alberta cities for filtering
        alberta_cities = {
            "calgary",
            "edmonton",
            "red deer",
            "lethbridge",
            "canmore",
            "st. albert",
            "grande prairie", 
            "banff",
            "camrose",
            "medicine hat",
        }
        
        # Add normalized city names
        filtered_facilities_df = filtered_facilities_df.copy()
        filtered_facilities_df['norm_city'] = filtered_facilities_df['city'].apply(lambda x: normalize_city_name(x))
        
        # Filter for Alberta cities
        alberta_facilities = filtered_facilities_df[filtered_facilities_df['norm_city'].isin(alberta_cities)]
        
        if alberta_facilities.empty:
            st.warning("No Alberta city data found for these trials.")
        else:
            # Create the Alberta-specific map
            fig_map_ab, missing_cities_ab = plot_geographic_map_alberta(filtered_facilities_df)
            
            if fig_map_ab:
                st.plotly_chart(fig_map_ab, use_container_width=True)
                st.info("Cities without precise coordinates are excluded from the Alberta map. The size of each point represents the number of trials in that city.")
                
                # Create bar chart for Alberta cities
                city_counts = alberta_facilities.groupby('norm_city')['nct_id'].nunique().reset_index()
                city_counts.columns = ['city', 'trial_count']
                city_counts = city_counts.sort_values('trial_count', ascending=True)
                
                # FIX: Debug information to verify counts
                with st.expander("Debug Information"):
                    total_unique_trials = alberta_facilities['nct_id'].nunique()
                    st.write(f"Total unique trials in Alberta: {total_unique_trials}")
                    
                    # Show count for each Alberta city
                    for city in alberta_cities:
                        city_mask = alberta_facilities['norm_city'] == city
                        city_trial_count = alberta_facilities[city_mask]['nct_id'].nunique()
                        if city_trial_count > 0:
                            st.write(f"{city.title()} unique trial count: {city_trial_count}")
                    
                    # Show sample of the data
                    st.write("Sample of Alberta facilities data:")
                    st.dataframe(alberta_facilities.head(10))
                
                fig_cities = px.bar(
                    city_counts,
                    y='city',
                    x='trial_count',
                    orientation='h',
                    title='Alberta Cities by Number of Pediatric Clinical Trials',
                    labels={'city': 'City', 'trial_count': 'Number of Trials'},
                    color='trial_count',
                    color_continuous_scale='Viridis',
                )
                
                fig_cities.update_layout(
                    height=max(350, len(city_counts) * 40),
                    xaxis_title="Number of Trials",
                    yaxis_title="City",
                    yaxis={'categoryorder':'total ascending'},
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_cities, use_container_width=True)
                
                # If there are missing cities, display them in an expander
                if missing_cities_ab:
                    with st.expander(f"Cities Without Coordinates in Alberta ({len(missing_cities_ab)})", expanded=False):
                        for city in missing_cities_ab:
                            st.write(f"• {city}")
            else:
                st.warning("Unable to create Alberta map visualization.")
    else:
        st.warning("No geographic data available for the selected trials.")

    
def plot_phase_distribution(df: pd.DataFrame) -> px.bar:
    """
    Create a bar chart showing the distribution of trials by phase,
    enforcing the exact custom order via Pandas categorical.
    """
    # 1) Replace NA variants and unify your labels
    df = df.copy()
    df['phase'] = (
        df['phase']
        .fillna('Not Applicable')
        .replace({'NA': 'Not Applicable'})
        # Example of unifying underscores/spaces, if needed:
        # .replace({'PHASE_1': 'PHASE1', 'PHASE 1': 'PHASE1'})
    )
    
    # 2) Exclude Not Applicable if desired (remove the line below if you want to keep it)
    df = df[df['phase'] != 'Not Applicable']
    
    # 3) Count occurrences
    phase_counts = df['phase'].value_counts().reset_index()
    phase_counts.columns = ['phase', 'count']
    
    # 4) Your custom order must match EXACTLY what appears in 'phase_counts["phase"]'
    custom_phase_order = [
        "EARLY_PHASE1",
        "PHASE1",
        "PHASE1/2",
        "PHASE2",
        "PHASE2/3",
        "PHASE3",
        "PHASE4",
        "Not Applicable"
    ]
    
    # 5) Convert to a categorical dtype with the custom order
    from pandas.api.types import CategoricalDtype
    cat_type = CategoricalDtype(categories=custom_phase_order, ordered=True)
    phase_counts['phase'] = phase_counts['phase'].astype(cat_type)
    
    # 6) Sort by the categorical order
    phase_counts.sort_values('phase', inplace=True)
    
    # 7) Build the figure
    n = phase_counts['count'].sum()
    fig = px.bar(
        phase_counts,
        x='phase',
        y='count',
        labels={'phase': 'Trial Phase', 'count': 'Number of Trials'},
        title=f"Distribution of Pediatric Trials by Phase (n = {n} Trials Reporting Phase)",
        color='phase',
        # category_orders won't matter if we already sorted the data, 
        # but you can still include it for safety:
        category_orders={"phase": custom_phase_order},
    )
    
    fig.update_layout(template="plotly_white")
    return fig

def plot_status_distribution(df: pd.DataFrame) -> px.pie:
    """
    Create a pie chart showing the distribution of trials by status.
    
    Args:
        df: DataFrame containing pediatric trial data
        
    Returns:
        Plotly pie chart figure
    """
    status_counts = df['overall_status'].value_counts().reset_index()
    status_counts.columns = ['status', 'count']
    
    fig = px.pie(
        status_counts,
        values='count',
        names='status',
        title='Distribution of Trials by Status',
        hole=0.4
    )
    return fig


def plot_yearly_trends(df: pd.DataFrame) -> Tuple[px.bar, px.area]:
    """
    Create charts showing trial trends by year and status.
    
    Args:
        df: DataFrame containing pediatric trial data
        
    Returns:
        Tuple of (bar chart figure, area chart figure)
    """
    if 'start_year' not in df.columns or df['start_year'].isna().all():
        return None, None
    
    year_status = df.groupby(['start_year', 'overall_status']).size().reset_index(name='count')
    year_status = year_status[~year_status['start_year'].isna()]
    
    # Bar chart by year and status
    fig_bar = px.bar(
        year_status,
        x='start_year',
        y='count',
        color='overall_status',
        labels={'start_year': 'Start Year', 'count': 'Number of Trials', 'overall_status': 'Status'},
        title='Pediatric Clinical Trials by Start Year and Status'
    )
    fig_bar.update_layout(
        xaxis_title="Start Year",
        yaxis_title="Number of Trials",
        barmode="group",
        template="plotly_white"
    )
    
    # Stacked area chart for year trends
    fig_area = px.area(
        year_status,
        x='start_year',
        y='count',
        color='overall_status',
        labels={'start_year': 'Start Year', 'count': 'Number of Trials', 'overall_status': 'Status'},
        title='Cumulative Pediatric Trials by Year and Status'
    )
    fig_area.update_layout(template="plotly_white")
    
    return fig_bar, fig_area


def plot_trial_age_distribution(df: pd.DataFrame) -> px.histogram:
    """
    Create a histogram showing the distribution of minimum ages in trials.
    
    Args:
        df: DataFrame containing pediatric trial data
        
    Returns:
        Plotly histogram figure
    """
    fig = px.histogram(
        df,
        x='age_in_months',
        labels={'age_in_months': 'Minimum Age (months)', 'count': 'Number of Trials'},
        title='Distribution of Trials by Minimum Age',
        nbins=20
    )
    fig.update_layout(template="plotly_white")
    
    # Add vertical lines for common pediatric age boundaries
    boundaries = [
        (0, "Birth"),
        (1, "1 month"),
        (12, "1 year"),
        (24, "2 years"),
        (72, "6 years"),
        (144, "12 years"),
        (192, "16 years")
    ]
    
    for value, label in boundaries:
        fig.add_vline(
            x=value, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=label,
            annotation_position="top"
        )
    
    return fig

def render_summary_metrics(df: pd.DataFrame):
    """
    Render summary metrics for the pediatric trials data.
    
    Args:
        df: DataFrame containing pediatric trial data
    """
    total_trials = len(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pediatric Trials", total_trials)
    with col2:
        active_trials = df[df['overall_status'].isin(['RECRUITING', 'ACTIVE, NOT RECRUITING', 'ENROLLING BY INVITATION'])].shape[0]
        st.metric("Active Trials", active_trials)
    with col3:
        recruiting_trials = df[df['overall_status'] == 'RECRUITING'].shape[0]
        st.metric("Recruiting Trials", recruiting_trials)


def render_trial_details(df: pd.DataFrame, conn, keywords_dict=None, conditions_dict=None):
    """
    Render detailed information for a selected trial.
    
    Args:
        df: DataFrame containing pediatric trial data
        conn: Database connection for fetching additional data
        keywords_dict: Dictionary mapping NCT IDs to keywords
        conditions_dict: Dictionary mapping NCT IDs to conditions
    """
    # Import the necessary function here to avoid circular imports
    from utils.database_utils import fetch_interventions_for_trials
    
    # Select a trial to show details
    selected_trial = st.selectbox(
        "Select a trial to view details:",
        options=df['nct_id'].tolist(),
        format_func=lambda nct_id: f"{nct_id} - {df[df['nct_id'] == nct_id]['brief_title'].iloc[0][:50]}..."
    )
    
    if selected_trial:
        st.write("### Trial Details")
        trial_data = df[df['nct_id'] == selected_trial].iloc[0]
        
        st.write(f"**NCT ID:** {selected_trial}")
        st.write(f"**Title:** {trial_data['brief_title']}")
        if pd.notna(trial_data['official_title']):
            st.write(f"**Official Title:** {trial_data['official_title']}")
        st.write(f"**Status:** {trial_data['overall_status']}")
        st.write(f"**Phase:** {trial_data['phase'] if pd.notna(trial_data['phase']) else 'Not Applicable'}")
        st.write(f"**Minimum Age:** {trial_data['minimum_age']}")
        st.write(f"**Canadian Sites:** {int(trial_data['num_canadian_sites'])}")
        
        # Brief summary
        if pd.notna(trial_data['brief_summary']):
            with st.expander("Brief Summary", expanded=True):
                st.write(trial_data['brief_summary'])
        
        # Detailed description
        if pd.notna(trial_data['detailed_description']):
            with st.expander("Detailed Description"):
                st.write(trial_data['detailed_description'])
        
        # Display conditions
        if conditions_dict and selected_trial in conditions_dict:
            conditions = conditions_dict[selected_trial]
            if conditions:
                st.write("**Conditions:**")
                st.write(", ".join(conditions))
        
        # Display keywords
        if keywords_dict and selected_trial in keywords_dict:
            keywords = keywords_dict[selected_trial]
            if keywords:
                st.write("**Keywords:**")
                st.write(", ".join(keywords))
        
        # Fetch and display interventions
        interventions_df = fetch_interventions_for_trials(conn, [selected_trial])
        if not interventions_df.empty:
            st.write("**Interventions:**")
            for _, row in interventions_df.iterrows():
                st.write(f"- {row['intervention_type']}: {row['name']}")
        
        # Link to ClinicalTrials.gov
        st.markdown(f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{selected_trial})")
        