"""
Visualization utility functions for the Clinical Trials app.
This module contains reusable visualization functions for creating charts and maps.
"""
import unicodedata
import re
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import plotly.express as px
import streamlit as st
from pandas.api.types import CategoricalDtype

# Population data for Canadian cities (based on recent census data)
CITY_POPULATIONS = {
    # Ontario
    'toronto': 2731571,
    'ottawa': 934243,
    'mississauga': 721599,
    'brampton': 593638,
    'hamilton': 536917,
    'london': 383822,
    'markham': 328966,
    'vaughan': 306233,
    'kitchener': 233222,
    'windsor': 217188,
    'richmond hill': 195022,
    'oakville': 193832,
    'burlington': 183314,
    'oshawa': 159458,
    'barrie': 141434,
    'st. catharines': 133113,
    'guelph': 131794,
    'cambridge': 129920,
    'whitby': 128377,
    'kingston': 117660,
    'thunder bay': 108843,
    'waterloo': 104986,
    'milton': 110128,
    'ajax': 119677,
    'newmarket': 84224,
    'peterborough': 81032,
    'sarnia': 71594,
    'north bay': 51553,
    'orillia': 31166,
    'cobourg': 19440,
    'caledon': 76581,
    'brantford': 134203,
    'east york': 118071,  # Part of Toronto
    'north york': 691600,  # Part of Toronto
    'scarborough': 632098,  # Part of Toronto
    'etobicoke': 365143,  # Part of Toronto
    'thornhill': 112719,  # Part of Vaughan/Markham
    'woodbridge': 105228,  # Part of Vaughan
    
    # Quebec
    'montreal': 1704694,
    'quebec city': 531902,
    'quebec': 531902,  # Same as quebec city
    'laval': 422993,
    'gatineau': 276245,
    'longueuil': 239700,
    'sherbrooke': 161323,
    'saguenay': 144230,
    'trois-rivières': 134413,
    'chicoutimi': 66547,  # Part of Saguenay
    'rimouski': 47006,
    'st-jérôme': 77301,
    'ste-foy': 98659,  # Part of Quebec City
    'saint-jean-sur-richelieu': 92394,
    'baie-saint-paul': 7146,
    'pointe-claire': 31380,
    'saint-eustache': 43784,
    'la malbaie': 8271,
    
    # British Columbia
    'vancouver': 2643000, # Changed from 631486 to population of Greater Vancouver Area (TODO: will need to remove other cities)
    'surrey': 518467,
    'burnaby': 232755,
    'richmond': 198309,
    'abbotsford': 141397,
    'coquitlam': 139284,
    'kelowna': 142146,
    'victoria': 85792,
    'nanaimo': 90504,
    'kamloops': 90280,
    'chilliwack': 83788,
    'new westminster': 70996,
    'west vancouver': 42473,
    'penticton': 33761,
    'whistler': 11854,
    'cranbrook': 20499,
    'fort st. james': 1598,
    'grand forks': 4049,
    'canal flats': 668,
    'chetwynd': 2503,
    'lytton': 249,
    
    # Alberta
    'calgary': 1239220,
    'edmonton': 932546,
    'red deer': 100418,
    'lethbridge': 92729,
    'st. albert': 65589,
    'medicine hat': 63271,
    'grande prairie': 63166,
    'sherwood park': 70618,  # Part of Strathcona County
    'canmore': 13992,
    'banff': 7847,
    'camrose': 18742,
    
    # Manitoba
    'winnipeg': 705244,
    
    # Saskatchewan
    'saskatoon': 273010,
    'regina': 228928,
    
    # Nova Scotia
    'halifax': 403131,
    
    # New Brunswick
    'saint john': 67575,
    'st. john': 67575,  # Same as saint john
    'moncton': 71889,
    'fredericton': 58220,
    
    # Newfoundland and Labrador
    "st. john's": 108860,
    
    # Special cases
    'windermere': 2800,  # Part of Vaughan
    'winchester': 2394,
    'alliston': 19243,  # Part of New Tecumseth
    'coburg': 19440,  # Same as Cobourg
}

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

# Define metro area groupings
METRO_VANCOUVER_CITIES = {
    'vancouver',
    'burnaby',
    'coquitlam',
    'delta',
    'langley',  # Covers both City and District
    'langley city',
    'langley district',
    'maple ridge',
    'new westminster',
    'north vancouver',  # Covers both City and District
    'north vancouver city',
    'north vancouver district',
    'pitt meadows',
    'port coquitlam',
    'port moody',
    'richmond',
    'surrey',
    'white rock',
    'west vancouver',
    'bowen island',
    'anmore',
    'belcarra',
    # Add common spelling variations
    'west van',
    'north van',
    'poco',  # Common abbreviation for Port Coquitlam
    'new west',  # Common abbreviation for New Westminster
}

GREATER_TORONTO_AREA_CITIES = {
    # Toronto (Core)
    'toronto',
    'north york',
    'scarborough',
    'etobicoke',
    'east york',
    'york',
    
    # Peel Region
    'mississauga',
    'brampton',
    'caledon',
    
    # York Region
    'vaughan',
    'markham',
    'richmond hill',
    'aurora',
    'newmarket',
    'king',
    'whitchurch-stouffville',
    'stouffville',
    'east gwillimbury',
    'georgina',
    
    # Durham Region
    'pickering',
    'ajax',
    'whitby',
    'oshawa',
    'clarington',
    'uxbridge',
    'scugog',
    'port perry',
    'brock',
    'beaverton',
    'cannington',
    'sunderland',
    
    # Halton Region
    'oakville',
    'burlington',
    'milton',
    'halton hills',
    'georgetown',
    'acton',
    
    # Common variations and abbreviations
    'gta',
    'the 6',
    'the six',
    'tdot',
    'the dot',
    't.o.',
    'to',
}

def metropolitan_area_mapping(city_name: str) -> str:
    """
    Map a normalized city name to its metropolitan area if applicable.
    
    Args:
        city_name: Normalized city name
        
    Returns:
        Metropolitan area name or original city name if not in a defined metro area
    """
    if city_name in METRO_VANCOUVER_CITIES:
        return 'metro vancouver'
    elif city_name in GREATER_TORONTO_AREA_CITIES:
        return 'greater toronto area'
    else:
        return city_name


def normalize_city_name(city_name: str, apply_metro_mapping: bool = True) -> str:
    """
    Normalize a city name by lower-casing, removing accents and punctuation,
    and then applying spelling corrections. Also, remove trailing 's' from
    names like 'johns' so that variants such as "saint john's" and "st john's"
    become a common form. Optionally map to metropolitan area.
    
    Args:
        city_name: City name to normalize
        apply_metro_mapping: Whether to apply metropolitan area mapping (default: True)
        
    Returns:
        Normalized city name or metropolitan area name
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
    if city in SPELLING_CORRECTIONS:
        correction = SPELLING_CORRECTIONS[city]
        if correction:
            city = correction
    
    # Optionally, remove a trailing 's' from names like 'johns' if that seems appropriate.
    city = re.sub(r'(?<=\bjohn)s\b', '', city).strip()
    
    # If the city name starts with common prefixes like "st", "st.", or "saint",
    # check if the corresponding name exists in the city coordinates.
    for prefix in ['saint', 'st', 'st']:
        if city.startswith(prefix + ' '):
            base = city.split(' ', 1)[1]
            if f"saint {base}" in CITY_COORDS:
                city = f"saint {base}"
            if f"st {base}" in CITY_COORDS:
                city = f"st {base}"
    
    # Apply metropolitan area mapping if requested
    if apply_metro_mapping:
        return metropolitan_area_mapping(city)
    
    return city


def plot_geographic_map(facilities_df: pd.DataFrame) -> Tuple[Optional[px.scatter_mapbox], List[str]]:
    """
    Create a map visualization showing the distribution of trials across Canada.
    Cities within the Greater Vancouver Area and Greater Toronto Area are consolidated.
    
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
        # Use normalize_city_name with apply_metro_mapping=False to get the original normalized name
        norm_name = normalize_city_name(city_name, apply_metro_mapping=False)
        if norm_name:
            normalized_city_coords[norm_name] = coords
    
    # Add metro area coordinates (using central city coordinates)
    normalized_city_coords['metro vancouver'] = CITY_COORDS['vancouver']
    normalized_city_coords['greater toronto area'] = CITY_COORDS['toronto']

    # Add a normalized city column to the facilities data, applying metro area mapping
    facilities_df = facilities_df.copy()
    facilities_df['norm_city'] = facilities_df['city'].apply(lambda x: normalize_city_name(x, apply_metro_mapping=True))
    
    # Aggregate counts by normalized city name (now including metro areas)
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
        
        # Special handling for metro areas
        if norm_city == 'metro vancouver':
            return CITY_COORDS['vancouver'], True
        elif norm_city == 'greater toronto area':
            return CITY_COORDS['toronto'], True
            
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
    
    # For display purposes, capitalize city names (including metro areas)
    map_df_for_viz['city'] = map_df_for_viz['city'].apply(
        lambda x: ' '.join(word.capitalize() for word in x.split())
    )
    
    fig_map = px.scatter_mapbox(
        map_df_for_viz,
        lat="lat", 
        lon="lon",
        size="trial_count",
        color="trial_count",
        color_continuous_scale="Viridis",
        hover_name="city",
        hover_data={"lat": False, "lon": False, "trial_count": True},
        title="Canadian Cities with Clinical Trials (Metro Areas Consolidated)",
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
    Cities within the Greater Vancouver Area and Greater Toronto Area are consolidated.
    
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
        
    # Always fetch fresh facility data for the FILTERED trials 
    with st.spinner("Loading geographic data..."):
        filtered_facilities_df = fetch_facilities_for_trials(conn, filtered_nct_ids)
    
    if filtered_facilities_df is not None and not filtered_facilities_df.empty:
        # Create map visualization using normalized city names (with metro area mapping)
        fig_map, missing_cities = plot_geographic_map(filtered_facilities_df)
        
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
            st.info("Cities within the Greater Toronto Area and Metro Vancouver have been consolidated. The size of each point represents the number of trials in that city or metropolitan area.")
            
            # For the bar chart, aggregate trials by normalized city names (with metro area mapping)
            filtered_facilities_df = filtered_facilities_df.copy()
            filtered_facilities_df['norm_city'] = filtered_facilities_df['city'].apply(
                lambda x: normalize_city_name(x, apply_metro_mapping=True)
            )
            
            # Optional debug information
            with st.expander("Debug Information", expanded=False):
                total_unique_trials = filtered_facilities_df['nct_id'].nunique()
                st.write(f"Total unique trials in filtered set: {total_unique_trials}")
                
                # Get count for Metro Vancouver and GTA
                metro_van_mask = filtered_facilities_df['norm_city'] == 'metro vancouver'
                metro_van_trial_count = filtered_facilities_df[metro_van_mask]['nct_id'].nunique()
                st.write(f"Metro Vancouver unique trial count: {metro_van_trial_count}")
                
                gta_mask = filtered_facilities_df['norm_city'] == 'greater toronto area'
                gta_trial_count = filtered_facilities_df[gta_mask]['nct_id'].nunique()
                st.write(f"Greater Toronto Area unique trial count: {gta_trial_count}")
                
                # Count original cities that were mapped to metro areas
                original_cities = filtered_facilities_df.copy()
                original_cities['unmapped_city'] = filtered_facilities_df['city'].apply(
                    lambda x: normalize_city_name(x, apply_metro_mapping=False)
                )
                
                # List cities that were mapped to Metro Vancouver
                metro_van_cities = original_cities[
                    original_cities['unmapped_city'].isin(METRO_VANCOUVER_CITIES)
                ]['unmapped_city'].unique()
                
                st.write(f"Cities mapped to Metro Vancouver ({len(metro_van_cities)}):")
                st.write(", ".join(sorted(metro_van_cities)))
                
                # List cities that were mapped to GTA  
                gta_cities = original_cities[
                    original_cities['unmapped_city'].isin(GREATER_TORONTO_AREA_CITIES)
                ]['unmapped_city'].unique()
                
                st.write(f"Cities mapped to Greater Toronto Area ({len(gta_cities)}):")
                st.write(", ".join(sorted(gta_cities)))
                
                # Show sample of the data
                st.write("Sample of filtered facilities data:")
                st.dataframe(filtered_facilities_df.head(10))
            
            # Continue with the visualization
            city_counts = filtered_facilities_df.groupby('norm_city')['nct_id'].nunique().reset_index()
            city_counts.columns = ['city', 'trial_count']
            city_counts = city_counts.sort_values('trial_count', ascending=True)
            
            # Capitalize city names for display
            city_counts['city_display'] = city_counts['city'].apply(
                lambda x: ' '.join(word.capitalize() for word in x.split())
            )
            
            top_n = min(10, len(city_counts))
            top_cities = city_counts.tail(top_n)
            
            fig_cities = px.bar(
                top_cities,
                y='city_display',
                x='trial_count',
                orientation='h',
                title=f'Top {top_n} Canadian Cities/Metro Areas by Number of Clinical Trials',
                labels={'city_display': 'City/Metro Area', 'trial_count': 'Number of Trials'},
                color='trial_count',
                color_continuous_scale='Viridis',
            )
            
            fig_cities.update_layout(
                height=max(400, top_n * 25),
                xaxis_title="Number of Trials",
                yaxis_title="City/Metro Area",
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


def plot_normalized_geographic_map(facilities_df: pd.DataFrame, min_population: int = 400000) -> Tuple[Optional[px.scatter_mapbox], List[str]]:
    """
    Create a map visualization showing the distribution of trials across Canada,
    normalized by city population (trials per 100,000 residents).
    Only includes cities with a population greater than or equal to the min_population threshold.
    Cities within the Greater Vancouver Area and Greater Toronto Area are consolidated.
    
    Args:
        facilities_df: DataFrame containing facility information with a 'city' column 
                      and 'nct_id' for trial IDs.
        min_population: Minimum population threshold for including cities (default: 400000)
        
    Returns:
        Tuple of (plotly scatter_mapbox figure, list of normalized cities missing coordinates or population data)
    """
    # Validate required columns
    if facilities_df.empty or 'city' not in facilities_df.columns or 'nct_id' not in facilities_df.columns:
        return None, []
    
    # Build a mapping of normalized city names to coordinates using the global CITY_COORDS
    normalized_city_coords = {}
    for city_name, coords in CITY_COORDS.items():
        # Use normalize_city_name with apply_metro_mapping=False to get the original normalized name
        norm_name = normalize_city_name(city_name, apply_metro_mapping=False)
        if norm_name:
            normalized_city_coords[norm_name] = coords

    # Add metro area coordinates (using central city coordinates)
    normalized_city_coords['metro vancouver'] = CITY_COORDS['vancouver']
    normalized_city_coords['greater toronto area'] = CITY_COORDS['toronto']

    # Add a normalized city column to the facilities data, applying metro area mapping
    facilities_df = facilities_df.copy()
    facilities_df['norm_city'] = facilities_df['city'].apply(lambda x: normalize_city_name(x, apply_metro_mapping=True))
    
    # Aggregate counts by normalized city name (now including metro areas)
    city_counts = facilities_df.groupby('norm_city')['nct_id'].nunique().reset_index()
    city_counts.columns = ['norm_city', 'trial_count']
    
    # Add population data
    # For metro areas, we need to define population manually
    # These are approximate population values for the metro areas
    METRO_POPULATIONS = {
        'metro vancouver': 2463431,  # Metro Vancouver population
        'greater toronto area': 6417516,  # GTA population
    }
    
    # Add city populations, prioritizing metro area populations for consolidated areas
    city_counts['population'] = city_counts['norm_city'].apply(
        lambda x: METRO_POPULATIONS.get(x, CITY_POPULATIONS.get(x, None))
    )
    
    # Filter out cities without population data and apply minimum population threshold
    city_counts_with_pop = city_counts[
        (city_counts['population'].notna()) & 
        (city_counts['population'] >= min_population)
    ].copy()
    
    # Calculate trials per 100,000 residents
    city_counts_with_pop['trials_per_100k'] = (
        city_counts_with_pop['trial_count'] / city_counts_with_pop['population'] * 100000
    ).round(2)
    
    # Sort by normalized count
    city_counts_with_pop = city_counts_with_pop.sort_values('trials_per_100k', ascending=False)
    
    # Create the map visualization
    cities_without_coords_or_pop = []
    cities_with_coords_and_pop = []
    
    def get_coords(norm_city: str) -> Tuple[Dict[str, float], bool]:
        if norm_city in normalized_city_coords:
            return normalized_city_coords[norm_city], True
        # Try checking against the spelling corrections (if part of the key)
        for variant, corrected in SPELLING_CORRECTIONS.items():
            if variant in norm_city or norm_city in variant:
                if corrected and corrected in CITY_COORDS:
                    return CITY_COORDS[corrected], True
                    
        # Special handling for metro areas
        if norm_city == 'metro vancouver':
            return CITY_COORDS['vancouver'], True
        elif norm_city == 'greater toronto area':
            return CITY_COORDS['toronto'], True
            
        cities_without_coords_or_pop.append(norm_city)
        # Default coordinate: center of Canada
        return {'lat': 56.130366, 'lon': -106.346771}, False
    
    for _, row in city_counts_with_pop.iterrows():
        norm_city = row['norm_city']
        coords, found = get_coords(norm_city)
        cities_with_coords_and_pop.append({
            'city': norm_city,
            'trial_count': row['trial_count'],
            'population': row['population'],
            'trials_per_100k': row['trials_per_100k'],
            'lat': coords['lat'],
            'lon': coords['lon'],
            'found': found
        })
    
    # Get cities with population data >= min_population but missing from city_counts_with_pop
    cities_with_pop_no_trials = []
    
    # First check standard cities
    for city in CITY_POPULATIONS.keys():
        if city not in city_counts_with_pop['norm_city'].values and CITY_POPULATIONS.get(city, 0) >= min_population:
            cities_with_pop_no_trials.append(city)
    
    # Then check metro areas
    for metro, pop in METRO_POPULATIONS.items():
        if metro not in city_counts_with_pop['norm_city'].values and pop >= min_population:
            cities_with_pop_no_trials.append(metro)
    
    for city in cities_with_pop_no_trials:
        if city in normalized_city_coords:
            coords = normalized_city_coords[city]
            cities_with_coords_and_pop.append({
                'city': city,
                'trial_count': 0,
                'population': METRO_POPULATIONS.get(city, CITY_POPULATIONS.get(city)),
                'trials_per_100k': 0,
                'lat': coords['lat'],
                'lon': coords['lon'],
                'found': True
            })
    
    map_df = pd.DataFrame(cities_with_coords_and_pop)
    map_df_for_viz = map_df[map_df['found']].copy()
    
    # For display purposes, capitalize city names (including metro areas)
    map_df_for_viz['city_display'] = map_df_for_viz['city'].apply(
        lambda x: ' '.join(word.capitalize() for word in x.split())
    )
    
    # Create the map
    fig_map = px.scatter_mapbox(
        map_df_for_viz,
        lat="lat", 
        lon="lon",
        size="trials_per_100k",
        color="trials_per_100k",
        color_continuous_scale="Viridis",
        hover_name="city_display",
        hover_data={
            "lat": False, 
            "lon": False, 
            "trials_per_100k": True, 
            "trial_count": True,
            "population": True,
            "city": False
        },
        title=f"Canadian Cities/Metro Areas (Pop. ≥ {min_population/1000:.0f}k): Clinical Trials per 100,000 Residents",
        size_max=35,
        zoom=3,
    )
    fig_map.update_layout(
        height=600,
        mapbox_style="carto-positron",
    )
    
    # Add cities without population data to the missing list
    cities_without_pop = city_counts[city_counts['population'].isna()]['norm_city'].tolist()
    for city in cities_without_pop:
        if city not in cities_without_coords_or_pop:
            cities_without_coords_or_pop.append(city)
    
    # Add cities below the population threshold to the missing list
    cities_below_threshold = city_counts[
        (city_counts['population'].notna()) & 
        (city_counts['population'] < min_population)
    ]['norm_city'].tolist()
    
    return fig_map, sorted(list(set(cities_without_coords_or_pop + cities_below_threshold)))


def render_city_visualization_normalized(filtered_df: pd.DataFrame, conn: any, facilities_df: Optional[pd.DataFrame] = None, min_population: int = 400000):
    """
    Render the city visualization section with a map and a bar chart,
    with trial counts normalized by city population.
    Only includes cities with a population greater than or equal to the min_population threshold.
    Cities within the Greater Vancouver Area and Greater Toronto Area are consolidated.
    
    Args:
        filtered_df: DataFrame with filtered trial data.
        conn: Database connection for fetching additional data if needed.
        facilities_df: Optional pre-fetched facilities data.
        min_population: Minimum population threshold for including cities (default: 400000)
    """
    # Import the database utility function to avoid circular imports
    from utils.database_utils import fetch_facilities_for_trials
    
    st.subheader(f"Population-Normalized Geographic Distribution (Pop. ≥ {min_population/1000:.0f}k)")
    
    # Get the list of filtered trial IDs
    filtered_nct_ids = filtered_df['nct_id'].tolist()
    
    # Fetch facilities data if not provided or empty
    if facilities_df is None or facilities_df.empty:
        with st.spinner("Loading geographic data..."):
            facilities_df = fetch_facilities_for_trials(conn, filtered_nct_ids)
    
    if facilities_df is not None and not facilities_df.empty:
        # Metro area populations
        METRO_POPULATIONS = {
            'metro vancouver': 2463431,  # Metro Vancouver population
            'greater toronto area': 6417516,  # GTA population
        }
        
        # Create population-normalized map visualization with population threshold
        fig_map, missing_cities = plot_normalized_geographic_map(facilities_df, min_population)
        
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
            st.info(f"This map shows trials per 100,000 residents for cities/metropolitan areas with population ≥ {min_population/1000:.0f}k. Cities within the Greater Toronto Area and Metro Vancouver have been consolidated.")
            
            # For the bar chart, aggregate trials by normalized city names with population data
            facilities_df = facilities_df.copy()
            facilities_df['norm_city'] = facilities_df['city'].apply(
                lambda x: normalize_city_name(x, apply_metro_mapping=True)
            )
            
            city_counts = facilities_df.groupby('norm_city')['nct_id'].nunique().reset_index()
            city_counts.columns = ['city', 'trial_count']
            
            # Add population data for cities and metro areas
            city_counts['population'] = city_counts['city'].apply(
                lambda x: METRO_POPULATIONS.get(x, CITY_POPULATIONS.get(x, None))
            )
            
            # Filter cities with population data above the threshold and calculate trials per 100k
            city_counts_with_pop = city_counts[
                (city_counts['population'].notna()) & 
                (city_counts['population'] >= min_population)
            ].copy()
            
            city_counts_with_pop['trials_per_100k'] = (
                city_counts_with_pop['trial_count'] / city_counts_with_pop['population'] * 100000
            ).round(2)
            
            # Sort by trials per 100k for bar chart
            city_counts_with_pop = city_counts_with_pop.sort_values('trials_per_100k', ascending=True)
            
            # Create display names with proper capitalization
            city_counts_with_pop['city_display'] = city_counts_with_pop['city'].apply(
                lambda x: ' '.join(word.capitalize() for word in x.split())
            )
            
            if not city_counts_with_pop.empty:
                top_n = len(city_counts_with_pop)  # Show all cities that meet the threshold
                
                fig_cities = px.bar(
                    city_counts_with_pop,
                    y='city_display',
                    x='trials_per_100k',
                    orientation='h',
                    title=f'Major Canadian Cities/Metro Areas (Pop. ≥ {min_population/1000:.0f}k) by Clinical Trials per 100,000 Residents',
                    labels={'city_display': 'City/Metro Area', 'trials_per_100k': 'Trials per 100,000 Residents'},
                    color='trials_per_100k',
                    color_continuous_scale='Viridis',
                    text='trial_count'
                )
                
                fig_cities.update_layout(
                    height=max(400, top_n * 30),
                    xaxis_title="Trials per 100,000 Residents",
                    yaxis_title="City/Metro Area",
                    yaxis={'categoryorder':'total ascending'},
                    template="plotly_white"
                )
                
                fig_cities.update_traces(
                    texttemplate='%{text} trials',
                    textposition='outside'
                )
                
                st.plotly_chart(fig_cities, use_container_width=True)
                
                # Generate a table with full data
                st.subheader(f"Population-Normalized Trial Data by Major City/Metro Area (Pop. ≥ {min_population/1000:.0f}k)")
                table_data = city_counts_with_pop.sort_values('trials_per_100k', ascending=False)
                table_data = table_data[['city_display', 'trial_count', 'population', 'trials_per_100k']]
                table_data.columns = ['City/Metro Area', 'Number of Trials', 'Population', 'Trials per 100,000 Residents']
                st.dataframe(table_data, use_container_width=True)
            else:
                st.warning(f"No cities with population ≥ {min_population/1000:.0f}k found in the data.")
                
            # Count cities excluded due to population threshold
            cities_below_threshold = city_counts[
                (city_counts['population'].notna()) & 
                (city_counts['population'] < min_population)
            ]
            
            # Show excluded cities
            with st.expander(f"Cities Excluded Due to Population Threshold (< {min_population/1000:.0f}k) ({len(cities_below_threshold)})", expanded=False):
                if not cities_below_threshold.empty:
                    st.write("The following cities/areas had trials but were excluded from the visualization due to population threshold:")
                    cities_table = cities_below_threshold.sort_values('population', ascending=False)
                    
                    # Add display names with proper capitalization
                    cities_table['city_display'] = cities_table['city'].apply(
                        lambda x: ' '.join(word.capitalize() for word in x.split())
                    )
                    
                    cities_table = cities_table[['city_display', 'trial_count', 'population']]
                    cities_table.columns = ['City/Area', 'Number of Trials', 'Population']
                    st.dataframe(cities_table, use_container_width=True)
                else:
                    st.write("No cities were excluded due to the population threshold.")
            
            # Info on metropolitan area consolidation
            with st.expander("Metropolitan Area Consolidation Details", expanded=False):
                st.write("### Metropolitan Area Groupings")
                
                st.write("#### Greater Toronto Area (GTA)")
                st.write(f"Population: {METRO_POPULATIONS['greater toronto area']:,}")
                st.write("The following cities and towns have been consolidated into the Greater Toronto Area:")
                st.write(", ".join(sorted(GREATER_TORONTO_AREA_CITIES)))
                
                st.write("#### Metro Vancouver")
                st.write(f"Population: {METRO_POPULATIONS['metro vancouver']:,}")
                st.write("The following cities and districts have been consolidated into Metro Vancouver:")
                st.write(", ".join(sorted(METRO_VANCOUVER_CITIES)))
                
                # Show which cities in the data were actually mapped
                filtered_facilities_df = facilities_df.copy()
                filtered_facilities_df['unmapped_city'] = filtered_facilities_df['city'].apply(
                    lambda x: normalize_city_name(x, apply_metro_mapping=False)
                )
                
                # List cities that were mapped to Metro Vancouver
                metro_van_cities = filtered_facilities_df[
                    filtered_facilities_df['unmapped_city'].isin(METRO_VANCOUVER_CITIES)
                ]['unmapped_city'].unique()
                
                st.write("##### Cities in this dataset mapped to Metro Vancouver:")
                if len(metro_van_cities) > 0:
                    st.write(", ".join(sorted(metro_van_cities)))
                else:
                    st.write("No cities in the current dataset were mapped to Metro Vancouver.")
                
                # List cities that were mapped to GTA  
                gta_cities = filtered_facilities_df[
                    filtered_facilities_df['unmapped_city'].isin(GREATER_TORONTO_AREA_CITIES)
                ]['unmapped_city'].unique()
                
                st.write("##### Cities in this dataset mapped to Greater Toronto Area:")
                if len(gta_cities) > 0:
                    st.write(", ".join(sorted(gta_cities)))
                else:
                    st.write("No cities in the current dataset were mapped to Greater Toronto Area.")

    
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
        title=f"Distribution of Trials by Phase (n = {n} Trials Reporting Phase)",
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
        df: DataFrame containing trial data
        
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
        df: DataFrame containing trial data
        
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
        title='Clinical Trials by Start Year and Status'
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
        title='Cumulative Trials by Year and Status'
    )
    fig_area.update_layout(template="plotly_white")
    
    return fig_bar, fig_area


def plot_trial_age_distribution(df: pd.DataFrame) -> px.histogram:
    """
    Create a histogram showing the distribution of minimum ages in trials.
    
    Args:
        df: DataFrame containing trial data
        
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
    
    # Add vertical lines for common age boundaries
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
    Render summary metrics for the trials data.
    
    Args:
        df: DataFrame containing trial data
    """
    total_trials = len(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trials", total_trials)
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
        df: DataFrame containing trial data
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

# Canadian provinces and territories mapping
PROVINCE_MAPPING = {
    'ON': 'Ontario',
    'QC': 'Quebec',
    'BC': 'British Columbia',
    'AB': 'Alberta',
    'MB': 'Manitoba',
    'SK': 'Saskatchewan',
    'NS': 'Nova Scotia',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland and Labrador',
    'PE': 'Prince Edward Island',
    'NT': 'Northwest Territories',
    'YT': 'Yukon',
    'NU': 'Nunavut',
    # Common city to province mappings for data cleanup
    'toronto': 'Ontario',
    'ottawa': 'Ontario',
    'mississauga': 'Ontario',
    'hamilton': 'Ontario',
    'london': 'Ontario',
    'markham': 'Ontario',
    'vaughan': 'Ontario',
    'kitchener': 'Ontario',
    'windsor': 'Ontario',
    'greater toronto area': 'Ontario',
    
    'montreal': 'Quebec',
    'quebec city': 'Quebec',
    'quebec': 'Quebec',
    'laval': 'Quebec',
    'gatineau': 'Quebec',
    'sherbrooke': 'Quebec',
    
    'vancouver': 'British Columbia',
    'victoria': 'British Columbia',
    'burnaby': 'British Columbia',
    'richmond': 'British Columbia',
    'surrey': 'British Columbia',
    'kelowna': 'British Columbia',
    'metro vancouver': 'British Columbia',
    
    'calgary': 'Alberta',
    'edmonton': 'Alberta',
    'red deer': 'Alberta',
    'lethbridge': 'Alberta',
    
    'winnipeg': 'Manitoba',
    
    'saskatoon': 'Saskatchewan',
    'regina': 'Saskatchewan',
    
    'halifax': 'Nova Scotia',
    
    'saint john': 'New Brunswick',
    'st. john': 'New Brunswick',
    'moncton': 'New Brunswick',
    'fredericton': 'New Brunswick',
    
    "st. john's": 'Newfoundland and Labrador',
    
    'charlottetown': 'Prince Edward Island'
}

# Province population data (2021 census)
PROVINCE_POPULATIONS = {
    'Ontario': 14223942,
    'Quebec': 8501833,
    'British Columbia': 5000879,
    'Alberta': 4262635,
    'Manitoba': 1342153,
    'Saskatchewan': 1132505,
    'Nova Scotia': 969383,
    'New Brunswick': 775610,
    'Newfoundland and Labrador': 510550,
    'Prince Edward Island': 154331,
    'Northwest Territories': 41070,
    'Yukon': 40232,
    'Nunavut': 36858
}

def get_province_from_city(city: str) -> str:
    """
    Get province name from city name using the PROVINCE_MAPPING dictionary.
    
    Args:
        city: Normalized city name
        
    Returns:
        Province name or "Unknown" if city not found
    """
    # Normalize city name
    norm_city = normalize_city_name(city, apply_metro_mapping=True).lower()
    
    # Check if city is in the mapping
    if norm_city in PROVINCE_MAPPING:
        return PROVINCE_MAPPING[norm_city]
    
    # Check if city contains a province abbreviation
    for abbr, province in PROVINCE_MAPPING.items():
        if len(abbr) == 2 and f", {abbr.lower()}" in norm_city:
            return province
    
    return "Unknown"

def process_trials_by_province(facilities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process facilities data to get trial counts by province.
    
    Args:
        facilities_df: DataFrame containing facility information with 'city' and 'nct_id' columns
        
    Returns:
        DataFrame with province-level aggregation of trial counts
    """
    if facilities_df.empty or 'city' not in facilities_df.columns or 'nct_id' not in facilities_df.columns:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original DataFrame
    df = facilities_df.copy()
    
    # Add a normalized city column to the facilities data
    df['norm_city'] = df['city'].apply(lambda x: normalize_city_name(x, apply_metro_mapping=True))
    
    # Add province based on the city
    df['province'] = df['norm_city'].apply(get_province_from_city)
    
    # Count unique trials by province
    province_counts = df.groupby('province')['nct_id'].nunique().reset_index()
    province_counts.columns = ['province', 'trial_count']
    
    # Sort by trial count in descending order
    province_counts = province_counts.sort_values('trial_count', ascending=False)
    
    # Add population data
    province_counts['population'] = province_counts['province'].map(PROVINCE_POPULATIONS)
    
    # Calculate trials per 100,000 residents
    province_counts['trials_per_100k'] = (
        province_counts['trial_count'] / province_counts['population'] * 100000
    ).round(2)
    
    return province_counts

def plot_province_distribution(province_counts: pd.DataFrame, title_prefix: str = "Clinical") -> Tuple[px.bar, px.bar]:
    """
    Create bar charts showing the distribution of trials across Canadian provinces.
    
    Args:
        province_counts: DataFrame with province-level aggregation of trial counts
        title_prefix: Prefix for chart titles (e.g., "Pediatric", "Adult", "Clinical")
        
    Returns:
        Tuple of (count bar chart, normalized bar chart)
    """
    if province_counts.empty:
        return None, None
    
    # Filter out "Unknown" province
    province_counts_filtered = province_counts[province_counts['province'] != "Unknown"].copy()
    
    if province_counts_filtered.empty:
        return None, None
    
    # Sort for visualization
    province_counts_by_count = province_counts_filtered.sort_values('trial_count', ascending=True)
    province_counts_by_normalized = province_counts_filtered.sort_values('trials_per_100k', ascending=True)
    
    # Create bar chart of raw counts
    fig_count = px.bar(
        province_counts_by_count,
        y='province',
        x='trial_count',
        orientation='h',
        title=f'{title_prefix} Trials by Province in Canada',
        labels={'province': 'Province', 'trial_count': 'Number of Trials'},
        color='trial_count',
        color_continuous_scale='Viridis',
    )
    fig_count.update_layout(
        height=max(400, len(province_counts_by_count) * 30),
        xaxis_title="Number of Trials",
        yaxis_title="Province",
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white"
    )
    
    # Create bar chart of population-normalized counts
    fig_normalized = px.bar(
        province_counts_by_normalized,
        y='province',
        x='trials_per_100k',
        orientation='h',
        title=f'{title_prefix} Trials per 100,000 Residents by Province in Canada',
        labels={'province': 'Province', 'trials_per_100k': 'Trials per 100,000 Residents'},
        color='trials_per_100k',
        color_continuous_scale='Viridis',
        text='trial_count'
    )
    fig_normalized.update_layout(
        height=max(400, len(province_counts_by_normalized) * 30),
        xaxis_title="Trials per 100,000 Residents",
        yaxis_title="Province",
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white"
    )
    fig_normalized.update_traces(
        texttemplate='%{text} trials',
        textposition='outside'
    )
    
    return fig_count, fig_normalized

def render_province_visualization(filtered_df: pd.DataFrame, conn: any, title_prefix: str = "Clinical"):
    """
    Render the province visualization section with bar charts and summary metrics.
    
    Args:
        filtered_df: DataFrame with filtered trial data
        conn: Database connection for fetching additional data if needed
        title_prefix: Prefix for chart titles (e.g., "Pediatric", "Adult", "Clinical")
    """
    # Import the database utility function to avoid circular imports
    from utils.database_utils import fetch_facilities_for_trials
    
    st.subheader("Distribution by Province")
    
    # Get the list of filtered trial IDs
    filtered_nct_ids = filtered_df['nct_id'].tolist()
    
    if not filtered_nct_ids:
        st.warning("No trials match the current filters.")
        return
    
    # Fetch facilities data for the filtered trials
    with st.spinner("Loading province data..."):
        filtered_facilities_df = fetch_facilities_for_trials(conn, filtered_nct_ids)
    
    if filtered_facilities_df is not None and not filtered_facilities_df.empty:
        # Process data by province
        province_counts = process_trials_by_province(filtered_facilities_df)
        
        if not province_counts.empty:
            # Display summary metrics for provinces
            provinces_with_trials = len(province_counts[province_counts['province'] != "Unknown"])
            total_trials = province_counts['trial_count'].sum()
            
            # Display metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Provinces/Territories with Trials", provinces_with_trials)
            with col2:
                st.metric("Total Trials with Province Data", total_trials)
            
            # Show unknown trials if any
            unknown_row = province_counts[province_counts['province'] == "Unknown"]
            if not unknown_row.empty and unknown_row['trial_count'].iloc[0] > 0:
                st.warning(f"{unknown_row['trial_count'].iloc[0]} trials could not be mapped to a specific province.")
            
            # Create and display visualizations
            fig_count, fig_normalized = plot_province_distribution(province_counts, title_prefix)
            
            if fig_count and fig_normalized:
                st.plotly_chart(fig_count, use_container_width=True)
                st.plotly_chart(fig_normalized, use_container_width=True)
                
                # Show data table
                with st.expander("Show Province Data Table", expanded=False):
                    table_data = province_counts.sort_values('trial_count', ascending=False).copy()
                    table_data = table_data[table_data['province'] != "Unknown"]
                    
                    # Add percentage column
                    total_mapped_trials = table_data['trial_count'].sum()
                    table_data['percentage'] = (table_data['trial_count'] / total_mapped_trials * 100).round(1)
                    
                    # Reorder and rename columns
                    table_data = table_data[['province', 'trial_count', 'percentage', 'population', 'trials_per_100k']]
                    table_data.columns = [
                        'Province', 
                        'Number of Trials', 
                        'Percentage of Trials (%)',
                        'Population', 
                        'Trials per 100,000 Residents'
                    ]
                    
                    st.dataframe(table_data, use_container_width=True)
            else:
                st.warning("Could not create province visualizations.")
        else:
            st.warning("No province data available for the selected trials.")
    else:
        st.warning("No facility data available for province analysis.")
        