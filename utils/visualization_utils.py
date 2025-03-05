"""
Visualization utility functions for the Clinical Trials app.
This module contains reusable visualization functions for creating charts and maps.
"""
import streamlit as st
import unicodedata
import re
import pandas as pd
from pandas.api.types import CategoricalDtype
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, Any

# Import data constants from the new module
from utils.data_constants import (
    METRO_VANCOUVER_CITIES,
    GREATER_TORONTO_AREA_CITIES,
    PROVINCE_MAPPING,
    PROVINCE_POPULATIONS,
    CITY_COORDS,
    METRO_POPULATIONS,
    CITY_POPULATIONS,
    SPELLING_CORRECTIONS,
    ONCOLOGY_KEYWORDS
)

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
        st.metric("Total Trial Sites", total_trials)
    with col2:
        active_trials = df[df['overall_status'].isin(['RECRUITING', 'ACTIVE, NOT RECRUITING', 'ENROLLING BY INVITATION'])].shape[0]
        st.metric("Active Trial Sites", active_trials)
    with col3:
        recruiting_trials = df[df['overall_status'] == 'RECRUITING'].shape[0]
        st.metric("Recruiting Trial Sites", recruiting_trials)


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


def classify_condition(condition: str) -> str:
    """
    Classify a condition into a category.
    Currently distinguishes between oncology and other conditions.
    
    Args:
        condition: Condition name to classify
        
    Returns:
        Category name (Oncology or Other)
    """
    # Uses ONCOLOGY_KEYWORDS imported from data_constants
    if condition:
        condition_lower = condition.lower()
        for keyword in ONCOLOGY_KEYWORDS:
            if keyword in condition_lower:
                return 'Oncology'
    
    return 'Other'


def count_intervention_types(interventions_df: pd.DataFrame) -> Dict[str, int]:
    """
    Count the frequency of each intervention type.
    
    Args:
        interventions_df: DataFrame with intervention data
        
    Returns:
        Dictionary mapping intervention types to counts
    """
    return dict(Counter(interventions_df['intervention_type']))


def count_top_conditions(conditions_dict: Dict[str, List[str]], limit: int = 15, by_category: bool = False) -> pd.DataFrame:
    """
    Count the frequency of each condition across all trials.
    
    Args:
        conditions_dict: Dictionary mapping NCT IDs to lists of conditions
        limit: Max number of top conditions to return
        by_category: Whether to return data grouped by category
        
    Returns:
        DataFrame with condition counts, sorted by frequency
    """
    # Flatten the conditions list
    all_conditions = []
    for conditions in conditions_dict.values():
        all_conditions.extend(conditions)
    
    # Count and sort
    condition_counts = Counter(all_conditions)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'condition': list(condition_counts.keys()),
        'count': list(condition_counts.values())
    })
    
    # Add category
    df['category'] = df['condition'].apply(classify_condition)
    
    # Sort by count (descending)
    df = df.sort_values('count', ascending=False)
    
    if by_category:
        # Return all data grouped by category
        return df
    else:
        # Return top N conditions
        return df.head(limit)


def plot_intervention_distribution(interventions_df: pd.DataFrame) -> px.pie:
    """
    Create a pie chart showing the distribution of intervention types.
    
    Args:
        interventions_df: DataFrame with intervention data
        
    Returns:
        Plotly pie chart figure
    """
    intervention_counts = count_intervention_types(interventions_df)
    
    df = pd.DataFrame({
        'intervention_type': list(intervention_counts.keys()),
        'count': list(intervention_counts.values())
    })
    
    fig = px.pie(
        df,
        values='count',
        names='intervention_type',
        title='Distribution of Intervention Types',
        hole=0.4
    )
    
    fig.update_layout(legend_title="Intervention Type")
    
    return fig


def plot_top_conditions(conditions_dict: Dict[str, List[str]], limit: int = 15, by_category: bool = True) -> go.Figure:
    """
    Create a horizontal bar chart of the top conditions, colored by category.
    
    Args:
        conditions_dict: Dictionary mapping NCT IDs to lists of conditions
        limit: Max number of top conditions to return per category
        by_category: Whether to split visualization by category
        
    Returns:
        Plotly bar chart figure
    """
    # Get condition data with categories
    df = count_top_conditions(conditions_dict, limit, by_category=True)
    
    if df.empty:
        return None
    
    if by_category:
        # Create a separate chart for each category
        fig = go.Figure()
        
        # Get top conditions for each category
        oncology_df = df[df['category'] == 'Oncology'].head(limit)
        other_df = df[df['category'] == 'Other'].head(limit)
        
        # Add Oncology conditions
        if not oncology_df.empty:
            fig.add_trace(go.Bar(
                y=oncology_df['condition'],
                x=oncology_df['count'],
                orientation='h',
                name='Oncology',
                marker_color='#FF4136',  # Red
                customdata=oncology_df['category'],
                hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Category: %{customdata}<extra></extra>'
            ))
        
        # Add Other conditions
        if not other_df.empty:
            fig.add_trace(go.Bar(
                y=other_df['condition'],
                x=other_df['count'],
                orientation='h',
                name='Other',
                marker_color='#0074D9',  # Blue
                customdata=other_df['category'],
                hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Category: %{customdata}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Top Conditions by Category',
            xaxis_title='Number of Trials',
            yaxis_title='Condition',
            height=max(400, min(limit * 25 * 2, 800)),
            legend_title='Category',
            barmode='group'
        )
        
    else:
        # Limited to top N overall
        df = df.head(limit)
        
        # Use different colors for different categories
        color_map = {'Oncology': '#FF4136', 'Other': '#0074D9'}
        
        fig = px.bar(
            df,
            y='condition',
            x='count',
            orientation='h',
            title=f'Top {limit} Conditions',
            labels={'condition': 'Condition', 'count': 'Number of Trials', 'category': 'Category'},
            color='category',
            color_discrete_map=color_map,
            hover_data=['category']
        )
        
        fig.update_layout(
            height=max(400, min(limit * 25, 800)),
            yaxis={'categoryorder': 'total ascending'}
        )
    
    return fig


def plot_conditions_network(conditions_dict: Dict[str, List[str]], min_co_occurrence: int = 2) -> Optional[go.Figure]:
    """
    Create a network diagram showing conditions that co-occur in trials.
    
    Args:
        conditions_dict: Dictionary mapping NCT IDs to lists of conditions
        min_co_occurrence: Minimum number of co-occurrences to include in the network
        
    Returns:
        Plotly graph object figure or None if networkx is not available
    """
    # Check if networkx is available
    try:
        import networkx as nx
    except ImportError:
        st.warning("The networkx library is required for the conditions network visualization. Please install it with `pip install networkx`.")
        return None
    
    # Count co-occurrences
    co_occurrences = {}
    
    for conditions in conditions_dict.values():
        if len(conditions) > 1:
            # For each pair of conditions in this trial
            for i, cond1 in enumerate(conditions):
                for cond2 in conditions[i+1:]:
                    pair = tuple(sorted([cond1, cond2]))
                    if pair in co_occurrences:
                        co_occurrences[pair] += 1
                    else:
                        co_occurrences[pair] = 1
    
    # Filter by minimum co-occurrence
    filtered_co_occurrences = {pair: count for pair, count in co_occurrences.items() 
                             if count >= min_co_occurrence}
    
    if not filtered_co_occurrences:
        return None
    
    # Get unique conditions
    unique_conditions = set()
    for pair in filtered_co_occurrences.keys():
        unique_conditions.add(pair[0])
        unique_conditions.add(pair[1])
    
    # Create node DataFrame
    nodes = pd.DataFrame({
        'id': list(range(len(unique_conditions))),
        'label': list(unique_conditions)
    })
    
    # Add category to nodes
    nodes['category'] = nodes['label'].apply(classify_condition)
    
    # Create mapping from condition to node ID
    condition_to_id = {cond: idx for idx, cond in enumerate(unique_conditions)}
    
    # Create edge DataFrame
    edges = []
    for pair, weight in filtered_co_occurrences.items():
        source_id = condition_to_id[pair[0]]
        target_id = condition_to_id[pair[1]]
        edges.append({
            'source': source_id,
            'target': target_id,
            'weight': weight
        })
    
    edges_df = pd.DataFrame(edges)
    
    # Create network visualization using plotly
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create network layout (using Fruchterman-Reingold algorithm)
    G = nx.Graph()
    
    # Add nodes
    for _, row in nodes.iterrows():
        G.add_node(row['id'], label=row['label'], category=row['category'])
    
    # Add edges
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    
    # Calculate layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Add edges to trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Create separate node traces for each category
    oncology_nodes = {'x': [], 'y': [], 'text': [], 'id': []}
    other_nodes = {'x': [], 'y': [], 'text': [], 'id': []}
    
    # Assign nodes to the appropriate category
    for node in G.nodes():
        x, y = pos[node]
        category = G.nodes[node]['category']
        node_info = G.nodes[node]['label']
        
        if category == 'Oncology':
            oncology_nodes['x'].append(x)
            oncology_nodes['y'].append(y)
            oncology_nodes['text'].append(node_info)
            oncology_nodes['id'].append(node)
        else:
            other_nodes['x'].append(x)
            other_nodes['y'].append(y)
            other_nodes['text'].append(node_info)
            other_nodes['id'].append(node)
    
    # Create oncology node trace
    oncology_node_trace = go.Scatter(
        x=oncology_nodes['x'],
        y=oncology_nodes['y'],
        text=oncology_nodes['text'],
        mode='markers',
        name='Oncology',
        marker=dict(
            color='#FF4136',  # Red
            size=15,
            line=dict(width=2)
        ),
        hovertemplate='<b>%{text}</b><br>Category: Oncology<extra></extra>'
    )
    
    # Create other node trace
    other_node_trace = go.Scatter(
        x=other_nodes['x'],
        y=other_nodes['y'],
        text=other_nodes['text'],
        mode='markers',
        name='Other',
        marker=dict(
            color='#0074D9',  # Blue
            size=15,
            line=dict(width=2)
        ),
        hovertemplate='<b>%{text}</b><br>Category: Other<extra></extra>'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, oncology_node_trace, other_node_trace],
                   layout=go.Layout(
                       title=dict(
                           text='Co-occurring Conditions Network (Colored by Category)',
                           font=dict(size=16)
                       ),
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       legend=dict(
                           title='Category',
                           yanchor="top",
                           y=0.99,
                           xanchor="right",
                           x=0.99
                       ),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig


def plot_interventions_by_condition(interventions_df: pd.DataFrame, 
                                   conditions_dict: Dict[str, List[str]], 
                                   top_n: int = 10) -> Optional[go.Figure]:
    """
    Create a heatmap showing the relationship between top conditions and intervention types,
    with condition categories indicated by colored markers.
    
    Args:
        interventions_df: DataFrame with intervention data
        conditions_dict: Dictionary mapping NCT IDs to lists of conditions
        top_n: Number of top conditions to include per category
        
    Returns:
        Plotly heatmap figure or None if insufficient data
    """
    # Get top conditions by category
    top_conditions_df = count_top_conditions(conditions_dict, top_n, by_category=True)
    
    # Get top conditions for each category
    oncology_top = top_conditions_df[top_conditions_df['category'] == 'Oncology'].head(top_n)['condition'].tolist()
    other_top = top_conditions_df[top_conditions_df['category'] == 'Other'].head(top_n)['condition'].tolist()
    
    # Combine top conditions from both categories
    top_conditions = oncology_top + other_top
    
    # Create mapping of nct_id to intervention types
    nct_to_interventions = {}
    for _, row in interventions_df.iterrows():
        if row['nct_id'] not in nct_to_interventions:
            nct_to_interventions[row['nct_id']] = []
        nct_to_interventions[row['nct_id']].append(row['intervention_type'])
    
    # Count intervention types for each top condition
    condition_intervention_counts = {}
    for nct_id, conditions in conditions_dict.items():
        if nct_id in nct_to_interventions:
            intervention_types = nct_to_interventions[nct_id]
            for condition in conditions:
                if condition in top_conditions:
                    if condition not in condition_intervention_counts:
                        condition_intervention_counts[condition] = Counter()
                    condition_intervention_counts[condition].update(intervention_types)
    
    # Get unique intervention types
    all_intervention_types = set()
    for counter in condition_intervention_counts.values():
        all_intervention_types.update(counter.keys())
    all_intervention_types = sorted(all_intervention_types)
    
    # Create heatmap data
    heatmap_data = []
    condition_labels = []
    categories = []
    
    for condition in top_conditions:
        if condition in condition_intervention_counts:
            counter = condition_intervention_counts[condition]
            row = [counter.get(int_type, 0) for int_type in all_intervention_types]
            heatmap_data.append(row)
            condition_labels.append(condition)
            categories.append('Oncology' if condition in oncology_top else 'Other')
    
    if not heatmap_data:
        return None
    
    # Create the customdata for hover information
    customdata = []
    for i, category in enumerate(categories):
        customdata.append([category] * len(all_intervention_types))
    
    # Create heatmap
    fig = go.Figure()
    
    # Add the heatmap
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=all_intervention_types,
        y=condition_labels,
        colorscale='Viridis',
        customdata=customdata,
        hovertemplate='<b>Condition:</b> %{y}<br><b>Intervention:</b> %{x}<br><b>Count:</b> %{z}<br><b>Category:</b> %{customdata}<extra></extra>'
    ))
    
    # Add category markers next to condition names
    # Use a separate scatter trace with markers for categories
    y_positions = list(range(len(condition_labels)))
    
    # Oncology markers (red)
    oncology_y = [y for y, cat in zip(y_positions, categories) if cat == 'Oncology']
    if oncology_y:
        fig.add_trace(go.Scatter(
            x=[-0.5] * len(oncology_y),  # Position to the left of y-axis
            y=[condition_labels[i] for i in oncology_y],
            mode='markers',
            marker=dict(
                symbol='circle',
                color='#FF4136',  # Red for Oncology
                size=8
            ),
            name='Oncology',
            hoverinfo='name',
            showlegend=True
        ))
    
    # Other markers (blue)
    other_y = [y for y, cat in zip(y_positions, categories) if cat == 'Other']
    if other_y:
        fig.add_trace(go.Scatter(
            x=[-0.5] * len(other_y),  # Position to the left of y-axis
            y=[condition_labels[i] for i in other_y],
            mode='markers',
            marker=dict(
                symbol='circle',
                color='#0074D9',  # Blue for Other
                size=8
            ),
            name='Other',
            hoverinfo='name',
            showlegend=True
        ))
    
    # Set layout with improved margins for the legend and title
    fig.update_layout(
        title='Intervention Types by Top Conditions',
        xaxis_title='Intervention Type',
        yaxis_title='Condition',
        height=max(500, len(condition_labels) * 25),
        margin=dict(l=10, r=50, t=50, b=50),
        legend=dict(
            title="Condition Category",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.1
        ),
        # Hide x-axis value for the markers
        xaxis=dict(
            range=[-0.7, len(all_intervention_types) - 0.3],  # Extend x-axis to show markers
            showgrid=True
        )
    )
    
    return fig


def render_interventions_conditions_analysis(filtered_df: pd.DataFrame, 
                                            conn: any, 
                                            title_prefix: str = "Clinical",
                                            conditions_dict: Optional[Dict[str, List[str]]] = None,
                                            interventions_df: Optional[pd.DataFrame] = None):
    """
    Render the interventions and conditions analysis section.
    
    Args:
        filtered_df: DataFrame with filtered trial data
        conn: Database connection for fetching additional data
        title_prefix: Prefix for section titles (e.g., "Pediatric", "Adult")
        conditions_dict: Optional pre-fetched conditions dictionary
        interventions_df: Optional pre-fetched interventions DataFrame
    """
    # Import necessary functions to avoid circular imports
    from utils.database_utils import fetch_conditions_for_trials, fetch_interventions_for_trials, build_conditions_dict
    from utils.session_state_utils import safe_get_session_state
    
    st.subheader(f"{title_prefix} Trials - Interventions & Conditions Analysis")
    
    # Get the list of filtered trial IDs
    filtered_nct_ids = filtered_df['nct_id'].tolist()
    
    if not filtered_nct_ids:
        st.warning("No trials match the current filters.")
        return
    
    # Fetch or filter conditions data
    if conditions_dict is None:
        with st.spinner("Loading conditions data..."):
            conditions_df = fetch_conditions_for_trials(conn, filtered_nct_ids)
            if conditions_df is not None and not conditions_df.empty:
                conditions_dict = build_conditions_dict(conditions_df)
            else:
                conditions_dict = {}
    else:
        # Filter existing conditions_dict to include only filtered trials
        conditions_dict = {nct_id: conditions for nct_id, conditions in conditions_dict.items() 
                         if nct_id in filtered_nct_ids}
    
    # Fetch or filter interventions data
    if interventions_df is None:
        with st.spinner("Loading interventions data..."):
            interventions_df = fetch_interventions_for_trials(conn, filtered_nct_ids)
    else:
        # Filter existing interventions_df to include only filtered trials
        interventions_df = interventions_df[interventions_df['nct_id'].isin(filtered_nct_ids)]
    
    # Create tabs for different visualizations
    tabs = st.tabs([
        "Intervention Types", 
        "Top Conditions", 
        "Conditions Network", 
        "Interventions by Condition"
    ])
    
    with tabs[0]:
        st.write("### Intervention Types Distribution")
        if interventions_df is not None and not interventions_df.empty:
            fig_intervention = plot_intervention_distribution(interventions_df)
            st.plotly_chart(fig_intervention, use_container_width=True)
            
            # Summary statistics for interventions
            int_counts = count_intervention_types(interventions_df)
            
            # Create columns for intervention stats
            cols = st.columns(min(len(int_counts), 4))
            for i, (int_type, count) in enumerate(sorted(int_counts.items(), key=lambda x: x[1], reverse=True)):
                if i < len(cols):
                    cols[i].metric(f"{int_type}", count)
        else:
            st.info("No intervention data available for the selected trials.")
    
    with tabs[1]:
        st.write("### Top Conditions")
        if conditions_dict:
            # Add radio button to choose visualization style
            viz_style = st.radio(
                "Visualization Style:",
                ["Split by Category", "Combined"],
                horizontal=True,
                key="conditions_viz_style"
            )
            
            # Get top conditions for display
            top_n_conditions = st.slider(
                "Number of Top Conditions Per Category",
                min_value=5,
                max_value=20,
                value=10,
                key="top_conditions_count"
            )
            
            fig_conditions = plot_top_conditions(
                conditions_dict, 
                limit=top_n_conditions, 
                by_category=(viz_style == "Split by Category")
            )
            
            if fig_conditions:
                st.plotly_chart(fig_conditions, use_container_width=True)
                
                # Show condition classification stats
                all_conditions_df = count_top_conditions(conditions_dict, by_category=True)
                oncology_count = len(all_conditions_df[all_conditions_df['category'] == 'Oncology'])
                other_count = len(all_conditions_df[all_conditions_df['category'] == 'Other'])
                total_count = len(all_conditions_df)
                
                # Display stats in columns
                cols = st.columns(3)
                cols[0].metric("Total Unique Conditions", total_count)
                cols[1].metric("Oncology Conditions", oncology_count, 
                              f"{oncology_count/total_count:.1%}" if total_count > 0 else "0%")
                cols[2].metric("Other Conditions", other_count,
                              f"{other_count/total_count:.1%}" if total_count > 0 else "0%")
            else:
                st.warning("Could not create conditions visualization.")
        else:
            st.info("No condition data available for the selected trials.")
    
    with tabs[2]:
        st.write("### Conditions Co-occurrence Network")
        if conditions_dict:
            st.info("This network shows conditions that frequently appear together in trials. Nodes are colored by category - red for Oncology and blue for Other conditions.")
            
            # Check if networkx is installed
            try:
                import networkx
                
                # Add slider for minimum co-occurrence
                min_co_occurrence = st.slider(
                    "Minimum Co-occurrence Threshold",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Minimum number of times conditions must co-occur to be included in the network"
                )
                
                fig_network = plot_conditions_network(conditions_dict, min_co_occurrence)
                if fig_network:
                    st.plotly_chart(fig_network, use_container_width=True)
                else:
                    st.info(f"No conditions co-occur at least {min_co_occurrence} times in the selected trials.")
            except ImportError:
                st.error("The networkx library is required for the conditions network visualization.")
                st.info("Please install networkx with: `pip install networkx` and restart the application.")
                
                # Display alternative visualization - simple co-occurrence matrix
                st.write("### Top Condition Co-occurrences (Alternative View)")
                
                # Count co-occurrences
                co_occurrences = {}
                for conditions in conditions_dict.values():
                    if len(conditions) > 1:
                        for i, cond1 in enumerate(conditions):
                            for cond2 in conditions[i+1:]:
                                pair = tuple(sorted([cond1, cond2]))
                                if pair in co_occurrences:
                                    co_occurrences[pair] += 1
                                else:
                                    co_occurrences[pair] = 1
                
                # Show top co-occurrences as a table
                if co_occurrences:
                    co_occur_df = pd.DataFrame([
                        {
                            "Condition 1": pair[0], 
                            "Category 1": classify_condition(pair[0]),
                            "Condition 2": pair[1], 
                            "Category 2": classify_condition(pair[1]),
                            "Co-occurrences": count
                        }
                        for pair, count in sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)[:15]
                    ])
                    st.dataframe(co_occur_df, use_container_width=True)
                else:
                    st.info("No condition co-occurrences found in the selected trials.")
        else:
            st.info("No condition data available for the selected trials.")
    
    with tabs[3]:
        st.write("### Interventions by Condition")
        if conditions_dict and interventions_df is not None and not interventions_df.empty:
            # Add slider for number of top conditions
            top_n = st.slider(
                "Number of Top Conditions per Category",
                min_value=5,
                max_value=20,
                value=8,
                help="Number of top conditions per category to include in the heatmap",
                key="heatmap_conditions_count"
            )
            
            # Use improved heatmap function that colors condition names by category
            fig_heatmap = plot_interventions_by_condition(
                interventions_df, 
                conditions_dict,
                top_n
            )
            
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.info("Conditions are color-coded: red for Oncology and blue for Other conditions.")
                
                # Add summary of intervention types by category
                st.write("### Intervention Types by Condition Category")
                
                # Prepare data
                interventions_by_category = {}
                
                # Create mapping of nct_id to intervention types
                nct_to_interventions = {}
                for _, row in interventions_df.iterrows():
                    if row['nct_id'] not in nct_to_interventions:
                        nct_to_interventions[row['nct_id']] = []
                    nct_to_interventions[row['nct_id']].append(row['intervention_type'])
                
                for nct_id, intervention_types in nct_to_interventions.items():
                    if nct_id in filtered_nct_ids and nct_id in conditions_dict:
                        trial_conditions = conditions_dict[nct_id]
                        for condition in trial_conditions:
                            category = classify_condition(condition)
                            if category not in interventions_by_category:
                                interventions_by_category[category] = Counter()
                            interventions_by_category[category].update(intervention_types)
                
                # Create data for bar chart
                categories = []
                intervention_types = []
                counts = []
                
                for category, counter in interventions_by_category.items():
                    for int_type, count in counter.items():
                        categories.append(category)
                        intervention_types.append(int_type)
                        counts.append(count)
                
                chart_df = pd.DataFrame({
                    'Category': categories,
                    'InterventionType': intervention_types,
                    'Count': counts
                })
                
                if not chart_df.empty:
                    # Create grouped bar chart
                    fig = px.bar(
                        chart_df,
                        x='InterventionType',
                        y='Count',
                        color='Category',
                        barmode='group',
                        title='Intervention Types by Condition Category',
                        color_discrete_map={'Oncology': '#FF4136', 'Other': '#0074D9'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for intervention-condition heatmap.")
        else:
            st.info("Condition and intervention data are required for this visualization.")