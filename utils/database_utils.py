"""
Database utility functions for the Pediatric Clinical Trials app.
This module handles all database interactions and query operations.
"""

import streamlit as st
import psycopg2
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import wraps


def connect_to_database() -> Optional[psycopg2.extensions.connection]:
    """
    Establish connection to the AACT database using credentials from Streamlit secrets.
    
    Returns:
        Database connection object or None if connection fails
    """
    # Access the secrets
    db_secrets = st.secrets["database"]
    
    try:
        conn = psycopg2.connect(
            host=db_secrets["host"],
            port=db_secrets["port"],
            dbname=db_secrets["dbname"],
            user=db_secrets["user"],
            password=db_secrets["password"]
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None


def handle_database_query(query_function, *args, error_message="Error executing database query", **kwargs):
    """
    Execute a database query with proper error handling.
    
    Args:
        query_function: Function that executes a database query
        *args: Arguments to pass to the query function
        error_message: Custom error message to display if the query fails
        **kwargs: Keyword arguments to pass to the query function
        
    Returns:
        Result of the query function or None if an error occurs
    """
    try:
        result = query_function(*args, **kwargs)
        return result
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        # Log the error for debugging
        print(f"Database error in {query_function.__name__}: {str(e)}")
        return None


@st.cache_data(ttl=3600)  # Cache for one hour
def fetch_pediatric_trials_in_canada(_conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """
    Fetch data for clinical trials that:
    1. Have at least one site in Canada (from facilities table)
    2. Include participants with minimum age < 18 years
    
    This is the base query that will be used across all pages of the app.
    
    Args:
        conn: Database connection object
        
    Returns:
        DataFrame containing pediatric trial data
    """
    query = """
    SELECT DISTINCT
        s.nct_id,
        s.brief_title,
        s.official_title,
        s.overall_status,
        e.minimum_age,
        s.phase,
        s.study_type,
        s.start_date,
        bs.description as brief_summary,
        dd.description as detailed_description,
        COUNT(DISTINCT f.id) AS num_canadian_sites
    FROM ctgov.studies s
    JOIN ctgov.facilities f ON s.nct_id = f.nct_id
    JOIN ctgov.eligibilities e ON s.nct_id = e.nct_id
    LEFT JOIN ctgov.brief_summaries bs ON s.nct_id = bs.nct_id
    LEFT JOIN ctgov.detailed_descriptions dd ON s.nct_id = dd.nct_id
    WHERE f.country = 'Canada'
    GROUP BY s.nct_id, s.brief_title, s.official_title, s.overall_status, 
             e.minimum_age, s.phase, s.study_type, s.start_date, 
             bs.description, dd.description
    ORDER BY s.start_date DESC;
    """
    
    try:
        df = pd.read_sql_query(query, _conn)
        
        # Parse minimum age to months for filtering
        df['age_in_months'] = df['minimum_age'].apply(parse_age_to_months)
        
        # Filter for pediatric trials (age < 18 years = 216 months)
        pediatric_df = df[df['age_in_months'] < 216].copy()
        
        # Add some useful columns for analysis
        if not pediatric_df.empty and 'start_date' in pediatric_df.columns:
            pediatric_df['start_year'] = pd.to_datetime(pediatric_df['start_date'], errors='coerce').dt.year
        
        return pediatric_df
    except Exception as e:
        st.error(f"Error fetching pediatric trials data: {e}")
        return pd.DataFrame()


def parse_age_to_months(age_str: Optional[str]) -> Optional[float]:
    """
    Parse age string to months.
    
    Examples:
    - "18 Years" -> 18*12 = 216 months
    - "2 Months" -> 2
    - "4 Weeks" -> 1 (rounded)
    - "30 Days" -> 1 (rounded)
    
    Args:
        age_str: Age string to parse
        
    Returns:
        Age in months or None if age_str is invalid
    """
    if not age_str or pd.isna(age_str):
        return None
    
    age_str = age_str.lower().strip()
    
    # Extract numeric value and unit
    match = re.match(r'(\d+\.?\d*)\s*(\w+)', age_str)
    if not match:
        return None
    
    value, unit = match.groups()
    value = float(value)
    
    # Convert to months based on unit
    if 'year' in unit:
        return value * 12
    elif 'month' in unit:
        return value
    elif 'week' in unit:
        return value / 4.3  # Approximate weeks to months
    elif 'day' in unit:
        return value / 30.4  # Approximate days to months
    else:
        return None


@st.cache_data(ttl=3600)
def fetch_conditions_for_trials(_conn: psycopg2.extensions.connection, nct_ids: List[str]) -> pd.DataFrame:
    """
    Fetch conditions for the specified trial IDs.
    
    Args:
        conn: Database connection object
        nct_ids: List of NCT IDs to fetch conditions for
        
    Returns:
        DataFrame containing conditions for the specified trials
    """
    if not nct_ids:
        return pd.DataFrame()
    
    placeholders = ', '.join(['%s'] * len(nct_ids))
    query = f"""
    SELECT nct_id, name
    FROM ctgov.conditions
    WHERE nct_id IN ({placeholders})
    ORDER BY nct_id;
    """
    
    try:
        df = pd.read_sql_query(query, _conn, params=nct_ids)
        return df
    except Exception as e:
        st.error(f"Error fetching conditions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_keywords_for_trials(_conn: psycopg2.extensions.connection, nct_ids: List[str]) -> pd.DataFrame:
    """
    Fetch keywords for the specified trial IDs.
    
    Args:
        conn: Database connection object
        nct_ids: List of NCT IDs to fetch keywords for
        
    Returns:
        DataFrame containing keywords for the specified trials
    """
    if not nct_ids:
        return pd.DataFrame()
    
    placeholders = ', '.join(['%s'] * len(nct_ids))
    query = f"""
    SELECT nct_id, name AS keyword
    FROM ctgov.keywords
    WHERE nct_id IN ({placeholders})
    ORDER BY nct_id;
    """
    
    try:
        df = pd.read_sql_query(query, _conn, params=nct_ids)
        return df
    except Exception as e:
        st.error(f"Error fetching keywords: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_interventions_for_trials(_conn: psycopg2.extensions.connection, nct_ids: List[str]) -> pd.DataFrame:
    """
    Fetch interventions for the specified trial IDs.
    
    Args:
        conn: Database connection object
        nct_ids: List of NCT IDs to fetch interventions for
        
    Returns:
        DataFrame containing interventions for the specified trials
    """
    if not nct_ids:
        return pd.DataFrame()
    
    placeholders = ', '.join(['%s'] * len(nct_ids))
    query = f"""
    SELECT nct_id, intervention_type, name
    FROM ctgov.interventions
    WHERE nct_id IN ({placeholders})
    ORDER BY nct_id;
    """
    
    try:
        df = pd.read_sql_query(query, _conn, params=nct_ids)
        return df
    except Exception as e:
        st.error(f"Error fetching interventions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_facilities_for_trials(_conn: psycopg2.extensions.connection, 
                         nct_ids: Optional[List[str]] = None,
                         country: str = 'Canada',
                         pediatric_only: bool = True) -> pd.DataFrame:
    """
    Unified function to fetch facilities data with various filtering options.
    
    Args:
        _conn: Database connection object
        nct_ids: Optional list of NCT IDs to filter by
        country: Country to filter facilities by (default: 'Canada')
        pediatric_only: Whether to filter for pediatric trials only (default: True)
        
    Returns:
        DataFrame with facility information
    """
    # Build the WHERE clause based on parameters
    where_conditions = [f"f.country = '{country}'", 
                       "f.city IS NOT NULL", 
                       "f.city != ''"]
    
    params = []
    
    # Add NCT IDs filter if provided
    if nct_ids:
        # FIX: Check if nct_ids is empty to avoid SQL errors
        if len(nct_ids) == 0:
            return pd.DataFrame()  # Return empty DataFrame if no NCT IDs
            
        placeholders = ', '.join(['%s'] * len(nct_ids))
        where_conditions.append(f"s.nct_id IN ({placeholders})")
        params.extend(nct_ids)
    
    # Construct the final WHERE clause
    where_clause = " AND ".join(where_conditions)
    
    query = f"""
    SELECT
        s.nct_id,
        f.city,
        f.state,
        f.country,
        f.name as facility_name,
        f.status as facility_status
    FROM ctgov.studies s
    JOIN ctgov.facilities f ON s.nct_id = f.nct_id
    WHERE {where_clause}
    ORDER BY s.nct_id, f.city;
    """
    
    try:
        df = pd.read_sql_query(query, _conn, params=params)
        
        # If pediatric only and we have a list of pediatric trial IDs, filter after query
        if pediatric_only and nct_ids:
            # We already filtered for pediatric trials in the nct_ids list
            return df
        elif pediatric_only:
            # Get the list of pediatric trial IDs and filter the facilities
            pediatric_trials = fetch_pediatric_trials_in_canada(_conn)
            pediatric_nct_ids = set(pediatric_trials['nct_id'].tolist())
            df = df[df['nct_id'].isin(pediatric_nct_ids)]
            
        return df
    except Exception as e:
        st.error(f"Error fetching facilities data: {e}")
        return pd.DataFrame()

def prepare_city_data(facilities_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare city-level aggregation data from facilities dataframe.
    
    Args:
        facilities_df: DataFrame with facility information
    
    Returns:
        DataFrame with city-level aggregation
    """
    if facilities_df.empty:
        return pd.DataFrame()
    
    # Count unique trials per city
    city_counts = facilities_df.groupby('city')['nct_id'].nunique().reset_index()
    city_counts.columns = ['city', 'trial_count']
    
    # Sort by trial count in descending order
    city_counts = city_counts.sort_values('trial_count', ascending=False)
    
    return city_counts


def build_conditions_dict(conditions_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Convert conditions DataFrame to a dictionary for easy lookup.
    
    Args:
        conditions_df: DataFrame containing conditions data
        
    Returns:
        Dictionary mapping NCT IDs to lists of conditions
    """
    conditions_dict = {}
    for _, row in conditions_df.iterrows():
        if row['nct_id'] not in conditions_dict:
            conditions_dict[row['nct_id']] = []
        conditions_dict[row['nct_id']].append(row['name'])
    
    return conditions_dict


def build_keywords_dict(keywords_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Convert keywords DataFrame to a dictionary for easy lookup.
    
    Args:
        keywords_df: DataFrame containing keywords data
        
    Returns:
        Dictionary mapping NCT IDs to lists of keywords
    """
    keywords_dict = {}
    for _, row in keywords_df.iterrows():
        if row['nct_id'] not in keywords_dict:
            keywords_dict[row['nct_id']] = []
        keywords_dict[row['nct_id']].append(row['keyword'])
    
    return keywords_dict
