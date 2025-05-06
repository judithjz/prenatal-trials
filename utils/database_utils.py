"""
Database utility functions for the Pediatric Clinical Trials app.
This module handles all database interactions and query operations.
"""

from contextlib import contextmanager
import streamlit as st
import pandas as pd
import re
import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import wraps
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def connect_to_database() -> Optional[Engine]:
    """
    Create a SQLAlchemy engine for database connection.
    
    Returns:
        SQLAlchemy engine or None if connection fails
    """
    try:
        # Try to access secrets, fall back to environment variables if not available
        try:
            # Check if st.secrets is available
            db_secrets = st.secrets["database"]
            
            host = db_secrets["host"]
            port = db_secrets["port"]
            dbname = db_secrets["dbname"]
            user = db_secrets["user"]
            password = db_secrets["password"]
        except (KeyError, AttributeError):
            # Fall back to environment variables
            host = os.getenv('DB_HOST')
            port = os.getenv('DB_PORT')
            dbname = os.getenv('DB_NAME')
            user = os.getenv('DB_USER')
            password = os.getenv('DB_PASSWORD')
        
        # Create SQLAlchemy engine
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(db_url, pool_pre_ping=True)
        
        # Test connection - create a connection first, then use as context manager
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.commit()
            
        return engine
    except Exception as e:
        st.error("Unable to connect to database. Please check your connection settings.")
        logging.error(f"Database connection error: {e}")
        return None


def handle_database_query(query_function, *args, error_message="Error executing database query", **kwargs):
    """
    Execute a database query with proper error handling.
    """
    try:
        result = query_function(*args, **kwargs)
        return result
    except Exception as e:
        # Log the detailed error for debugging but don't show to users
        logging.error(f"Database error in {query_function.__name__}: {str(e)}")
        
        # Show a sanitized message to users
        st.error(f"{error_message}")
        return None

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
def fetch_conditions_for_trials(_engine: Engine, nct_ids: List[str]) -> pd.DataFrame:
    """
    Fetch conditions for the specified trial IDs.
    
    Args:
        _engine: SQLAlchemy engine
        nct_ids: List of NCT IDs to fetch conditions for
        
    Returns:
        DataFrame containing conditions for the specified trials
    """
    if not nct_ids:
        return pd.DataFrame()
    
    try:
        query = text("""
        SELECT nct_id, name
        FROM ctgov.conditions
        WHERE nct_id IN :nct_ids
        ORDER BY nct_id;
        """)
        
        df = pd.read_sql_query(query, _engine, params={"nct_ids": tuple(nct_ids)})
        return df
    except SQLAlchemyError as e:
        st.error(f"Error fetching conditions")
        logging.error(f"Error fetching conditions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_keywords_for_trials(_engine: Engine, nct_ids: List[str]) -> pd.DataFrame:
    """
    Fetch keywords for the specified trial IDs.
    
    Args:
        _engine: SQLAlchemy engine
        nct_ids: List of NCT IDs to fetch keywords for
        
    Returns:
        DataFrame containing keywords for the specified trials
    """
    if not nct_ids:
        return pd.DataFrame()
    
    try:
        query = text("""
        SELECT nct_id, name AS keyword
        FROM ctgov.keywords
        WHERE nct_id IN :nct_ids
        ORDER BY nct_id;
        """)
        
        df = pd.read_sql_query(query, _engine, params={"nct_ids": tuple(nct_ids)})
        return df
    except SQLAlchemyError as e:
        st.error(f"Error fetching keywords")
        logging.error(f"Error fetching keywords: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_interventions_for_trials(_engine: Engine, nct_ids: List[str]) -> pd.DataFrame:
    """
    Fetch interventions for the specified trial IDs.
    
    Args:
        _engine: SQLAlchemy engine
        nct_ids: List of NCT IDs to fetch interventions for
        
    Returns:
        DataFrame containing interventions for the specified trials
    """
    if not nct_ids:
        return pd.DataFrame()
    
    try:
        query = text("""
        SELECT nct_id, intervention_type, name
        FROM ctgov.interventions
        WHERE nct_id IN :nct_ids
        ORDER BY nct_id;
        """)
        
        df = pd.read_sql_query(query, _engine, params={"nct_ids": tuple(nct_ids)})
        return df
    except SQLAlchemyError as e:
        st.error(f"Error fetching interventions")
        logging.error(f"Error fetching interventions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_facilities_for_trials(_engine: Engine, 
                         nct_ids: Optional[List[str]] = None,
                         country: str = 'Canada',
                         pediatric_only: bool = True) -> pd.DataFrame:
    """
    Unified function to fetch facilities data with various filtering options.
    """
    try:
        # Start with a base query
        base_query = """
        SELECT
            s.nct_id,
            f.city,
            f.state,
            f.country,
            f.name as facility_name,
            f.status as facility_status
        FROM ctgov.studies s
        JOIN ctgov.facilities f ON s.nct_id = f.nct_id
        WHERE f.country = :country AND f.city IS NOT NULL AND f.city != ''
        """
        
        params = {"country": country}
        
        # Add NCT IDs filter if provided
        if nct_ids:
            if len(nct_ids) == 0:
                return pd.DataFrame()
                
            base_query += " AND s.nct_id IN :nct_ids"
            params["nct_ids"] = tuple(nct_ids)
        
        # Complete the query
        base_query += " ORDER BY s.nct_id, f.city"
        query = text(base_query)
        
        # Execute the query
        df = pd.read_sql_query(query, _engine, params=params)
        
        # If pediatric only and we have a list of pediatric trial IDs, filter after query
        if pediatric_only and nct_ids:
            # We already filtered for pediatric trials in the nct_ids list
            return df
        elif pediatric_only:
            # Get the list of pediatric trial IDs and filter the facilities
            pediatric_trials = fetch_pediatric_trials_in_canada(_engine)
            pediatric_nct_ids = set(pediatric_trials['nct_id'].tolist())
            df = df[df['nct_id'].isin(pediatric_nct_ids)]
            
        return df
    except SQLAlchemyError as e:
        st.error(f"Error fetching facilities data")
        logging.error(f"Error fetching facilities data: {e}")
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
