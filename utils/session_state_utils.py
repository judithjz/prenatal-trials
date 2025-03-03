"""
Session state management utilities for the Pediatric Clinical Trials app.
This module centralizes session state initialization and access.
"""

import streamlit as st
from typing import Any, Dict, List, Optional, Tuple, Union


def initialize_session_state():
    """
    Initialize all session state variables used across the app.
    Creates default values for any missing variables.
    """
    # Core data variables
    if 'pediatric_trials' not in st.session_state:
        st.session_state.pediatric_trials = None
    
    if 'filtered_trials' not in st.session_state:
        st.session_state.filtered_trials = None
    
    if 'facilities_df' not in st.session_state:
        st.session_state.facilities_df = None
    
    if 'city_data' not in st.session_state:
        st.session_state.city_data = None
    
    # Metadata dictionaries
    if 'conditions_dict' not in st.session_state:
        st.session_state.conditions_dict = {}
    
    if 'keywords_dict' not in st.session_state:
        st.session_state.keywords_dict = {}
    
    # Filter state variables
    if 'status_filter' not in st.session_state:
        st.session_state.status_filter = ["RECRUITING"]
    
    if 'phase_filter' not in st.session_state:
        st.session_state.phase_filter = []
    
    if 'year_filter' not in st.session_state:
        st.session_state.year_filter = None  # Will be set based on available data
    
    if 'keyword_filter' not in st.session_state:
        st.session_state.keyword_filter = ""
    
    if 'condition_filter' not in st.session_state:
        st.session_state.condition_filter = ""
    
    # Rare disease classifier variables
    if 'classification_started' not in st.session_state:
        st.session_state.classification_started = False
    
    if 'showing_classification' not in st.session_state:
        st.session_state.showing_classification = False
    
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = None
    
    if 'trials_to_classify' not in st.session_state:
        st.session_state.trials_to_classify = []
    
    # Navigation variables
    if 'page' not in st.session_state:
        st.session_state.page = "search"  # Default page
    
    if 'selected_trial_id' not in st.session_state:
        st.session_state.selected_trial_id = None
    
    # Utility variables
    if 'need_data_reload' not in st.session_state:
        st.session_state.need_data_reload = True
    
    if 'filter_options' not in st.session_state:
        st.session_state.filter_options = None


def safe_get_session_state(key: str, default: Any = None) -> Any:
    """
    Safely get a value from session state with a default if not present.
    
    Args:
        key: Key to retrieve from session state
        default: Default value to return if key not in session state
        
    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)


def set_session_state(key: str, value: Any) -> None:
    """
    Safely set a value in session state.
    
    Args:
        key: Key to set in session state
        value: Value to set for the key
    """
    st.session_state[key] = value


def update_filter_state(status: Optional[List[str]] = None, 
                        phase: Optional[List[str]] = None,
                        year: Optional[Tuple[int, int]] = None,
                        keyword: Optional[str] = None,
                        condition: Optional[str] = None) -> None:
    """
    Update the filter state variables in session state.
    Only updates provided values, leaving others unchanged.
    
    Args:
        status: Status filter list
        phase: Phase filter list
        year: Year filter tuple (min_year, max_year)
        keyword: Keyword search string
        condition: Condition search string
    """
    if status is not None:
        st.session_state.status_filter = status
    
    if phase is not None:
        st.session_state.phase_filter = phase
    
    if year is not None:
        st.session_state.year_filter = year
    
    if keyword is not None:
        st.session_state.keyword_filter = keyword
    
    if condition is not None:
        st.session_state.condition_filter = condition
        
    # Mark that we need to refilter data
    st.session_state.need_refilter = True


def reset_classifier_state() -> None:
    """
    Reset the classifier state variables.
    Use this when starting a new classification process.
    """
    st.session_state.classification_started = False
    st.session_state.showing_classification = False
    st.session_state.classification_results = None
    st.session_state.trials_to_classify = []


def navigate_to_page(page: str, trial_id: Optional[str] = None) -> None:
    """
    Navigate to a specific page in the app.
    
    Args:
        page: Page name to navigate to
        trial_id: Optional trial ID to select
    """
    st.session_state.page = page
    
    if trial_id is not None:
        st.session_state.selected_trial_id = trial_id
        