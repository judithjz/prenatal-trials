"""
Filtering utility functions for the Pediatric Clinical Trials app.
This module contains functions for filtering and processing trial data.
"""

import pandas as pd
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple, Union, Any


def filter_trials_by_criteria(df: pd.DataFrame, 
                             status_filter: Optional[Union[str, List[str]]] = None, 
                             date_filter: Optional[Tuple[date, date]] = None,
                             keyword_filter: Optional[str] = None,
                             gender_filter: Optional[str] = None,
                             age_filter: Optional[str] = None,
                             type_filter: Optional[str] = None,
                             condition_filter: Optional[str] = None,
                             conditions_dict: Optional[Dict[str, List[str]]] = None,
                             keywords_dict: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """
    Filter the trials DataFrame based on various criteria.    
    Args:
        df: DataFrame containing trials data
        status_filter: Status or list of statuses to filter by
        date_filter: Tuple of (min_date, max_date) to filter by
        keyword_filter: Keyword to search for in trial titles, descriptions, and keywords
        condition_filter: Condition to search for in trial conditions
        gender_filter: All, Male, Female
        age_filter: All, < 18, 18+
        type_filter: Interventional, Observational, Expanded access
        conditions_dict: Dictionary mapping NCT IDs to lists of conditions
        keywords_dict: Dictionary mapping NCT IDs to lists of keywords
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
        
    filtered_df = df.copy()
    
    # Apply status filter (single value or list)
    if status_filter:
        if isinstance(status_filter, list):
            # For multiselect filter (list of statuses)
            filtered_df = filtered_df[filtered_df['overall_status'].isin(status_filter)]
        else:
            # For single value filter
            filtered_df = filtered_df[filtered_df['overall_status'] == status_filter]
    
    # Apply type filter (single value or list)
    if type_filter:
        filtered_df = filtered_df[filtered_df['type'] == type_filter]
    
    # Apply date filter (tuple for range or single year)
    if date_filter and isinstance(date_filter, tuple) and len(date_filter) == 2:
        # Date range filter (min_date, max_date)
            min_date, max_date = date_filter
            filtered_df = filtered_df[(filtered_df['start_date'] >= min_date) & 
                                     (filtered_df['start_date'] <= max_date)]

    if age_filter:
      if age_filter == 'PEDIATRIC':
          filtered_df = filtered_df[filtered_df['max_age_in_months'] < 216]
      elif age_filter == 'ADULT':
          filtered_df = filtered_df[filtered_df['min_age_in_months'] >= 216]
  
    if gender_filter:
        filtered_df = filtered_df[filtered_df['gender'] == gender_filter]
    
    # For keyword and condition filters, we need more complex filtering
    if keyword_filter or condition_filter:
        nct_ids_to_keep = []
        
        for nct_id in filtered_df['nct_id']:
            keep_this_trial = True
            
            # Check if trial title, description, or keywords contain the keyword
            if keyword_filter:
                trial_row = filtered_df[filtered_df['nct_id'] == nct_id].iloc[0]
                
                # Check in title and description fields
                title_contains = keyword_filter.lower() in str(trial_row.get('brief_title', '')).lower()
                official_title_contains = keyword_filter.lower() in str(trial_row.get('official_title', '')).lower()
                summary_contains = keyword_filter.lower() in str(trial_row.get('brief_summary', '')).lower()
                description_contains = keyword_filter.lower() in str(trial_row.get('detailed_description', '')).lower()
                
                # Check in keywords if available
                keywords_contain = False
                if keywords_dict and nct_id in keywords_dict:
                    keywords = [k.lower() for k in keywords_dict[nct_id]]
                    keywords_contain = any(keyword_filter.lower() in k for k in keywords)
                
                # Keep if any of these conditions are met
                keyword_match = (title_contains or official_title_contains or 
                               summary_contains or description_contains or keywords_contain)
                
                if not keyword_match:
                    keep_this_trial = False
            
            # Check if trial conditions contain the condition filter
            if condition_filter and keep_this_trial:
                condition_match = False
                
                if conditions_dict and nct_id in conditions_dict:
                    conditions = [c.lower() for c in conditions_dict[nct_id]]
                    condition_match = any(condition_filter.lower() in condition for condition in conditions)
                
                if not condition_match:
                    keep_this_trial = False
            
            if keep_this_trial:
                nct_ids_to_keep.append(nct_id)
        
        # Filter the dataframe to only include the trials that passed all filters
        filtered_df = filtered_df[filtered_df['nct_id'].isin(nct_ids_to_keep)]
    
    return filtered_df
    
