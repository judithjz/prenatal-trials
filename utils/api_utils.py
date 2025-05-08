"""
API utility functions for the Prenatal Clinical Trials app.
This module contains functions for API interactions and data exchange.
"""

import streamlit as st
import pandas as pd
import base64
import requests
from typing import Dict, Any, Optional
import time
import os


def get_download_link(df: pd.DataFrame, filename: str = "prenatal_trials.csv", 
                     button_text: str = "Download CSV", max_size_mb: int = 10) -> str:
    """Generate a download link for a DataFrame."""
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        return "Error: Invalid DataFrame"
        
    # Validate filename to prevent directory traversal
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename != filename:
        safe_filename = "data.csv"
        
    # Check size before encoding
    csv = df.to_csv(index=False)
    size_mb = len(csv.encode()) / (1024 * 1024)
    #if size_mb > max_size_mb:
    #    return f"Error: Data size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
    
    # Encode and create link
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{safe_filename}">{button_text}</a>'
    return href


def prenatal_disease(trial_data: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Classify if a trial is for a prenatal population using the Anthropic API."""
    # Check if we're exceeding rate limits
    if not _check_rate_limit():
        return {"success": False, "error": "Rate limit exceeded, please try again later"}
    
    # Validate and sanitize inputs
    if not isinstance(trial_data, dict):
        return {"success": False, "error": "Invalid trial data format"}
        
    # Sanitize and truncate inputs to prevent prompt injection
    def sanitize(text):
        if not text or not isinstance(text, str):
            return "Not provided"
        # Truncate long text fields
        return text[:5000].replace("{", "").replace("}", "")
    
    conditions = [sanitize(c) for c in trial_data.get('conditions', [])] if isinstance(trial_data.get('conditions'), list) else ["Not provided"]
    conditions_text = ", ".join(conditions)
    
    # Prepare prompt for API
    prompt = f"""
    Please classify whether the following clinical trial designed for prenatal population based on accepted criteria.
    
    Trials for a prenatal population include:
    
    Clinical Trial Information:
    - Official Title: {sanitize(trial_data.get('official_title'))}
    - Brief Title: {sanitize(trial_data.get('brief_title'))}
    - Conditions: {conditions_text}
    - Brief Summary: {sanitize(trial_data.get('brief_summary'))}
    - Detailed Description: {sanitize(trial_data.get('detailed_description'))}
    
    Based on this information, is this a clinical trial for the prenatal population? 
    
    Please structure your response exactly like this:
    
    ## Classification
    [Start with a clear "Yes" or "No" statement about whether this is a prenatal trial]
    
    ## Reasoning
    [Explain your reasoning]
    
    ## Confidence Level
    [State your confidence level as low/medium/high]
    
    ## Prenatal Indicators
    [List any specific indicators if applicable]
    """
    
    # API call configuration
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30,  # Add timeout
            verify=True  # Verify SSL certificates
        )
        
        if response.status_code == 200:
            result = response.json()
            classification_text = result["content"][0]["text"]
            
            # Determine if it's a prenatal trial
            text_lower = classification_text.lower()
            is_rare = False
            
            if "classification: yes" in text_lower[:500] or "classification: prenatal" in text_lower[:500]:
                is_rare = True
            elif "classification: no" in text_lower[:500] or "classification: not prenatal" in text_lower[:500]:
                is_rare = False
            elif "yes, this is" in text_lower[:300] and "prenatal" in text_lower[:300]:
                is_rare = True
            elif "no, this is not" in text_lower[:300] and "prenatal" in text_lower[:300]:
                is_rare = False
            elif "this is a rare disease" in text_lower[:300]:
                is_rare = True
            elif "this is not a prenatal trial" in text_lower[:300]:
                is_rare = False
            
            return {
                "success": True,
                "classification_text": classification_text,
                "is_rare": is_rare
            }
        else:
            return {
                "success": False,
                "error": f"API Error: {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Exception: {str(e)}"
        }

def classify_rare_disease(trial_data: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Classify if a trial is for a rare disease using the Anthropic API."""
    # Check if we're exceeding rate limits
    if not _check_rate_limit():
        return {"success": False, "error": "Rate limit exceeded, please try again later"}
    
    # Validate and sanitize inputs
    if not isinstance(trial_data, dict):
        return {"success": False, "error": "Invalid trial data format"}
        
    # Sanitize and truncate inputs to prevent prompt injection
    def sanitize(text):
        if not text or not isinstance(text, str):
            return "Not provided"
        # Truncate long text fields
        return text[:5000].replace("{", "").replace("}", "")
    
    conditions = [sanitize(c) for c in trial_data.get('conditions', [])] if isinstance(trial_data.get('conditions'), list) else ["Not provided"]
    conditions_text = ", ".join(conditions)
    
    # Prepare prompt for API
    prompt = f"""
    Please classify whether the following clinical trial is for a rare disease based on accepted criteria.
    
    A rare disease is typically defined as:
    - In the US: affecting fewer than 200,000 people
    - In Europe: affecting no more than 1 in 2,000 people
    - Generally characterized by low prevalence, chronicity, and often genetic origin
    
    Clinical Trial Information:
    - Official Title: {sanitize(trial_data.get('official_title'))}
    - Brief Title: {sanitize(trial_data.get('brief_title'))}
    - Conditions: {conditions_text}
    - Brief Summary: {sanitize(trial_data.get('brief_summary'))}
    - Detailed Description: {sanitize(trial_data.get('detailed_description'))}
    
    Based on this information, is this a clinical trial for a rare disease? 
    
    Please structure your response exactly like this:
    
    ## Classification
    [Start with a clear "Yes" or "No" statement about whether this is a rare disease trial]
    
    ## Reasoning
    [Explain your reasoning]
    
    ## Confidence Level
    [State your confidence level as low/medium/high]
    
    ## Rare Disease Indicators
    [List any specific indicators if applicable]
    """
    
    # API call configuration
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30,  # Add timeout
            verify=True  # Verify SSL certificates
        )
        
        if response.status_code == 200:
            result = response.json()
            classification_text = result["content"][0]["text"]
            
            # Determine if it's a rare disease
            text_lower = classification_text.lower()
            is_rare = False
            
            if "classification: yes" in text_lower[:500] or "classification: rare disease" in text_lower[:500]:
                is_rare = True
            elif "classification: no" in text_lower[:500] or "classification: not a rare disease" in text_lower[:500]:
                is_rare = False
            elif "yes, this is" in text_lower[:300] and "rare disease" in text_lower[:300]:
                is_rare = True
            elif "no, this is not" in text_lower[:300] and "rare disease" in text_lower[:300]:
                is_rare = False
            elif "this is a rare disease" in text_lower[:300]:
                is_rare = True
            elif "this is not a rare disease" in text_lower[:300]:
                is_rare = False
            
            return {
                "success": True,
                "classification_text": classification_text,
                "is_rare": is_rare
            }
        else:
            return {
                "success": False,
                "error": f"API Error: {response.status_code}",
                "details": response.text
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Exception: {str(e)}"
        }


def _check_rate_limit():
    """Simple rate limiting implementation"""
    if 'api_calls' not in st.session_state:
        st.session_state.api_calls = []
        st.session_state.api_call_limit = 10  # Limit per hour
    
    # Clean up old calls (older than 1 hour)
    current_time = time.time()
    st.session_state.api_calls = [call_time for call_time in st.session_state.api_calls 
                                 if current_time - call_time < 3600]
    
    # Check if we're under the limit
    if len(st.session_state.api_calls) >= st.session_state.api_call_limit:
        return False
    
    # Add this call
    st.session_state.api_calls.append(current_time)
    return True


def extract_classification_statement(text: str) -> str:
    """
    Extract the classification statement from the API response.
    
    Args:
        text: API response text
        
    Returns:
        Classification statement extracted from the text
    """
    lines = text.lower().split('\n')
    for i, line in enumerate(lines):
        if '## classification' in line and i+1 < len(lines):
            return lines[i+1].strip()
    return "Not explicitly stated"
