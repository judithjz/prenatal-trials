"""
API utility functions for the Pediatric Clinical Trials app.
This module contains functions for API interactions and data exchange.
"""

import streamlit as st
import pandas as pd
import base64
import requests
from typing import Dict, Any, Optional


def get_download_link(df: pd.DataFrame, filename: str = "pediatric_trials.csv", button_text: str = "Download CSV") -> str:
    """
    Generate a download link for a DataFrame.
    
    Args:
        df: DataFrame to download
        filename: Name of the file to download
        button_text: Text to display on the download button
        
    Returns:
        HTML string containing the download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
    return href


def classify_rare_disease(trial_data: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Classify if a trial is for a rare disease using the Anthropic API.
    
    Args:
        trial_data: Dictionary containing trial data
        api_key: Anthropic API key
        
    Returns:
        Dictionary containing classification results
    """
    if not api_key:
        return {"success": False, "error": "Anthropic API key not provided"}
    
    # Prepare conditions text
    conditions_text = ", ".join(trial_data.get('conditions', ['Not provided']))
    
    # Prepare prompt for API
    prompt = f"""
    Please classify whether the following clinical trial is for a rare disease based on accepted criteria.
    
    A rare disease is typically defined as:
    - In the US: affecting fewer than 200,000 people
    - In Europe: affecting no more than 1 in 2,000 people
    - Generally characterized by low prevalence, chronicity, and often genetic origin
    
    Clinical Trial Information:
    - Official Title: {trial_data.get('official_title', 'Not provided')}
    - Brief Title: {trial_data.get('brief_title', 'Not provided')}
    - Conditions: {conditions_text}
    - Brief Summary: {trial_data.get('brief_summary', 'Not provided')}
    - Detailed Description: {trial_data.get('detailed_description', 'Not provided')}
    
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
            json=payload
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
    