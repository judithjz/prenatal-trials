import streamlit as st
import psycopg2
import pandas as pd
import requests
import json
import os

# Access the secrets
db_secrets = st.secrets["database"]

def connect_to_database():
    """Establish connection to the AACT database."""
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

def fetch_detailed_trials(conn, keyword=None, condition=None, status=None, limit=100):
    """
    Fetch detailed trial information including official title, conditions, descriptions, keywords and interventions
    with optional filtering parameters.
    """
    query = """
    SELECT DISTINCT
        s.nct_id,
        s.official_title,
        s.brief_title,
        s.overall_status,
        s.phase,
        s.study_type,
        s.start_date,
        dd.description as detailed_description
    FROM ctgov.studies s
    LEFT JOIN ctgov.detailed_descriptions dd ON s.nct_id = dd.nct_id
    """
    
    # Add WHERE clauses if filters are provided
    where_clauses = []
    params = []
    
    if keyword:
        query += """
        JOIN ctgov.keywords k ON s.nct_id = k.nct_id
        """
        where_clauses.append("k.name ILIKE %s")
        params.append(f"%{keyword}%")
    
    if condition:
        query += """
        JOIN ctgov.conditions c ON s.nct_id = c.nct_id
        """
        where_clauses.append("c.name ILIKE %s")
        params.append(f"%{condition}%")
    
    if status:
        where_clauses.append("s.overall_status = %s")
        params.append(status)
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    query += f"""
    ORDER BY s.start_date DESC
    LIMIT {limit};
    """
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        st.error(f"Error fetching detailed trial data: {e}")
        return pd.DataFrame()

def fetch_conditions_for_trial(conn, nct_id):
    """Fetch conditions for a specific trial"""
    query = """
    SELECT name
    FROM ctgov.conditions
    WHERE nct_id = %s
    ORDER BY id;
    """
    try:
        df = pd.read_sql_query(query, conn, params=[nct_id])
        return df
    except Exception as e:
        st.error(f"Error fetching conditions: {e}")
        return pd.DataFrame()

def fetch_keywords_for_trial(conn, nct_id):
    """Fetch keywords for a specific trial"""
    query = """
    SELECT name
    FROM ctgov.keywords
    WHERE nct_id = %s
    ORDER BY id;
    """
    try:
        df = pd.read_sql_query(query, conn, params=[nct_id])
        return df
    except Exception as e:
        st.error(f"Error fetching keywords: {e}")
        return pd.DataFrame()

def fetch_interventions_for_trial(conn, nct_id):
    """Fetch interventions for a specific trial"""
    query = """
    SELECT intervention_type, name, description
    FROM ctgov.interventions
    WHERE nct_id = %s
    ORDER BY id;
    """
    try:
        df = pd.read_sql_query(query, conn, params=[nct_id])
        return df
    except Exception as e:
        st.error(f"Error fetching interventions: {e}")
        return pd.DataFrame()

def classify_rare_disease(trial_data):
    """
    Send trial data to Anthropic API to classify if it's a rare disease trial
    
    Return:
        dict: Classification results including is_rare_disease and explanation
    """
    # Get API key from Streamlit secrets
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    if not anthropic_api_key:
        st.error("Anthropic API key not configured. Please add it to your Streamlit secrets.")
        return {"error": "API key not configured"}
    
    # Prepare trial data for classification
    prompt = f"""
    Please classify whether the following clinical trial is for a rare disease based on accepted criteria.
    
    A rare disease is typically defined as:
    - In the US: affecting fewer than 200,000 people
    - In Europe: affecting no more than 1 in 2,000 people
    - Generally characterized by low prevalence, chronicity, and often genetic origin
    
    Clinical Trial Information:
    - Official Title: {trial_data.get('official_title', 'Not provided')}
    - Brief Title: {trial_data.get('brief_title', 'Not provided')}
    - Conditions: {', '.join(trial_data.get('conditions', ['Not provided']))}
    - Keywords: {', '.join(trial_data.get('keywords', ['Not provided']))}
    - Detailed Description: {trial_data.get('detailed_description', 'Not provided')}
    - Interventions: {'; '.join([f"{i.get('intervention_type', '')}: {i.get('name', '')}" for i in trial_data.get('interventions', [])])}
    
    Based on this information, is this a clinical trial for a rare disease? Please provide:
    1. A yes/no classification
    2. Your reasoning and explanation
    3. Confidence level (low/medium/high)
    4. Any specific rare disease indicators identified
    """
    
    # API call configuration
    headers = {
        "x-api-key": anthropic_api_key,
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
            return {
                "success": True,
                "classification_text": result["content"][0]["text"],
                "raw_response": result
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

def main():
    st.title("Rare Disease Trial Classification")
    st.write("This tool helps identify clinical trials focused on rare diseases by analyzing trial data and using AI classification.")
    
    # Connect to the database
    conn = connect_to_database()
    if not conn:
        st.error("Failed to connect to the database. Check your credentials and try again.")
        return
    
    with conn:
        # Search and filter section
        st.subheader("Search for Clinical Trials")
        
        col1, col2 = st.columns(2)
        with col1:
            keyword_search = st.text_input("Search by Keyword:", "")
        with col2:
            condition_search = st.text_input("Search by Condition:", "")
        
        status_options = ["RECRUITING", "ACTIVE, NOT RECRUITING", "COMPLETED", "TERMINATED", "WITHDRAWN", "UNKNOWN STATUS"]
        selected_status = st.selectbox("Filter by Status:", ["Any"] + status_options)
        status_filter = None if selected_status == "Any" else selected_status
        
        if st.button("Search Trials"):
            st.session_state.search_results = fetch_detailed_trials(
                conn, 
                keyword=keyword_search, 
                condition=condition_search, 
                status=status_filter
            )
        
        # Display search results
        if 'search_results' in st.session_state and not st.session_state.search_results.empty:
            st.subheader("Search Results")
            st.write(f"Found {len(st.session_state.search_results)} trials matching your criteria")
            
            # Display the dataframe with minimal columns for selection
            st.dataframe(
                st.session_state.search_results[['nct_id', 'brief_title', 'overall_status', 'phase']],
                use_container_width=True
            )
            
            # Select a trial for detailed analysis
            selected_trial = st.selectbox(
                "Select a trial for detailed analysis:",
                options=st.session_state.search_results['nct_id'].tolist(),
                format_func=lambda nct_id: f"{nct_id} - {st.session_state.search_results[st.session_state.search_results['nct_id'] == nct_id]['brief_title'].iloc[0][:50]}..."
            )
            
            if selected_trial:
                # Fetch additional information for the selected trial
                trial_row = st.session_state.search_results[st.session_state.search_results['nct_id'] == selected_trial].iloc[0]
                
                conditions_df = fetch_conditions_for_trial(conn, selected_trial)
                keywords_df = fetch_keywords_for_trial(conn, selected_trial)
                interventions_df = fetch_interventions_for_trial(conn, selected_trial)
                
                # Display detailed information
                st.subheader("Trial Details")
                st.write(f"**NCT ID:** {selected_trial}")
                st.write(f"**Brief Title:** {trial_row['brief_title']}")
                if pd.notna(trial_row['official_title']):
                    st.write(f"**Official Title:** {trial_row['official_title']}")
                st.write(f"**Status:** {trial_row['overall_status']}")
                st.write(f"**Phase:** {trial_row['phase'] if pd.notna(trial_row['phase']) else 'Not Applicable'}")
                
                # Display conditions
                if not conditions_df.empty:
                    st.write("**Conditions:**")
                    for _, row in conditions_df.iterrows():
                        st.write(f"- {row['name']}")
                
                # Display keywords
                if not keywords_df.empty:
                    st.write("**Keywords:**")
                    st.write(", ".join(keywords_df['name'].tolist()))
                
                # Display interventions
                if not interventions_df.empty:
                    st.write("**Interventions:**")
                    for _, row in interventions_df.iterrows():
                        intervention_text = f"- {row['intervention_type']}: {row['name']}"
                        if pd.notna(row['description']):
                            intervention_text += f" ({row['description']})"
                        st.write(intervention_text)
                
                # Display detailed description (collapsed by default)
                if pd.notna(trial_row['detailed_description']):
                    with st.expander("Detailed Description"):
                        st.write(trial_row['detailed_description'])
                
                # Prepare data for classification
                trial_data = {
                    'nct_id': selected_trial,
                    'brief_title': trial_row['brief_title'],
                    'official_title': trial_row['official_title'] if pd.notna(trial_row['official_title']) else '',
                    'detailed_description': trial_row['detailed_description'] if pd.notna(trial_row['detailed_description']) else '',
                    'conditions': conditions_df['name'].tolist() if not conditions_df.empty else [],
                    'keywords': keywords_df['name'].tolist() if not keywords_df.empty else [],
                    'interventions': interventions_df.to_dict('records') if not interventions_df.empty else []
                }
                
                # Button to classify the trial
                if st.button("Classify as Rare Disease"):
                    with st.spinner("Analyzing trial data for rare disease indicators..."):
                        classification_result = classify_rare_disease(trial_data)
                        
                        if classification_result.get('success', False):
                            st.success("Classification complete!")
                            st.subheader("AI Classification Result")
                            st.markdown(classification_result['classification_text'])
                            
                            # Save the classification to the session state
                            st.session_state.classification_result = classification_result
                            
                            # Link to ClinicalTrials.gov
                            st.markdown("---")
                            st.markdown(f"[View this trial on ClinicalTrials.gov](https://clinicaltrials.gov/study/{selected_trial})")
                        else:
                            st.error("Classification failed")
                            st.write(classification_result.get('error', 'Unknown error'))
                            if 'details' in classification_result:
                                with st.expander("Error Details"):
                                    st.text(classification_result['details'])
        else:
            if 'search_results' in st.session_state:
                st.info("No trials found matching your search criteria. Try broadening your search.")
            else:
                st.info("Use the search options above to find clinical trials to analyze.")

if __name__ == "__main__":
    main()
    