import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import re

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

def parse_age_to_months(age_str):
    """
    Parse age string to months.
    Examples:
    - "18 Years" -> 18*12 = 216 months
    - "2 Months" -> 2 months
    - "4 Weeks" -> 1 month (rounded)
    - "30 Days" -> 1 month (rounded)
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

def fetch_pediatric_trials_in_canada(conn):
    """
    Fetch data for clinical trials that:
    1. Have at least one site in Canada (from facilities table)
    2. Include participants with minimum age < 18 years
    """
    query = """
    SELECT DISTINCT
        s.nct_id,
        s.brief_title,
        s.overall_status,
        e.minimum_age,
        s.phase,
        s.study_type,
        s.start_date,
        COUNT(DISTINCT f.id) AS num_canadian_sites
    FROM ctgov.studies s
    JOIN ctgov.facilities f ON s.nct_id = f.nct_id
    JOIN ctgov.eligibilities e ON s.nct_id = e.nct_id
    WHERE f.country = 'Canada'
    GROUP BY s.nct_id, s.brief_title, s.overall_status, e.minimum_age, s.phase, s.study_type, s.start_date
    ORDER BY s.start_date DESC;
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        
        # Parse minimum age to months for filtering
        df['age_in_months'] = df['minimum_age'].apply(parse_age_to_months)
        
        # Filter for pediatric trials (age < 18 years = 216 months)
        pediatric_df = df[df['age_in_months'] < 216].copy()
        
        # Add some useful columns for analysis
        if not pediatric_df.empty and 'start_date' in pediatric_df.columns:
            pediatric_df['start_year'] = pd.to_datetime(pediatric_df['start_date'], errors='coerce').dt.year
        
        return pediatric_df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def fetch_keywords_for_trials(conn, nct_ids):
    """Fetch keywords for the specified trial IDs"""
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
        df = pd.read_sql_query(query, conn, params=nct_ids)
        return df
    except Exception as e:
        st.error(f"Error fetching keywords: {e}")
        return pd.DataFrame()

def fetch_interventions_for_trials(conn, nct_ids):
    """Fetch interventions for the specified trial IDs"""
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
        df = pd.read_sql_query(query, conn, params=nct_ids)
        return df
    except Exception as e:
        st.error(f"Error fetching interventions: {e}")
        return pd.DataFrame()

def main():
    st.title("Pediatric Clinical Trials in Canada")
    st.write("This page displays clinical trials with sites in Canada that include participants under 18 years old.")

    # Connect to the database
    conn = connect_to_database()
    if conn:
        with conn:
            # Fetch data
            st.info("Fetching pediatric trial data from the AACT database...")
            pediatric_trials = fetch_pediatric_trials_in_canada(conn)
            
            if not pediatric_trials.empty:
                # Display summary statistics
                st.subheader("Summary Statistics")
                total_trials = len(pediatric_trials)
                st.metric("Total Pediatric Trials in Canada", total_trials)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    active_trials = pediatric_trials[pediatric_trials['overall_status'].isin(['RECRUITING', 'ACTIVE, NOT RECRUITING', 'ENROLLING BY INVITATION'])].shape[0]
                    st.metric("Active Trials", active_trials)
                with col2:
                    recruiting_trials = pediatric_trials[pediatric_trials['overall_status'] == 'RECRUITING'].shape[0]
                    st.metric("Recruiting Trials", recruiting_trials)
                with col3:
                    completed_trials = pediatric_trials[pediatric_trials['overall_status'] == 'COMPLETED'].shape[0]
                    st.metric("Completed Trials", completed_trials)
                
                # Visualization: Trials by Phase
                st.subheader("Trials by Phase")
                phase_counts = pediatric_trials['phase'].fillna('Not Applicable').value_counts().reset_index()
                phase_counts.columns = ['phase', 'count']
                
                fig_phase = px.bar(
                    phase_counts,
                    x='phase',
                    y='count',
                    labels={'phase': 'Trial Phase', 'count': 'Number of Trials'},
                    title='Distribution of Pediatric Trials by Phase',
                    color='phase'
                )
                fig_phase.update_layout(template="plotly_white")
                st.plotly_chart(fig_phase, use_container_width=True)
                
                # Visualization: Trials by Year
                if 'start_year' in pediatric_trials.columns:
                    year_status = pediatric_trials.groupby(['start_year', 'overall_status']).size().reset_index(name='count')
                    year_status = year_status[~year_status['start_year'].isna()]
                    
                    st.subheader("Pediatric Trials by Year and Status")
                    fig_year = px.bar(
                        year_status,
                        x='start_year',
                        y='count',
                        color='overall_status',
                        labels={'start_year': 'Start Year', 'count': 'Number of Trials', 'overall_status': 'Status'},
                        title='Pediatric Clinical Trials by Start Year and Status'
                    )
                    fig_year.update_layout(
                        xaxis_title="Start Year",
                        yaxis_title="Number of Trials",
                        barmode="group",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_year, use_container_width=True)
                
                # Fetch and visualize trial keywords and interventions
                st.subheader("Trial Details")
                
                # Filter options
                status_filter = st.multiselect(
                    "Filter by Status:",
                    options=sorted(pediatric_trials['overall_status'].unique()),
                    default=['RECRUITING']
                )
                
                phase_filter = st.multiselect(
                    "Filter by Phase:",
                    options=sorted(pediatric_trials['phase'].fillna('Not Applicable').unique()),
                    default=[]
                )
                
                # Apply filters
                filtered_trials = pediatric_trials
                if status_filter:
                    filtered_trials = filtered_trials[filtered_trials['overall_status'].isin(status_filter)]
                if phase_filter:
                    filtered_trials = filtered_trials[filtered_trials['phase'].fillna('Not Applicable').isin(phase_filter)]
                
                # Display filtered trial data
                st.dataframe(
                    filtered_trials[['nct_id', 'brief_title', 'overall_status', 'minimum_age', 'phase', 'num_canadian_sites']],
                    use_container_width=True
                )
                
                # Select a trial to show details
                if not filtered_trials.empty:
                    selected_trial = st.selectbox(
                        "Select a trial to view details:",
                        options=filtered_trials['nct_id'].tolist(),
                        format_func=lambda nct_id: f"{nct_id} - {filtered_trials[filtered_trials['nct_id'] == nct_id]['brief_title'].iloc[0][:50]}..."
                    )
                    
                    if selected_trial:
                        st.write("### Trial Details")
                        trial_data = filtered_trials[filtered_trials['nct_id'] == selected_trial].iloc[0]
                        
                        st.write(f"**Title:** {trial_data['brief_title']}")
                        st.write(f"**Status:** {trial_data['overall_status']}")
                        st.write(f"**Phase:** {trial_data['phase'] if pd.notna(trial_data['phase']) else 'Not Applicable'}")
                        st.write(f"**Minimum Age:** {trial_data['minimum_age']}")
                        st.write(f"**Canadian Sites:** {int(trial_data['num_canadian_sites'])}")
                        
                        # Fetch additional data for the selected trial
                        keywords_df = fetch_keywords_for_trials(conn, [selected_trial])
                        interventions_df = fetch_interventions_for_trials(conn, [selected_trial])
                        
                        # Display keywords
                        if not keywords_df.empty:
                            st.write("**Keywords:**")
                            st.write(", ".join(keywords_df['keyword'].tolist()))
                        
                        # Display interventions
                        if not interventions_df.empty:
                            st.write("**Interventions:**")
                            for _, row in interventions_df.iterrows():
                                st.write(f"- {row['intervention_type']}: {row['name']}")
                        
                        # Link to ClinicalTrials.gov
                        st.markdown(f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{selected_trial})")
            else:
                st.warning("No pediatric trials in Canada found in the database.")
    else:
        st.error("Failed to connect to the database. Check your credentials and try again.")

if __name__ == "__main__":
    main()
    