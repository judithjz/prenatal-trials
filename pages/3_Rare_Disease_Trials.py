import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import utility modules
from utils.database_utils import (
    connect_to_database,
    fetch_conditions_for_trials,
    fetch_keywords_for_trials,
    fetch_interventions_for_trials,
    fetch_facilities_for_trials,
    prepare_city_data,
    build_conditions_dict,
    build_keywords_dict
)

from utils.session_state_utils import (
    initialize_session_state,
    safe_get_session_state,
    set_session_state
)

from utils.filtering_utils import (
    filter_trials_by_criteria
)

from utils.visualization_utils import (
    plot_status_distribution,
    plot_yearly_trends,
    render_city_visualization,
    render_summary_metrics,
    render_province_visualization,
    render_interventions_conditions_analysis
)

from utils.api_utils import (
    get_download_link
)


def main():
    # Set page config
    st.set_page_config(
        page_title="Rare Disease Trials in Canada",
        page_icon="🧬",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    st.title("Rare Disease Trials in Canada")
    st.write("This page displays clinical trials that have been classified as rare disease trials.")
    
    # Connect to the database
    conn = connect_to_database()
    if not conn:
        st.error("Failed to connect to the database. Check your credentials and try again.")
        return
    
    try:
        # Check if pre-classified data exists
        try:
            rare_df = pd.read_csv('data/rare_disease_classification.csv')
            if not rare_df.empty:
                st.success(f"Loaded {len(rare_df)} pre-classified rare disease trials.")
            else:
                st.warning("No pre-classified rare disease trials found.")
                return
        except Exception as e:
            st.error(f"Error loading rare disease classification data: {e}")
            st.info("Please run the Rare Disease Classifier first to generate classification data.")
            return
        
        # Filter for rare disease trials only
        rare_df = rare_df[rare_df['is_rare'] == True].copy()
        st.info(f"Found {len(rare_df)} trials classified as rare diseases.")
        
        # Load trial data for these NCT IDs
        rare_nct_ids = rare_df['nct_id'].tolist()
        
        # Fetch trial data from database
        if safe_get_session_state('rare_trials_data') is None:
            with st.spinner("Loading trial data from database..."):
                # Query to get the trial data for the rare disease trials
                nct_placeholders = ','.join(['%s'] * len(rare_nct_ids))
                query = f"""
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
                WHERE s.nct_id IN ({nct_placeholders})
                AND f.country = 'Canada'
                GROUP BY s.nct_id, s.brief_title, s.official_title, s.overall_status, 
                         e.minimum_age, s.phase, s.study_type, s.start_date, 
                         bs.description, dd.description;
                """
                
                try:
                    rare_trials_data = pd.read_sql_query(query, conn, params=rare_nct_ids)
                    
                    # Add year information
                    if not rare_trials_data.empty and 'start_date' in rare_trials_data.columns:
                        rare_trials_data['start_year'] = pd.to_datetime(rare_trials_data['start_date'], errors='coerce').dt.year
                    
                    set_session_state('rare_trials_data', rare_trials_data)
                except Exception as e:
                    st.error(f"Error fetching trial data: {e}")
                    return
        
        # Load filters
        if safe_get_session_state('rare_trials_filters') is None:
            # Fetch additional data needed for filtering
            with st.spinner("Loading additional data for filtering..."):
                # Fetch conditions for trials
                conditions_df = fetch_conditions_for_trials(conn, rare_nct_ids)
                if not conditions_df.empty:
                    conditions_dict = build_conditions_dict(conditions_df)
                else:
                    conditions_dict = {}
                
                # Fetch keywords for trials
                keywords_df = fetch_keywords_for_trials(conn, rare_nct_ids)
                if not keywords_df.empty:
                    keywords_dict = build_keywords_dict(keywords_df)
                else:
                    keywords_dict = {}
                
                # Fetch interventions for trials
                interventions_df = fetch_interventions_for_trials(conn, rare_nct_ids)
                
                set_session_state('rare_conditions_dict', conditions_dict)
                set_session_state('rare_keywords_dict', keywords_dict)
                set_session_state('rare_interventions_df', interventions_df)
                set_session_state('rare_trials_filters', True)
        
        # Create sidebar filters
        st.sidebar.header("Filters")
        
        # Get dataset for filtering
        rare_trials_data = safe_get_session_state('rare_trials_data')
        
        if rare_trials_data is not None and not rare_trials_data.empty:
            # Status filter
            status_options = rare_trials_data['overall_status'].dropna().unique().tolist()
            status_filter = st.sidebar.multiselect(
                "Trial Status:",
                options=status_options,
                default=["RECRUITING"]
            )
            
            # Phase filter
            phase_options = rare_trials_data['phase'].dropna().unique().tolist() + ['Not Applicable']
            phase_filter = st.sidebar.multiselect(
                "Trial Phase:",
                options=phase_options,
                default=[]
            )
            
            # Year filter
            year_options = rare_trials_data['start_year'].dropna().astype(int).unique().tolist()
            if year_options:
                year_filter = st.sidebar.slider(
                    "Start Year Range:",
                    min_value=min(year_options),
                    max_value=max(year_options),
                    value=(min(year_options), max(year_options))
                )
            else:
                year_filter = None
            
            # Keyword and condition search
            keyword_filter = st.sidebar.text_input("Keyword Search:", "")
            keyword_filter = keyword_filter.strip() if keyword_filter else None
            
            condition_filter = st.sidebar.text_input("Condition Search:", "")
            condition_filter = condition_filter.strip() if condition_filter else None
            
            # Apply filters
            filtered_df = filter_trials_by_criteria(
                rare_trials_data,
                status_filter=status_filter,
                phase_filter=phase_filter,
                year_filter=year_filter,
                keyword_filter=keyword_filter,
                condition_filter=condition_filter,
                conditions_dict=safe_get_session_state('rare_conditions_dict', {}),
                keywords_dict=safe_get_session_state('rare_keywords_dict', {})
            )
            
            # Store filtered data in session state
            set_session_state('filtered_rare_trials', filtered_df)
            
            # Display results
            if filtered_df.empty:
                st.warning("No rare disease trials found matching your criteria.")
            else:
                # Display summary statistics
                st.subheader("Summary Statistics")
                render_summary_metrics(filtered_df)
                
                # Create tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Trial List", 
                    "Charts", 
                    "Geographic Distribution", 
                    "Interventions & Conditions",
                    "Trial Details"
                ])
                
                with tab1:
                    st.subheader("Rare Disease Clinical Trials")
                    # Display trials in a dataframe
                    display_cols = ['nct_id', 'brief_title', 'overall_status', 'minimum_age', 'phase', 'start_date', 'num_canadian_sites']
                    st.dataframe(filtered_df[display_cols], use_container_width=True)
                    
                    # Download button
                    st.markdown(get_download_link(filtered_df[display_cols], filename="rare_disease_trials.csv"), unsafe_allow_html=True)
                
                with tab2:
                    st.subheader("Data Visualizations")
                    
                    # Visualization 2: Trials by Status
                    st.write("### Trials by Status")
                    fig_status = plot_status_distribution(filtered_df)
                    st.plotly_chart(fig_status, use_container_width=True)
                    
                    # Visualization 3: Trials by Year
                    if 'start_year' in filtered_df.columns and not filtered_df['start_year'].isna().all():
                        st.write("### Trials by Year")
                        fig_year, fig_area = plot_yearly_trends(filtered_df)
                        if fig_year and fig_area:
                            st.plotly_chart(fig_year, use_container_width=True)
                            st.plotly_chart(fig_area, use_container_width=True)
                
                with tab3:
                    st.subheader("Geographic Distribution")
                    # Use the centralized visualization function for geographic distribution
                    render_city_visualization(filtered_df, conn)
                    # Add province distribution visualization
                    render_province_visualization(filtered_df, conn, title_prefix="Rare Disease")
                
                with tab4:
                    # Use the centralized visualization function for interventions and conditions analysis
                    filtered_nct_ids = filtered_df['nct_id'].tolist()
                    conditions_dict = safe_get_session_state('rare_conditions_dict', {})
                    interventions_df = safe_get_session_state('rare_interventions_df')
                    
                    # Filter to only include the trials from filtered_df
                    filtered_conditions_dict = {
                        nct_id: conditions 
                        for nct_id, conditions in conditions_dict.items() 
                        if nct_id in filtered_nct_ids
                    }
                    
                    filtered_interventions_df = None
                    if interventions_df is not None and not interventions_df.empty:
                        filtered_interventions_df = interventions_df[interventions_df['nct_id'].isin(filtered_nct_ids)]
                    
                    render_interventions_conditions_analysis(
                        filtered_df,
                        conn,
                        title_prefix="Rare Disease",
                        conditions_dict=filtered_conditions_dict,
                        interventions_df=filtered_interventions_df
                    )
                
                with tab5:
                    st.subheader("Trial Details")
                    
                    # Select a trial to show details
                    selected_trial = st.selectbox(
                        "Select a trial to view details:",
                        options=filtered_df['nct_id'].tolist(),
                        format_func=lambda nct_id: f"{nct_id} - {filtered_df[filtered_df['nct_id'] == nct_id]['brief_title'].iloc[0][:50]}..."
                    )
                    
                    if selected_trial:
                        st.write("### Trial Details")
                        trial_data = filtered_df[filtered_df['nct_id'] == selected_trial].iloc[0]
                        
                        st.write(f"**NCT ID:** {selected_trial}")
                        st.write(f"**Title:** {trial_data['brief_title']}")
                        if pd.notna(trial_data['official_title']):
                            st.write(f"**Official Title:** {trial_data['official_title']}")
                        st.write(f"**Status:** {trial_data['overall_status']}")
                        st.write(f"**Phase:** {trial_data['phase'] if pd.notna(trial_data['phase']) else 'Not Applicable'}")
                        st.write(f"**Minimum Age:** {trial_data['minimum_age']}")
                        st.write(f"**Canadian Sites:** {int(trial_data['num_canadian_sites'])}")
                        
                        # Display classification information from rare_df
                        classification = rare_df[rare_df['nct_id'] == selected_trial]['classification'].iloc[0]
                        with st.expander("Rare Disease Classification", expanded=True):
                            st.markdown(classification)
                        
                        # Brief summary
                        if pd.notna(trial_data['brief_summary']):
                            with st.expander("Brief Summary", expanded=True):
                                st.write(trial_data['brief_summary'])
                        
                        # Detailed description
                        if pd.notna(trial_data['detailed_description']):
                            with st.expander("Detailed Description"):
                                st.write(trial_data['detailed_description'])
                        
                        # Display conditions
                        conditions_dict = safe_get_session_state('rare_conditions_dict', {})
                        if conditions_dict and selected_trial in conditions_dict:
                            conditions = conditions_dict[selected_trial]
                            if conditions:
                                st.write("**Conditions:**")
                                st.write(", ".join(conditions))
                        
                        # Display keywords
                        keywords_dict = safe_get_session_state('rare_keywords_dict', {})
                        if keywords_dict and selected_trial in keywords_dict:
                            keywords = keywords_dict[selected_trial]
                            if keywords:
                                st.write("**Keywords:**")
                                st.write(", ".join(keywords))
                        
                        # Display interventions
                        interventions_df = safe_get_session_state('rare_interventions_df')
                        if interventions_df is not None:
                            trial_interventions = interventions_df[interventions_df['nct_id'] == selected_trial]
                            if not trial_interventions.empty:
                                st.write("**Interventions:**")
                                for _, row in trial_interventions.iterrows():
                                    st.write(f"- {row['intervention_type']}: {row['name']}")
                        
                        # Link to ClinicalTrials.gov
                        st.markdown(f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{selected_trial})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
    
