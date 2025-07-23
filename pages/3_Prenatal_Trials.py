import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
import plotly.graph_objects as go
import logging
import time

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
    get_download_link,
    prenatal_disease
)


def main():
    # Set page config
    st.set_page_config(
        page_title="Prenatal Trials in Canada",
        page_icon="🧬",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    st.title("Prenatal Trials in Canada")
    st.write("This page displays clinical trials that include the prenatal population.")
    
    # Connect to the database
    conn = connect_to_database()
    if not conn:
        st.error("Failed to connect to the database. Check your credentials and try again.")
        return
    

    with conn.connect():
        # Check if existing Adult Clinical Trials data exists
        adult_trials = safe_get_session_state('adult_trials')
        if adult_trials is None:
            st.info("Please run the Adult Trials page first to load trial data.")
            return
        
        prenatal_df = adult_trials.copy()
        st.info(f"Found {len(prenatal_df)} trials from all clinical trials")
        prenatal_df = prenatal_df[prenatal_df['is_prenatal'] == True].copy()
        # TODO: Update the text to show both original DF and then filtered DF counts
        st.info(f"Found {len(prenatal_df)} pre-natal trials.")
        
        # Load trial data for these NCT IDs
        prenatal_nct_ids = prenatal_df['nct_id'].tolist()
        
        prenatal_trials_data = prenatal_df.copy()
        st.info(f"Found {len(prenatal_trials_data)} trials for prenatal populations.")
        set_session_state('prenatal_trials_data', prenatal_trials_data)
        
        # Load filters
        if safe_get_session_state('prenatal_trials_filters') is None:
            # Fetch additional data needed for filtering
            with st.spinner("Loading additional data for filtering..."):
                import time
                # Fetch conditions for trials
                logging.info("Starting DB read: fetch_conditions_for_trials")
                t0 = time.time()
                conditions_df = fetch_conditions_for_trials(conn, prenatal_nct_ids)
                t1 = time.time()
                logging.info(f"Finished DB read: fetch_conditions_for_trials in {t1-t0:.2f} seconds")
                if not conditions_df.empty:
                    conditions_dict = build_conditions_dict(conditions_df)
                else:
                    conditions_dict = {}

                # Fetch keywords for trials
                logging.info("Starting DB read: fetch_keywords_for_trials")
                t0 = time.time()
                keywords_df = fetch_keywords_for_trials(conn, prenatal_nct_ids)
                t1 = time.time()
                logging.info(f"Finished DB read: fetch_keywords_for_trials in {t1-t0:.2f} seconds")
                if not keywords_df.empty:
                    keywords_dict = build_keywords_dict(keywords_df)
                else:
                    keywords_dict = {}

                # Fetch interventions for trials
                logging.info("Starting DB read: fetch_interventions_for_trials")
                t0 = time.time()
                interventions_df = fetch_interventions_for_trials(conn, prenatal_nct_ids)
                t1 = time.time()
                logging.info(f"Finished DB read: fetch_interventions_for_trials in {t1-t0:.2f} seconds")

                set_session_state('prenatal_conditions_dict', conditions_dict)
                set_session_state('prenatal_keywords_dict', keywords_dict)
                set_session_state('prenatal_interventions_df', interventions_df)
                set_session_state('prenatal_trials_filters', True)
        
        # Create sidebar filters
        st.sidebar.header("Filters")
        
        # Get dataset for filtering
        prenatal_trials_data = safe_get_session_state('prenatal_trials_data')
        
        if prenatal_trials_data is not None and not prenatal_trials_data.empty:
            # Status filter
            status_options = prenatal_trials_data['overall_status'].dropna().unique().tolist()
            status_filter = st.sidebar.multiselect(
                "Trial Status:",
                options=status_options,
                default=[]
            )

            # Date range filter
            date_filter = st.sidebar.date_input(
            "Study Start Date Rage",
            [date(2000,1,1), date(2025,12,31)]
            )

            # Gender filter
            gender_filter = st.sidebar.selectbox(
                "Gender Eligibility:",
                options=["ALL", "MALE", "FEMALE"],
                index=0,
            )

            # Age filter
            age_filter = st.sidebar.selectbox(
                "Age Range:",
                options=["ALL", "PEDIATRIC", "ADULT"],
                index=0,
            )

            # Keyword and condition search
            keyword_filter = st.sidebar.text_input("Keyword Search:", "")
            keyword_filter = keyword_filter.strip() if keyword_filter else None

            condition_filter = st.sidebar.text_input("Condition Search:", "")
            condition_filter = condition_filter.strip() if condition_filter else None
            
            # Apply filters
            filtered_df = filter_trials_by_criteria(
                prenatal_trials_data,
                status_filter=status_filter,
                date_filter=date_filter,
                gender_filter=gender_filter,
                age_filter=age_filter,
                keyword_filter=keyword_filter,
                condition_filter=condition_filter,
                conditions_dict=safe_get_session_state('prenatal_conditions_dict', {}),
                keywords_dict=safe_get_session_state('prenatal_keywords_dict', {})
            )
            st.info(f"Found {len(filtered_df)} trials post-filtering.")
            
            # Store filtered data in session state
            set_session_state('filtered_prenatal_trials', filtered_df)
            
            # Display results
            if filtered_df.empty:
                st.warning("No prenatal trials found matching your criteria.")
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
                
                tabs = [tab1, tab2, tab3, tab4, tab5]
                for i, tab in enumerate(tabs):
                    with tab:
                        if i == 0:
                            _ShowTrials(filtered_df)
                        elif i == 1:
                            _ShowVisualizations(filtered_df)
                        elif i == 2:
                            _ShowGeographicDistribution(filtered_df, conn)
                        elif i == 3:
                            _ShowInterventions(filtered_df, conn)
                        elif i == 4:
                            _ShowTrialDetails(filtered_df, prenatal_df)


def _ShowTrials(filtered_df):
    import time
    st.subheader("Prenatal Clinical Trials")
    # Display trials in a dataframe
    # Instrumentation for dataframe and download link
    display_cols = ['nct_id', 'brief_title', 'overall_status', 'minimum_age', 'start_date', 'num_canadian_sites']
    t0 = time.time()
    st.dataframe(filtered_df[display_cols], use_container_width=True)
    t1 = time.time()
    logging.info(f"st.dataframe in _ShowTrials took {t1-t0:.2f} seconds")
    t0 = time.time()
    st.markdown(get_download_link(filtered_df, filename="prenatal_disease_trials.csv"), unsafe_allow_html=True)
    t1 = time.time()
    logging.info(f"get_download_link in _ShowTrials took {t1-t0:.2f} seconds")


def _ShowGeographicDistribution(filtered_df, conn):

    st.subheader("Geographic Distribution")
    # Instrumentation for geographic visualizations
    t0 = time.time()
    render_city_visualization(filtered_df, conn)
    t1 = time.time()
    logging.info(f"render_city_visualization in _ShowGeographicDistribution took {t1-t0:.2f} seconds")
    t0 = time.time()
    render_province_visualization(filtered_df, conn, title_prefix="Rare Disease")
    t1 = time.time()
    logging.info(f"render_province_visualization in _ShowGeographicDistribution took {t1-t0:.2f} seconds")


def _ShowInterventions(filtered_df, conn):
                
    st.subheader("Interventions & Conditions")
    # Use the centralized visualization function for interventions and conditions analysis
    filtered_nct_ids = filtered_df['nct_id'].tolist()
    conditions_dict = safe_get_session_state('prenatal_conditions_dict', {})
    interventions_df = safe_get_session_state('prenatal_interventions_df')
    # Filter to only include the trials from filtered_df
    filtered_conditions_dict = {
        nct_id: conditions 
        for nct_id, conditions in conditions_dict.items() 
        if nct_id in filtered_nct_ids
    }
    filtered_interventions_df = None
    if interventions_df is not None and not interventions_df.empty:
        filtered_interventions_df = interventions_df[interventions_df['nct_id'].isin(filtered_nct_ids)]
    import time
    t0 = time.time()
    render_interventions_conditions_analysis(
        filtered_df,
        conn,
        title_prefix="Prenatal Trials",
        conditions_dict=filtered_conditions_dict,
        interventions_df=filtered_interventions_df
    )
    t1 = time.time()
    logging.info(f"render_interventions_conditions_analysis in _ShowInterventions took {t1-t0:.2f} seconds")


def _ShowVisualizations(filtered_df):

    st.subheader("Data Visualizations")
    # Visualization 1: Trials by Status
    # Instrumentation for status chart
    t0 = time.time()
    st.write("### Trials by Status")
    fig_status = plot_status_distribution(filtered_df)
    st.plotly_chart(fig_status, use_container_width=True)
    t1 = time.time()
    logging.info(f"plot_status_distribution in _ShowVisualizations took {t1-t0:.2f} seconds")
    
    # Visualization 2: Trials by Year
    if 'start_year' in filtered_df.columns and not filtered_df['start_year'].isna().all():
        t0 = time.time()
        st.write("### Trials by Year")
        fig_year, fig_area = plot_yearly_trends(filtered_df)
        if fig_year and fig_area:
            st.plotly_chart(fig_year, use_container_width=True)
            st.plotly_chart(fig_area, use_container_width=True)
        t1 = time.time()
        logging.info(f"plot_yearly_trends in _ShowVisualizations took {t1-t0:.2f} seconds")


def _ShowTrialDetails(filtered_df, prenatal_df):
  
    st.subheader("Trial Details")
    # Instrumentation for selectbox and details rendering
    t0 = time.time()
    selected_trial = st.selectbox(
        "Select a trial to view details:",
        options=filtered_df['nct_id'].tolist(),
        format_func=lambda nct_id: f"{nct_id} - {filtered_df[filtered_df['nct_id'] == nct_id]['brief_title'].iloc[0][:50]}..."
    )
    t1 = time.time()
    logging.info(f"st.selectbox in _ShowTrialDetails took {t1-t0:.2f} seconds")
    if selected_trial:
        t0 = time.time()
        st.write("### Trial Details")
        trial_data = filtered_df[filtered_df['nct_id'] == selected_trial].iloc[0]
        st.write(f"**NCT ID:** {selected_trial}")
        st.write(f"**Title:** {trial_data['brief_title']}")
        if pd.notna(trial_data['official_title']):
            st.write(f"**Official Title:** {trial_data['official_title']}")
        st.write(f"**Status:** {trial_data['overall_status']}")
        st.write(f"**Minimum Age:** {trial_data['minimum_age']}")
        st.write(f"**Canadian Sites:** {int(trial_data['num_canadian_sites'])}")
        
        # TODO: Add support for the classification when we add ut
        # Display classification information from prenatal_df
        # logging.info('fitered_df rows %s', filtered_df.columns)
        # logging.info('prenatal_df rows %s', prenatal_df.columns)
        # classification = filtered_df[prenatal_df['nct_id'] == selected_trial]['classification'].iloc[0]
        # with st.expander("Rare Disease Classification", expanded=True):
        #    st.markdown(classification)
        
        # Brief summary
        if pd.notna(trial_data['brief_summary']):
            with st.expander("Brief Summary", expanded=True):
                st.write(trial_data['brief_summary'])
        
        # Detailed description
        if pd.notna(trial_data['detailed_description']):
            with st.expander("Detailed Description"):
                st.write(trial_data['detailed_description'])
        
        # Display conditions
        conditions_dict = safe_get_session_state('prenatal_conditions_dict', {})
        if conditions_dict and selected_trial in conditions_dict:
            conditions = conditions_dict[selected_trial]
            if conditions:
                st.write("**Conditions:**")
                st.write(", ".join(conditions))
        
        # Criteria
        if pd.notna(trial_data['criteria']):
            with st.expander("Criteria", expanded=True):
                st.write(trial_data['criteria'])

        # Display keywords
        keywords_dict = safe_get_session_state('prenatal_keywords_dict', {})
        if keywords_dict and selected_trial in keywords_dict:
            keywords = keywords_dict[selected_trial]
            if keywords:
                st.write("**Keywords:**")
                st.write(", ".join(keywords))
        
        # Display interventions
        interventions_df = safe_get_session_state('prenatal_interventions_df')
        if interventions_df is not None:
            trial_interventions = interventions_df[interventions_df['nct_id'] == selected_trial]
            if not trial_interventions.empty:
                st.write("**Interventions:**")
                for _, row in trial_interventions.iterrows():
                    st.write(f"- {row['intervention_type']}: {row['name']}")
        
        # Link to ClinicalTrials.gov
        st.markdown(f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{selected_trial})")
        t1 = time.time()
        logging.info(f"_ShowTrialDetails inner block took {t1-t0:.2f} seconds")
   

if __name__ == "__main__":
    main()

