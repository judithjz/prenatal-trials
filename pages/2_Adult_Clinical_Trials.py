import streamlit as st
from datetime import date
import pandas as pd
import logging

# Import from utility modules
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
    plot_trial_age_distribution,
    render_city_visualization,
    render_city_visualization_normalized,
    render_summary_metrics,
    render_trial_details,
    render_province_visualization,
    render_interventions_conditions_analysis
)

from utils.api_utils import (
    get_download_link
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s — %(levelname)s — %(message)s",  # Define the log message format
    datefmt="%Y-%m-%d %H:%M:%S",  # Define the date format
    handlers=[
        logging.FileHandler("app.log"),  # Logs to a file named 'app.log'
        logging.StreamHandler()          # Logs to the console
    ]
)

def fetch_adult_trials_in_canada(engine):
    """
    Fetch data for clinical trials that:
    1. Have at least one site in Canada (from facilities table)
    2. Include participants with minimum age >= 18 years
    3. Include participants with a maximum age <= 65 years
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        DataFrame containing adult trial data
    """
    query = """
    SELECT DISTINCT
        s.nct_id,
        s.brief_title,
        s.official_title,
        s.overall_status,
        e.minimum_age,
        e.maximum_age,
        s.study_type,
        s.start_date,
        e.gender,
        bs.description as brief_summary,
        dd.description as detailed_description,
        COUNT(DISTINCT f.id) AS num_canadian_sites,
        e.criteria,
        position('pregnan' IN LOWER(e.criteria)) > 0 as is_prenatal
    FROM ctgov.studies s
    JOIN ctgov.facilities f ON s.nct_id = f.nct_id
    JOIN ctgov.eligibilities e ON s.nct_id = e.nct_id
    LEFT JOIN ctgov.brief_summaries bs ON s.nct_id = bs.nct_id
    LEFT JOIN ctgov.detailed_descriptions dd ON s.nct_id = dd.nct_id
    WHERE f.country = 'Canada'
    AND s.study_type = 'INTERVENTIONAL'
    GROUP BY s.nct_id, s.brief_title, s.official_title, s.overall_status, 
             e.minimum_age, e.maximum_age, s.study_type, s.start_date,
             bs.description, dd.description, e.gender, is_prenatal,
             e.criteria
    ORDER BY s.start_date DESC;
    """
    
    try:
        # Use SQLAlchemy's text() to properly prepare the query
        from sqlalchemy import text
        df = pd.read_sql_query(text(query), engine)

        logging.info("original read: %d rows", df.shape[0])
        
        # Parse minimum age to months for filtering
        from utils.database_utils import parse_age_to_months
        df['min_age_in_months'] = df['minimum_age'].apply(parse_age_to_months)
        df['max_age_in_months'] = df['maximum_age'].apply(parse_age_to_months)

        # Filter out any studies that are have a min age of 65 years or where
        # min age is undefined
        adult_df = df[(df['min_age_in_months'].isna()) | (df['min_age_in_months'] <= 780)].copy()
        
        # Add some useful columns for analysis
        if not adult_df.empty and 'start_date' in adult_df.columns:
            adult_df['start_year'] = pd.to_datetime(adult_df['start_date'], errors='coerce').dt.year
       
        logging.info("Returnign DF: %s", adult_df)
        logging.info('filtered df: %d rows', adult_df.shape[0])

        return adult_df
    except Exception as e:
        st.error(f"Error fetching adult trials data: {e}")
        return pd.DataFrame()


def main():
    # Set page config
    st.set_page_config(
        page_title="Adult Clinical Trials in Canada - For Prenatal Search",
        page_icon="🧬",
        layout="wide"
    )
    
    # Initialize session state variables
    initialize_session_state()
        
    st.title("Adult Clinical Trials in Canada - For Prenatal Search")
    st.write("This page displays clinical trials with sites in Canada that include participants 18 years or older.\n\nUPDATE ME!")
    
    # Connect to the database
    engine = connect_to_database()
    if not engine:
        st.error("Failed to connect to the database. Check your credentials and try again.")
        return
    
    # Load the adult trials data if it doesn't exist in session state or reload is requested
    if safe_get_session_state('adult_trials') is None or safe_get_session_state('need_data_reload', True):
        with st.spinner("Loading adult trials from the AACT database..."):
            # Fetch adult trials
            adult_trials = fetch_adult_trials_in_canada(engine)
            logging.info('fetched adult trials')
            
            # Store in session state
            set_session_state('adult_trials', adult_trials)
            logging.info('storing in session state. adult trials size %d', adult_trials.shape[0])
            
            if not adult_trials.empty:
                # Fetch conditions for trials
                nct_ids = adult_trials['nct_id'].tolist()
                conditions_df = fetch_conditions_for_trials(engine, nct_ids)
                
                # Convert to dict for easy access
                conditions_dict = build_conditions_dict(conditions_df)
                set_session_state('adult_conditions_dict', conditions_dict)
                
                # Fetch keywords for trials
                keywords_df = fetch_keywords_for_trials(engine, nct_ids)
                keywords_dict = build_keywords_dict(keywords_df)
                set_session_state('adult_keywords_dict', keywords_dict)
                
                # Fetch facilities data that includes city information
                facilities_df = fetch_facilities_for_trials(engine, nct_ids=nct_ids)
                set_session_state('adult_facilities_df', facilities_df)
                
                # Prepare city-level aggregation data
                if not facilities_df.empty:
                    city_data = prepare_city_data(facilities_df)
                    set_session_state('adult_city_data', city_data)
                else:
                    set_session_state('adult_city_data', pd.DataFrame())
            
            set_session_state('need_data_reload', False)
   
    # Add a refresh button to force reload data
    if st.sidebar.button("Reload Data"):
        set_session_state('need_data_reload', True)
        st.rerun()
    
    # Query basic data to populate filters
    basic_query = """
    SELECT DISTINCT
        s.overall_status, 
        e.gender,
        EXTRACT(YEAR FROM s.start_date) as start_year
    FROM ctgov.studies s
    JOIN ctgov.eligibilities e ON s.nct_id = e.nct_id
    WHERE s.overall_status IS NOT NULL
    AND e.gender IS NOT NULL
    ORDER BY s.overall_status;

    """
    
    try:
        # Use SQLAlchemy's text() to properly prepare the query
        from sqlalchemy import text
        filter_options = pd.read_sql_query(text(basic_query), engine)
        gender_options = filter_options['gender'].dropna().unique().tolist()
        
        # Year options manually filtered to avoid too many options
        year_options = sorted(filter_options['start_year'].dropna().astype(int).unique().tolist())
        year_options = [year for year in year_options if year >= 2000]
    except Exception as e:
        st.sidebar.error(f"Error loading filter options: {e}")
        year_options = list(range(2010, 2025))
        gender_options = ["All", "Female", "Male"]        
    
    # Create sidebar filters
    st.sidebar.header("Filters")
    
  
    date_filter = st.sidebar.date_input(
    "Study Start Date Rage",
    [date(2000,1,1), date(2025,12,31)]
    )
    
    gender_filter = st.sidebar.selectbox(
        "Gender Eligibility:",
        options=["ALL", "MALE", "FEMALE"],  
        index=0,
    )

    age_filter = st.sidebar.selectbox(
      "Age Range:",
      options=["ALL", "PEDIATRIC", "ADULT"],
      index=0,
    )
    
    keyword_filter = st.sidebar.text_input("Keyword Search:", "")
    keyword_filter = keyword_filter.strip() if keyword_filter else None
    
    condition_filter = st.sidebar.text_input("Condition Search:", "")
    condition_filter = condition_filter.strip() if condition_filter else None
    
    # Apply filters to the dataframe
    adult_df = safe_get_session_state('adult_trials')
    if adult_df is not None and not adult_df.empty:
        # Use the filtering utility to apply all filters at once
        filtered_df = filter_trials_by_criteria(
            adult_df,
            date_filter=date_filter,
            gender_filter=gender_filter,
            age_filter=age_filter,
            keyword_filter=keyword_filter,
            condition_filter=condition_filter,
            conditions_dict=safe_get_session_state('adult_conditions_dict', {}),
            keywords_dict=safe_get_session_state('adult_keywords_dict', {}),
            )
        
        # Display results
        if filtered_df.empty:
            st.warning("No adult trials found matching your criteria.")
        else:
            # Display summary statistics
            st.subheader("Summary Statistics")
            render_summary_metrics(filtered_df)
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Trial List", 
                "Charts", 
                "Geographic Distribution", 
                "Interventions and Conditions", 
                "Trial Details"
            ])

            with tab1:
                st.subheader("Adult Clinical Trials")
                # Display the trials in a dataframe
                display_cols = ['nct_id', 'brief_title', 'minimum_age', 'maximum_age', 'start_date', 'num_canadian_sites', 'gender']
                st.dataframe(filtered_df[display_cols], use_container_width=True)
                
                # Download button for the results
                st.markdown(get_download_link(filtered_df[display_cols]), unsafe_allow_html=True)

            with tab2:
                st.subheader("Data Visualizations")
               
                
                # Visualization 1: Trials by Year (if available)
                if 'start_year' in filtered_df.columns and not filtered_df['start_year'].isna().all():
                    fig_year, fig_area = plot_yearly_trends(filtered_df)
                    if fig_year and fig_area:
                        st.plotly_chart(fig_year, use_container_width=True)
                        st.plotly_chart(fig_area, use_container_width=True)
                
                # Visualization 2: Age Distribution
                st.write("### Age Distribution")
                fig_age = plot_trial_age_distribution(filtered_df)
                st.plotly_chart(fig_age, use_container_width=True)

            
            
            with tab3:
                # Pass filtered_df and conn to ensure that the visualization 
                # only uses facilities for the filtered trials
                render_city_visualization(filtered_df, engine)

                # Use the population-normalized visualization
                render_city_visualization_normalized(filtered_df, engine)

                # Add province distribution visualization
                render_province_visualization(filtered_df, engine, title_prefix="Adult")
                
            with tab4:
                # Add interventions and conditions analysis
                render_interventions_conditions_analysis(
                    filtered_df, 
                    engine, 
                    title_prefix="Adult",
                    conditions_dict=safe_get_session_state('adult_conditions_dict', {}),
                    interventions_df=fetch_interventions_for_trials(engine, filtered_df['nct_id'].tolist())
                )
                     
            with tab5:
                st.subheader("Trial Details")
                
                # Use the utility function to render trial details
                render_trial_details(
                    filtered_df, 
                    engine, 
                    safe_get_session_state('adult_keywords_dict', {}), 
                    safe_get_session_state('adult_conditions_dict', {})
                )
    else:
        st.info("Loading adult trials data...")


if __name__ == "__main__":
    main()




