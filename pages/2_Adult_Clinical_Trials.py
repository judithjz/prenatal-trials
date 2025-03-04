import streamlit as st
import pandas as pd

# Import from utility modules
from utils.database_utils import (
    connect_to_database,
    fetch_conditions_for_trials,
    fetch_keywords_for_trials,
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
    plot_phase_distribution,
    plot_status_distribution,
    plot_yearly_trends,
    plot_trial_age_distribution,
    render_city_visualization,
    render_city_visualization_normalized,
    render_summary_metrics,
    render_trial_details
)

from utils.api_utils import (
    get_download_link
)


def fetch_adult_trials_in_canada(conn):
    """
    Fetch data for clinical trials that:
    1. Have at least one site in Canada (from facilities table)
    2. Include participants with minimum age >= 18 years
    
    Args:
        conn: Database connection object
        
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
    WHERE f.country = 'Canada'
    GROUP BY s.nct_id, s.brief_title, s.official_title, s.overall_status, 
             e.minimum_age, s.phase, s.study_type, s.start_date, 
             bs.description, dd.description
    ORDER BY s.start_date DESC;
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        
        # Parse minimum age to months for filtering
        from utils.database_utils import parse_age_to_months
        df['age_in_months'] = df['minimum_age'].apply(parse_age_to_months)
        
        # Filter for adult trials (age >= 18 years = 216 months)
        # Use >= instead of < as in pediatric query
        adult_df = df[df['age_in_months'] >= 216].copy()
        
        # Add some useful columns for analysis
        if not adult_df.empty and 'start_date' in adult_df.columns:
            adult_df['start_year'] = pd.to_datetime(adult_df['start_date'], errors='coerce').dt.year
        
        return adult_df
    except Exception as e:
        st.error(f"Error fetching adult trials data: {e}")
        return pd.DataFrame()


def main():
    # Set page config
    st.set_page_config(
        page_title="Adult Clinical Trials in Canada",
        page_icon="🧬",
        layout="wide"
    )
    
    # Initialize session state variables
    initialize_session_state()
        
    st.title("Adult Clinical Trials in Canada")
    st.write("This page displays clinical trials with sites in Canada that include participants 18 years or older.")
    
    # Connect to the database
    conn = connect_to_database()
    if not conn:
        st.error("Failed to connect to the database. Check your credentials and try again.")
        return
    
    with conn:
        # Load the adult trials data if it doesn't exist in session state or reload is requested
        if safe_get_session_state('adult_trials') is None or safe_get_session_state('need_data_reload', True):
            with st.spinner("Loading adult trials from the AACT database..."):
                # Fetch adult trials
                adult_trials = fetch_adult_trials_in_canada(conn)
                
                # Store in session state
                set_session_state('adult_trials', adult_trials)
                
                if not adult_trials.empty:
                    # Fetch conditions for trials
                    nct_ids = adult_trials['nct_id'].tolist()
                    conditions_df = fetch_conditions_for_trials(conn, nct_ids)
                    
                    # Convert to dict for easy access
                    conditions_dict = build_conditions_dict(conditions_df)
                    set_session_state('adult_conditions_dict', conditions_dict)
                    
                    # Fetch keywords for trials
                    keywords_df = fetch_keywords_for_trials(conn, nct_ids)
                    keywords_dict = build_keywords_dict(keywords_df)
                    set_session_state('adult_keywords_dict', keywords_dict)
                    
                    # Fetch facilities data that includes city information
                    facilities_df = fetch_facilities_for_trials(conn, nct_ids=nct_ids)
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
        SELECT 
            DISTINCT overall_status, 
            phase,
            EXTRACT(YEAR FROM start_date) as start_year
        FROM ctgov.studies
        WHERE overall_status IS NOT NULL
        ORDER BY overall_status;
        """
        
        try:
            filter_options = pd.read_sql_query(basic_query, conn)
            status_options = filter_options['overall_status'].dropna().unique().tolist()
            phase_options = filter_options['phase'].dropna().unique().tolist() + ['Not Applicable']
            
            # Year options manually filtered to avoid too many options
            year_options = sorted(filter_options['start_year'].dropna().astype(int).unique().tolist())
            year_options = [year for year in year_options if year >= 2000]
        except Exception as e:
            st.sidebar.error(f"Error loading filter options: {e}")
            status_options = ["RECRUITING", "ACTIVE, NOT RECRUITING", "COMPLETED"]
            phase_options = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Not Applicable"]
            year_options = list(range(2010, 2025))
        
        # Create sidebar filters
        st.sidebar.header("Filters")
        
        status_filter = st.sidebar.multiselect(
            "Trial Status:",
            options=status_options,
            default=["RECRUITING"]
        )
        
        phase_filter = st.sidebar.multiselect(
            "Trial Phase:",
            options=phase_options,
            default=[]
        )
        
        year_filter = st.sidebar.slider(
            "Start Year Range:",
            min_value=min(year_options) if year_options else 2000,
            max_value=max(year_options) if year_options else 2025,
            value=(min(year_options) if year_options else 2000, 
                   max(year_options) if year_options else 2025)
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
                status_filter=status_filter,
                phase_filter=phase_filter,
                year_filter=year_filter,
                keyword_filter=keyword_filter,
                condition_filter=condition_filter,
                conditions_dict=safe_get_session_state('adult_conditions_dict', {}),
                keywords_dict=safe_get_session_state('adult_keywords_dict', {})
            )
            
            # Display results
            if filtered_df.empty:
                st.warning("No adult trials found matching your criteria.")
            else:
                # Display summary statistics
                st.subheader("Summary Statistics")
                render_summary_metrics(filtered_df)
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trial List", "Charts", "Canada Map", "Population-Normalized Map", "Trial Details"])

                with tab1:
                    st.subheader("Adult Clinical Trials")
                    # Display the trials in a dataframe
                    display_cols = ['nct_id', 'brief_title', 'overall_status', 'minimum_age', 'phase', 'start_date', 'num_canadian_sites']
                    st.dataframe(filtered_df[display_cols], use_container_width=True)
                    
                    # Download button for the results
                    st.markdown(get_download_link(filtered_df[display_cols]), unsafe_allow_html=True)

                with tab2:
                    st.subheader("Data Visualizations")
                    
                    # Visualization 1: Trials by Phase
                    st.write("### Trials by Phase")
                    fig_phase = plot_phase_distribution(filtered_df)
                    st.plotly_chart(fig_phase, use_container_width=True)
                    
                    # Visualization 2: Trials by Status
                    st.write("### Trials by Status")
                    fig_status = plot_status_distribution(filtered_df)
                    st.plotly_chart(fig_status, use_container_width=True)
                    
                    # Visualization 3: Trials by Year (if available)
                    if 'start_year' in filtered_df.columns and not filtered_df['start_year'].isna().all():
                        st.write("### Trials by Year")
                        fig_year, fig_area = plot_yearly_trends(filtered_df)
                        if fig_year and fig_area:
                            st.plotly_chart(fig_year, use_container_width=True)
                            st.plotly_chart(fig_area, use_container_width=True)
                    
                    # Visualization 4: Age Distribution
                    st.write("### Age Distribution")
                    fig_age = plot_trial_age_distribution(filtered_df)
                    st.plotly_chart(fig_age, use_container_width=True)

                with tab3:
                    # Pass filtered_df and conn to ensure that the visualization 
                    # only uses facilities for the filtered trials
                    render_city_visualization(filtered_df, conn)

                with tab4:
                    # Use the population-normalized visualization
                    render_city_visualization_normalized(filtered_df, conn)
                    
                with tab5:
                    st.subheader("Trial Details")
                    
                    # Use the utility function to render trial details
                    render_trial_details(
                        filtered_df, 
                        conn, 
                        safe_get_session_state('adult_keywords_dict', {}), 
                        safe_get_session_state('adult_conditions_dict', {})
                    )
        else:
            st.info("Loading adult trials data...")


if __name__ == "__main__":
    main()
