import streamlit as st
import pandas as pd
import plotly.express as px

# Import from reorganized utility modules
from utils.database_utils import (
    connect_to_database,
    fetch_pediatric_trials_in_canada,
    fetch_conditions_for_trials,
    fetch_keywords_for_trials,
    fetch_interventions_for_trials,
    build_conditions_dict,
    build_keywords_dict
)

from utils.session_state_utils import (
    initialize_session_state,
    safe_get_session_state,
    set_session_state,
    navigate_to_page,
    reset_classifier_state
)

from utils.filtering_utils import (
    filter_trials_by_criteria
)

from utils.visualization_utils import (
    render_summary_metrics
)

from utils.api_utils import (
    get_download_link,
    classify_rare_disease,
    extract_classification_statement
)


def on_search_button_click():
    """Handler for search button click."""
    navigate_to_page("search")
    set_session_state("showing_classification", False)
    set_session_state("selected_trial_id", None)


def on_classify_button_click():
    """Handler for classify button click."""
    navigate_to_page("results")


def on_view_details_button_click(nct_id):
    """Handler for view details button click."""
    navigate_to_page("detail", nct_id)


def on_back_button_click():
    """Handler for back button click."""
    navigate_to_page("results")


def render_search_page():
    """Render the search page."""
    st.title("Adult Trials in Canada - Prenatal Classifier")
    
    # Connect to the database
    conn = connect_to_database()
    if not conn:
        st.error("Failed to connect to the database. Check your credentials and try again.")
        return
    
    with conn.connect():
        # Initialize session state if necessary
        initialize_session_state()
        
        # Load the pediatric trials data if it doesn't exist in session state or reload is requested
        if safe_get_session_state('pediatric_trials') is None or safe_get_session_state('need_data_reload', True):
            with st.spinner("Loading pediatric trials from the AACT database..."):
                # Fetch pediatric trials
                pediatric_trials = fetch_pediatric_trials_in_canada(conn)
                
                # Store in session state
                set_session_state('pediatric_trials', pediatric_trials)
                
                if not pediatric_trials.empty:
                    # Fetch conditions for trials
                    nct_ids = pediatric_trials['nct_id'].tolist()
                    conditions_df = fetch_conditions_for_trials(conn, nct_ids)
                    
                    # Convert to dict for easy access
                    conditions_dict = build_conditions_dict(conditions_df)
                    set_session_state('conditions_dict', conditions_dict)
                    
                    # Fetch keywords for trials
                    keywords_df = fetch_keywords_for_trials(conn, nct_ids)
                    keywords_dict = build_keywords_dict(keywords_df)
                    set_session_state('keywords_dict', keywords_dict)
                
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
        pediatric_df = safe_get_session_state('pediatric_trials')
        if pediatric_df is not None and not pediatric_df.empty:
            # Use the filtering utility to apply all filters at once
            filtered_df = filter_trials_by_criteria(
                pediatric_df,
                status_filter=status_filter,
                phase_filter=phase_filter,
                year_filter=year_filter,
                keyword_filter=keyword_filter,
                condition_filter=condition_filter,
                conditions_dict=safe_get_session_state('conditions_dict', {}),
                keywords_dict=safe_get_session_state('keywords_dict', {})
            )
            
            # Store the filtered dataframe
            set_session_state('filtered_trials', filtered_df)
            
            # Display results
            if filtered_df.empty:
                st.warning("No pediatric trials found matching your criteria.")
            else:
                # Display summary statistics
                st.subheader("Summary Statistics")
                render_summary_metrics(filtered_df)
                
                # Display the trials in a dataframe
                st.subheader("Filtered Pediatric Clinical Trials")
                display_cols = ['nct_id', 'brief_title', 'overall_status', 'minimum_age', 'phase', 'start_date', 'num_canadian_sites']
                st.dataframe(filtered_df[display_cols], use_container_width=True)
                
                # Download button for the results
                st.markdown(get_download_link(filtered_df[display_cols]), unsafe_allow_html=True)
                
                # Rare disease classification section
                st.subheader("Rare Disease Classification")
                
                # Get API key from secrets
                anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
                if not anthropic_api_key and hasattr(st.secrets, "database") and hasattr(st.secrets.database, "ANTHROPIC_API_KEY"):
                    anthropic_api_key = st.secrets.database.ANTHROPIC_API_KEY
                
                # Select which trials to classify
                col1, col2 = st.columns(2)
                with col1:
                    classification_option = st.radio(
                        "Classification Option:",
                        ["Select Single Trial", "Classify All Filtered Trials"],
                        key="classification_option"
                    )
                
                trials_to_classify = []
                if classification_option == "Select Single Trial":
                    with col2:
                        selected_trial = st.selectbox(
                            "Select a trial:",
                            options=filtered_df['nct_id'].tolist(),
                            format_func=lambda nct_id: f"{nct_id} - {filtered_df[filtered_df['nct_id'] == nct_id]['brief_title'].iloc[0][:50]}...",
                            key="single_trial_select"
                        )
                    trials_to_classify = [selected_trial] if selected_trial else []
                else:
                    trials_to_classify = filtered_df['nct_id'].tolist()
                    limit = min(len(trials_to_classify), 750)  # API usage increased to allow for classification of pediatric studies.
                    trials_to_classify = trials_to_classify[:limit]
                    st.write(f"Will classify {limit} trials (limited to 750 for API usage)")
                
                # Check if API key exists
                if not anthropic_api_key:
                    st.warning("No Anthropic API key found in Streamlit secrets. Please add ANTHROPIC_API_KEY to your .streamlit/secrets.toml file.")
                
                # Only show classification button if trials are selected
                if trials_to_classify:
                    if st.button("Classify Selected Trials", key="classify_button"):
                        # Store the trials to classify and start the process
                        set_session_state('classification_started', True)
                        set_session_state('trials_to_classify', trials_to_classify)
                        set_session_state('showing_classification', False)  # Reset this flag
                        set_session_state('classification_results', None)  # Clear previous results
                        navigate_to_page("results")  # Move to results page
                        st.rerun()
                else:
                    st.info("Please select at least one trial to classify.")
        else:
            st.info("No pediatric trials data available. Please check your database connection.")


def render_results_page():
    """Render the classification results page."""
    st.title("Classification Results")
    
    # Button to go back to search
    if st.button("← Back to Search", key="back_to_search", on_click=on_search_button_click):
        pass
    
    # Connect to the database
    conn = connect_to_database()
    if not conn:
        st.error("Failed to connect to the database. Check your credentials and try again.")
        return

    try:
        # Get API key
        anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if not anthropic_api_key and hasattr(st.secrets, "database") and hasattr(st.secrets.database, "ANTHROPIC_API_KEY"):
            anthropic_api_key = st.secrets.database.ANTHROPIC_API_KEY
        
        if not anthropic_api_key:
            st.error("No Anthropic API key found. Please add it to your secrets.toml file.")
            return
        
        # Debug information
        filtered_trials = safe_get_session_state('filtered_trials')
        if filtered_trials is None:
            st.error("No filtered trials found in session state.")
            st.info("Please go back to the search page and apply filters first.")
            if st.button("Go to Search", on_click=on_search_button_click):
                pass
            return
            
        # Check if classification has been done
        if not safe_get_session_state('showing_classification', False):
            if safe_get_session_state('classification_started', False):
                trials_to_classify = safe_get_session_state('trials_to_classify')
                if not trials_to_classify:
                    st.error("No trials to classify found in session state.")
                    return
                    
                filtered_df = safe_get_session_state('filtered_trials')
                
                st.info(f"Starting classification for {len(trials_to_classify)} trials...")
                
                classification_results = []
                
                # Setup progress bar
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                for i, nct_id in enumerate(trials_to_classify):
                    # Update progress
                    progress_percent = int((i / len(trials_to_classify)) * 100)
                    progress_bar.progress(progress_percent)
                    progress_text.text(f"Classifying trial {i+1}/{len(trials_to_classify)}: {nct_id}")
                    
                    try:
                        # Get trial data
                        if nct_id not in filtered_df['nct_id'].values:
                            st.warning(f"Trial ID {nct_id} not found in filtered trials dataset.")
                            continue
                            
                        trial_data = filtered_df[filtered_df['nct_id'] == nct_id].iloc[0].to_dict()
                        
                        # Add conditions from our separate lookup
                        conditions_dict = safe_get_session_state('conditions_dict', {})
                        if nct_id in conditions_dict:
                            trial_data['conditions'] = conditions_dict[nct_id]
                        else:
                            trial_data['conditions'] = []
                        
                        # Classify using the utility function
                        classification = classify_rare_disease(trial_data, anthropic_api_key)
                        
                        # Store result
                        if classification.get('success', False):
                            classification_results.append({
                                'nct_id': nct_id,
                                'brief_title': trial_data['brief_title'],
                                'classification': classification['classification_text'],
                                'is_rare': classification.get('is_rare', False)
                            })
                        else:
                            error_msg = classification.get('error', 'Unknown error')
                            st.error(f"Failed to classify {nct_id}: {error_msg}")
                            # Add it anyway with error information
                            classification_results.append({
                                'nct_id': nct_id,
                                'brief_title': trial_data['brief_title'],
                                'classification': f"Error: {error_msg}",
                                'is_rare': False
                            })
                    except Exception as e:
                        st.error(f"Error processing trial {nct_id}: {str(e)}")
                
                # Complete progress
                progress_bar.progress(100)
                progress_text.text("Classification complete!")
                
                # Store results in session state
                if classification_results:
                    set_session_state('classification_results', pd.DataFrame(classification_results))
                    set_session_state('showing_classification', True)
                    st.rerun()
                else:
                    st.warning("No successful classifications were performed.")
                    st.error("Please try again or check the API key configuration.")
                    # Add debug button to show session state
                    if st.button("Debug Session State"):
                        st.write({k: v for k, v in st.session_state.items() if k not in ['classification_results', 'filtered_trials']})
            else:
                st.warning("Classification has not been initiated. Please go back to search page.")
                if st.button("Initialize Classification"):
                    set_session_state('classification_started', True)
                    st.rerun()
        
        # Display classification results
        if safe_get_session_state('showing_classification', False) and safe_get_session_state('classification_results') is not None:
            results_df = safe_get_session_state('classification_results')
            
            # Display summary statistics
            rare_count = sum(results_df['is_rare'])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Trials Classified", len(results_df))
            with col2:
                st.metric("Identified as Rare Disease", rare_count)
            
            # Create a table with results
            summary_df = results_df[['nct_id', 'brief_title', 'is_rare']].copy()
            summary_df.columns = ['NCT ID', 'Trial Title', 'Is Rare Disease']
            
            # Extract classification statements using the utility function
            classification_statements = []
            for _, row in results_df.iterrows():
                classification_statements.append(extract_classification_statement(row['classification']))
            
            summary_df['Classification Statement'] = classification_statements
            
            # Display the summary table
            st.subheader("Classification Summary")
            st.dataframe(summary_df, use_container_width=True)
            
            # Download button for classification results
            st.markdown(get_download_link(results_df, 
                                        filename="rare_disease_classification.csv", 
                                        button_text="Download Classification Results"), 
                        unsafe_allow_html=True)
            
            # Show trial selection for detailed view
            st.subheader("View Detailed Classification")
            selected_trial_id = st.selectbox(
                "Select a trial to view detailed classification:",
                options=results_df['nct_id'].tolist(),
                format_func=lambda nct_id: f"{nct_id} - {'Rare' if results_df[results_df['nct_id'] == nct_id]['is_rare'].iloc[0] else 'Not Rare'}",
                key="detail_selection_box"
            )
            
            if selected_trial_id:
                if st.button(f"View details for {selected_trial_id}", key="view_details_button"):
                    on_view_details_button_click(selected_trial_id)
                    st.rerun()
    finally:
        conn.close()


def render_detail_page():
    """Render the detailed classification page for a specific trial."""
    selected_trial_id = safe_get_session_state('selected_trial_id')
    classification_results = safe_get_session_state('classification_results')
    
    if selected_trial_id is None or classification_results is None:
        st.error("No trial selected for detailed view.")
        return
    
    # Get the trial data
    trial_data = classification_results[classification_results['nct_id'] == selected_trial_id]
    
    if trial_data.empty:
        st.error(f"Trial {selected_trial_id} not found in classification results.")
        return
    
    trial_row = trial_data.iloc[0]
    
    # Button to go back to results
    if st.button("← Back to Results", key="back_to_results", on_click=on_back_button_click):
        pass
    
    # Display trial details
    st.title(f"Classification Details: {trial_row['brief_title']}")
    st.subheader(f"NCT ID: {selected_trial_id}")
    st.info(f"Classification: {'Rare Disease' if trial_row['is_rare'] else 'Not a Rare Disease'}")
    
    # Display the full classification text
    st.markdown(trial_row['classification'])
    
    # Link to ClinicalTrials.gov
    st.markdown(f"[View on ClinicalTrials.gov](https://clinicaltrials.gov/study/{selected_trial_id})")


def main():
    """Main function to run the Streamlit app."""
    try:
        # Set page config
        st.set_page_config(
            page_title="Pediatric Trials - Rare Disease Classifier",
            page_icon="🧬",
            layout="wide"
        )
        
        # Initialize session state if not already done
        initialize_session_state()
        
        # Add a reset button to clear all session state (useful for debugging)
        if st.sidebar.checkbox("Show Debug Tools", False):
            if st.sidebar.button("Reset All Session State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                initialize_session_state()
                st.sidebar.success("Session state has been reset")
                st.rerun()
        
        # Handle page routing based on session state
        current_page = safe_get_session_state('page', "search")
        
        if current_page == "search":
            render_search_page()
        elif current_page == "results":
            render_results_page()
        elif current_page == "detail":
            render_detail_page()
        else:
            st.error("Invalid page state")
            set_session_state('page', "search")
            st.rerun()
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.code(f"Error traceback:\n{st.exception}")
        
        # Provide a reset button
        if st.button("Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            set_session_state('page', "search")
            st.rerun()


if __name__ == "__main__":
    main()
    
