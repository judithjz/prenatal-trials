import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text

# Import utility modules
from utils.database_utils import (
    connect_to_database,
    handle_database_query
)

from utils.session_state_utils import (
    initialize_session_state,
    safe_get_session_state,
    set_session_state
)

from utils.api_utils import (
    get_download_link
)


def fetch_trials_by_country(engine, limit=20, min_trials=100):
    """
    Fetch trial counts by country from the AACT database.
    
    Args:
        engine: SQLAlchemy engine
        limit: Maximum number of top countries to return
        min_trials: Minimum number of trials a country must have to be included
        
    Returns:
        DataFrame with country trial counts
    """
    query = text("""
    SELECT
        f.country,
        COUNT(DISTINCT s.nct_id) AS trial_count
    FROM
        ctgov.studies s
    JOIN
        ctgov.facilities f ON s.nct_id = f.nct_id
    WHERE
        f.country IS NOT NULL AND f.country != ''
    GROUP BY
        f.country
    HAVING
        COUNT(DISTINCT s.nct_id) >= :min_trials
    ORDER BY
        trial_count DESC
    LIMIT :limit
    """)
    
    try:
        df = pd.read_sql_query(query, engine, params={"limit": limit, "min_trials": min_trials})
        return df
    except Exception as e:
        st.error(f"Error fetching trial counts by country: {e}")
        return pd.DataFrame()


def fetch_trials_by_country_and_year(engine, countries=None, start_year=2000, end_year=2025):
    """
    Fetch trial counts by country and year.
    
    Args:
        engine: SQLAlchemy engine
        countries: List of countries to include (if None, fetches top countries)
        start_year: Start year for the date range
        end_year: End year for the date range
        
    Returns:
        DataFrame with country and year trial counts
    """
    if countries is None or len(countries) == 0:
        # If no countries specified, get the top countries first
        top_countries_df = fetch_trials_by_country(engine)
        if top_countries_df.empty:
            return pd.DataFrame()
        countries = top_countries_df['country'].tolist()
    
    countries_tuple = tuple(countries)
    
    query = text("""
    SELECT
        f.country,
        EXTRACT(YEAR FROM s.start_date) AS year,
        COUNT(DISTINCT s.nct_id) AS trial_count
    FROM
        ctgov.studies s
    JOIN
        ctgov.facilities f ON s.nct_id = f.nct_id
    WHERE
        f.country IN :countries
        AND s.start_date IS NOT NULL
        AND EXTRACT(YEAR FROM s.start_date) BETWEEN :start_year AND :end_year
    GROUP BY
        f.country, EXTRACT(YEAR FROM s.start_date)
    ORDER BY
        f.country, year
    """)
    
    try:
        df = pd.read_sql_query(
            query, 
            engine, 
            params={
                "countries": countries_tuple, 
                "start_year": start_year, 
                "end_year": end_year
            }
        )
        return df
    except Exception as e:
        st.error(f"Error fetching trial counts by country and year: {e}")
        return pd.DataFrame()


def fetch_trials_by_country_and_phase(engine, countries=None, limit=20):
    """
    Fetch trial counts by country and phase.
    
    Args:
        engine: SQLAlchemy engine
        countries: List of countries to include (if None, fetches top countries)
        limit: Maximum number of top countries to return
        
    Returns:
        DataFrame with country and phase trial counts
    """
    if countries is None or len(countries) == 0:
        # If no countries specified, get the top countries first
        top_countries_df = fetch_trials_by_country(engine)
        if top_countries_df.empty:
            return pd.DataFrame()
        countries = top_countries_df['country'].tolist()
    
    countries_tuple = tuple(countries)
    
    query = text("""
    SELECT
        f.country,
        COALESCE(s.phase, 'Not Applicable') AS phase,
        COUNT(DISTINCT s.nct_id) AS trial_count
    FROM
        ctgov.studies s
    JOIN
        ctgov.facilities f ON s.nct_id = f.nct_id
    WHERE
        f.country IN :countries
    GROUP BY
        f.country, phase
    ORDER BY
        f.country, phase
    """)
    
    try:
        df = pd.read_sql_query(query, engine, params={"countries": countries_tuple})
        
        # Replace NULL/None values with 'Not Applicable'
        df['phase'] = df['phase'].fillna('Not Applicable')
        
        return df
    except Exception as e:
        st.error(f"Error fetching trial counts by country and phase: {e}")
        return pd.DataFrame()


def fetch_trials_by_country_and_status(engine, countries=None, limit=20):
    """
    Fetch trial counts by country and status.
    
    Args:
        engine: SQLAlchemy engine
        countries: List of countries to include (if None, fetches top countries)
        limit: Maximum number of top countries to return
        
    Returns:
        DataFrame with country and status trial counts
    """
    if countries is None or len(countries) == 0:
        # If no countries specified, get the top countries first
        top_countries_df = fetch_trials_by_country(engine)
        if top_countries_df.empty:
            return pd.DataFrame()
        countries = top_countries_df['country'].tolist()
    
    countries_tuple = tuple(countries)
    
    query = text("""
    SELECT
        f.country,
        s.overall_status,
        COUNT(DISTINCT s.nct_id) AS trial_count
    FROM
        ctgov.studies s
    JOIN
        ctgov.facilities f ON s.nct_id = f.nct_id
    WHERE
        f.country IN :countries
        AND s.overall_status IS NOT NULL
    GROUP BY
        f.country, s.overall_status
    ORDER BY
        f.country, s.overall_status
    """)
    
    try:
        df = pd.read_sql_query(query, engine, params={"countries": countries_tuple})
        return df
    except Exception as e:
        st.error(f"Error fetching trial counts by country and status: {e}")
        return pd.DataFrame()


def fetch_pediatric_trials_by_country(engine, limit=20, min_trials=50):
    """
    Fetch pediatric trial counts by country.
    
    Args:
        engine: SQLAlchemy engine
        limit: Maximum number of top countries to return
        min_trials: Minimum number of trials a country must have to be included
        
    Returns:
        DataFrame with country pediatric trial counts
    """
    query = text("""
    SELECT
        f.country,
        COUNT(DISTINCT s.nct_id) AS pediatric_trial_count
    FROM
        ctgov.studies s
    JOIN
        ctgov.facilities f ON s.nct_id = f.nct_id
    JOIN
        ctgov.eligibilities e ON s.nct_id = e.nct_id
    WHERE
        f.country IS NOT NULL AND f.country != ''
        AND (
            e.minimum_age LIKE '%day%' OR
            e.minimum_age LIKE '%week%' OR
            e.minimum_age LIKE '%month%' OR
            (e.minimum_age LIKE '%year%' AND CAST(SUBSTRING(e.minimum_age FROM '^[0-9]+') AS INTEGER) < 18)
        )
    GROUP BY
        f.country
    HAVING
        COUNT(DISTINCT s.nct_id) >= :min_trials
    ORDER BY
        pediatric_trial_count DESC
    LIMIT :limit
    """)
    
    try:
        df = pd.read_sql_query(query, engine, params={"limit": limit, "min_trials": min_trials})
        return df
    except Exception as e:
        st.error(f"Error fetching pediatric trial counts by country: {e}")
        return pd.DataFrame()


def fetch_country_comparison_data(engine, top_countries=None, include_pediatric=True):
    """
    Fetch comprehensive comparison data for countries.
    
    Args:
        engine: SQLAlchemy engine
        top_countries: List of countries to include (if None, fetches top countries)
        include_pediatric: Whether to include pediatric trial data
        
    Returns:
        DataFrame with combined country comparison data
    """
    if top_countries is None or len(top_countries) == 0:
        # If no countries specified, get the top countries first
        top_countries_df = fetch_trials_by_country(engine)
        if top_countries_df.empty:
            return pd.DataFrame()
        top_countries = top_countries_df['country'].tolist()
    
    countries_tuple = tuple(top_countries)
    
    # Query to get total trials and active trials by country
    query = text("""
    WITH CountryTrials AS (
        SELECT
            f.country,
            s.nct_id,
            s.overall_status
        FROM
            ctgov.studies s
        JOIN
            ctgov.facilities f ON s.nct_id = f.nct_id
        WHERE
            f.country IN :countries
    )
    
    SELECT
        country,
        COUNT(DISTINCT nct_id) AS total_trials,
        COUNT(DISTINCT CASE WHEN overall_status IN ('RECRUITING', 'ACTIVE, NOT RECRUITING', 'ENROLLING BY INVITATION') THEN nct_id END) AS active_trials,
        COUNT(DISTINCT CASE WHEN overall_status = 'RECRUITING' THEN nct_id END) AS recruiting_trials
    FROM
        CountryTrials
    GROUP BY
        country
    ORDER BY
        total_trials DESC
    """)
    
    try:
        df = pd.read_sql_query(query, engine, params={"countries": countries_tuple})
        
        # If including pediatric data, fetch and merge it
        if include_pediatric:
            # Query to get pediatric trials by country
            pediatric_query = text("""
            SELECT
                f.country,
                COUNT(DISTINCT s.nct_id) AS pediatric_trials
            FROM
                ctgov.studies s
            JOIN
                ctgov.facilities f ON s.nct_id = f.nct_id
            JOIN
                ctgov.eligibilities e ON s.nct_id = e.nct_id
            WHERE
                f.country IN :countries
                AND (
                    e.minimum_age LIKE '%day%' OR
                    e.minimum_age LIKE '%week%' OR
                    e.minimum_age LIKE '%month%' OR
                    (e.minimum_age LIKE '%year%' AND CAST(SUBSTRING(e.minimum_age FROM '^[0-9]+') AS INTEGER) < 18)
                )
            GROUP BY
                f.country
            ORDER BY
                f.country
            """)
            
            pediatric_df = pd.read_sql_query(pediatric_query, engine, params={"countries": countries_tuple})
            
            # Merge with main dataframe
            if not pediatric_df.empty:
                df = pd.merge(df, pediatric_df, on='country', how='left')
                df['pediatric_trials'] = df['pediatric_trials'].fillna(0).astype(int)
                df['pediatric_percentage'] = (df['pediatric_trials'] / df['total_trials'] * 100).round(1)
            
        return df
    except Exception as e:
        st.error(f"Error fetching country comparison data: {e}")
        return pd.DataFrame()


def plot_trials_by_country(df, title="Top Countries by Clinical Trial Count"):
    """
    Create a horizontal bar chart of trial counts by country.
    
    Args:
        df: DataFrame with country and trial_count columns
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty:
        return None
    
    # Sort by trial count (descending) for the plot
    df_sorted = df.sort_values('trial_count', ascending=True)
    
    fig = px.bar(
        df_sorted,
        y='country',
        x='trial_count',
        orientation='h',
        title=title,
        labels={'country': 'Country', 'trial_count': 'Number of Trials'},
        color='trial_count',
        color_continuous_scale='Viridis',
    )
    
    fig.update_layout(
        height=max(400, len(df) * 25),
        xaxis_title="Number of Trials",
        yaxis_title="Country",
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white"
    )
    
    return fig


def plot_trials_by_country_map(df, value_column='trial_count', title="Global Distribution of Clinical Trials"):
    """
    Create a choropleth map of trial counts by country.
    
    Args:
        df: DataFrame with country and trial_count columns
        value_column: Column name to use for the color scale
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty:
        return None
    
    # Create a copy to avoid modifying the original
    df_map = df.copy()
    
    # Create choropleth map
    fig = px.choropleth(
        df_map,
        locations="country",
        locationmode="country names",
        color=value_column,
        hover_name="country",
        color_continuous_scale="Viridis",
        title=title,
        labels={value_column: "Number of Trials"},
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig


def plot_trials_by_country_and_year(df, title="Clinical Trials by Country Over Time"):
    """
    Create a line chart of trial counts by country over time.
    
    Args:
        df: DataFrame with country, year, and trial_count columns
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty:
        return None
    
    fig = px.line(
        df,
        x='year',
        y='trial_count',
        color='country',
        title=title,
        labels={'year': 'Year', 'trial_count': 'Number of Trials', 'country': 'Country'},
        markers=True,
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="Year",
        yaxis_title="Number of Trials",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_trials_by_country_and_phase(df, title="Trial Phases by Country"):
    """
    Create a grouped bar chart of trial counts by country and phase.
    
    Args:
        df: DataFrame with country, phase, and trial_count columns
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty:
        return None
    
    # Define a custom order for phases
    phase_order = [
        "EARLY_PHASE1",
        "PHASE1",
        "PHASE1/2",
        "PHASE2",
        "PHASE2/3",
        "PHASE3",
        "PHASE4",
        "Not Applicable"
    ]
    
    # Convert phase to categorical with custom order
    phase_cat = pd.CategoricalDtype(categories=phase_order, ordered=True)
    df['phase'] = pd.Categorical(df['phase'], categories=phase_cat.categories, ordered=True)
    
    # Sort by country and phase
    df_sorted = df.sort_values(['country', 'phase'])
    
    fig = px.bar(
        df_sorted,
        x='country',
        y='trial_count',
        color='phase',
        title=title,
        labels={'country': 'Country', 'trial_count': 'Number of Trials', 'phase': 'Phase'},
        barmode='group',
        category_orders={"phase": phase_order},
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="Country",
        yaxis_title="Number of Trials",
        template="plotly_white",
        legend=dict(
            title="Phase",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig


def plot_pediatric_comparison(df, title="Pediatric vs. Adult Trials by Country"):
    """
    Create a grouped bar chart comparing pediatric vs. total trials by country.
    
    Args:
        df: DataFrame with country, total_trials, and pediatric_trials columns
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty or 'pediatric_trials' not in df.columns:
        return None
    
    # Calculate adult trials
    df['adult_trials'] = df['total_trials'] - df['pediatric_trials']
    
    # Create a long-format DataFrame for the grouped bar chart
    data = []
    for _, row in df.iterrows():
        data.append({'country': row['country'], 'trial_type': 'Pediatric', 'count': row['pediatric_trials']})
        data.append({'country': row['country'], 'trial_type': 'Adult', 'count': row['adult_trials']})
    
    long_df = pd.DataFrame(data)
    
    # Sort by total trials
    country_order = df.sort_values('total_trials', ascending=False)['country'].tolist()
    
    fig = px.bar(
        long_df,
        x='country',
        y='count',
        color='trial_type',
        title=title,
        labels={'country': 'Country', 'count': 'Number of Trials', 'trial_type': 'Trial Type'},
        barmode='group',
        color_discrete_map={'Pediatric': '#00CC96', 'Adult': '#636EFA'},
        category_orders={"country": country_order}
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="Country",
        yaxis_title="Number of Trials",
        template="plotly_white",
        legend=dict(
            title="Trial Type",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_pediatric_percentage(df, title="Percentage of Pediatric Trials by Country"):
    """
    Create a horizontal bar chart of pediatric trial percentages by country.
    
    Args:
        df: DataFrame with country and pediatric_percentage columns
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if df.empty or 'pediatric_percentage' not in df.columns:
        return None
    
    # Sort by pediatric percentage
    df_sorted = df.sort_values('pediatric_percentage', ascending=True)
    
    fig = px.bar(
        df_sorted,
        y='country',
        x='pediatric_percentage',
        orientation='h',
        title=title,
        labels={'country': 'Country', 'pediatric_percentage': 'Percentage of Total Trials (%)'},
        color='pediatric_percentage',
        color_continuous_scale='Viridis',
        text='pediatric_percentage'
    )
    
    fig.update_layout(
        height=max(400, len(df) * 25),
        xaxis_title="Percentage of Total Trials (%)",
        yaxis_title="Country",
        yaxis={'categoryorder': 'total ascending'},
        template="plotly_white"
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    
    return fig


def main():
    # Set page config
    st.set_page_config(
        page_title="Global Clinical Trials Comparison",
        page_icon="🌍",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Add a title and description
    st.title("Global Clinical Trials Comparison")
    st.write("This page allows you to compare clinical trial activity across different countries around the world.")
    
    # Connect to the database
    engine = connect_to_database()
    if not engine:
        st.error("Failed to connect to the database. Check your credentials and try again.")
        return
    
    # Create sidebar for filters and controls
    st.sidebar.header("Settings")
    
    # Number of countries to display
    num_countries = st.sidebar.slider(
        "Number of countries to display",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )
    
    # Minimum number of trials required for a country to be included
    min_trials = st.sidebar.slider(
        "Minimum trials per country",
        min_value=50,
        max_value=1000,
        value=100,
        step=50
    )
    
    # Year range for time-based visualizations
    year_range = st.sidebar.slider(
        "Year range",
        min_value=2000,
        max_value=2025,
        value=(2010, 2023),
        step=1
    )
    
    # Option to include pediatric trials comparison
    include_pediatric = st.sidebar.checkbox("Include pediatric trials analysis", value=True)
    
    # Option to reload data
    if st.sidebar.button("Reload Data"):
        # Clear session state entries related to this page
        for key in list(st.session_state.keys()):
            if key.startswith('global_trials_'):
                del st.session_state[key]
        st.rerun()
    
    # Fetch data or use cached version
    if safe_get_session_state('global_trials_countries') is None:
        with st.spinner("Loading global trial data..."):
            # Fetch the data
            country_data = fetch_trials_by_country(engine, limit=num_countries, min_trials=min_trials)
            
            if not country_data.empty:
                set_session_state('global_trials_countries', country_data)
                
                # Get list of top countries
                top_countries = country_data['country'].tolist()
                
                # Fetch comparison data
                comparison_data = fetch_country_comparison_data(engine, top_countries, include_pediatric)
                if not comparison_data.empty:
                    set_session_state('global_trials_comparison', comparison_data)
                
                # Fetch year trends
                year_data = fetch_trials_by_country_and_year(
                    engine, 
                    countries=top_countries, 
                    start_year=year_range[0], 
                    end_year=year_range[1]
                )
                if not year_data.empty:
                    set_session_state('global_trials_years', year_data)
                
                # Fetch phase data
                phase_data = fetch_trials_by_country_and_phase(engine, countries=top_countries)
                if not phase_data.empty:
                    set_session_state('global_trials_phases', phase_data)
    
    # Display the data
    country_data = safe_get_session_state('global_trials_countries')
    
    if country_data is not None and not country_data.empty:
        # Display summary metrics
        st.subheader("Global Trial Distribution")
        
        # Calculate total trials
        total_trials = country_data['trial_count'].sum()
        
        # Create metric columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Countries", len(country_data))
        with col2:
            st.metric("Total Trials", f"{total_trials:,}")
        with col3:
            avg_trials = int(total_trials / len(country_data))
            st.metric("Average Trials per Country", f"{avg_trials:,}")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Countries Overview", 
            "Geographic Distribution", 
            "Trends Over Time", 
            "Trial Characteristics"
        ])
        
        with tab1:
            st.subheader("Top Countries by Clinical Trial Count")
            
            # Create bar chart
            fig_countries = plot_trials_by_country(country_data)
            if fig_countries:
                st.plotly_chart(fig_countries, use_container_width=True)
            
            # Display the data table
            st.write("### Data Table")
            display_df = country_data.copy()
            display_df.columns = ['Country', 'Number of Trials']
            display_df = display_df.sort_values('Number of Trials', ascending=False).reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            st.markdown(get_download_link(display_df, filename="global_trials_by_country.csv"), unsafe_allow_html=True)
            
            # Show comparison data if available
            comparison_data = safe_get_session_state('global_trials_comparison')
            if comparison_data is not None and not comparison_data.empty:
                st.subheader("Detailed Country Comparison")
                
                # Format the display data
                display_comparison = comparison_data.copy()
                display_columns = ['country', 'total_trials', 'active_trials', 'recruiting_trials']
                column_names = ['Country', 'Total Trials', 'Active Trials', 'Recruiting Trials']
                
                if 'pediatric_trials' in display_comparison.columns:
                    display_columns.extend(['pediatric_trials', 'pediatric_percentage'])
                    column_names.extend(['Pediatric Trials', 'Pediatric (%)'])
                
                display_comparison = display_comparison[display_columns]
                display_comparison.columns = column_names
                
                # Sort by total trials
                display_comparison = display_comparison.sort_values('Total Trials', ascending=False).reset_index(drop=True)
                
                # Display the table
                st.dataframe(display_comparison, use_container_width=True)
                
                # Download button
                st.markdown(get_download_link(display_comparison, filename="global_trials_detailed_comparison.csv"), unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Geographic Distribution of Clinical Trials")
            
            # Create map visualization
            fig_map = plot_trials_by_country_map(country_data)
            if fig_map:
                st.plotly_chart(fig_map, use_container_width=True)
            
            # If pediatric data is available, show a pediatric percentage map
            comparison_data = safe_get_session_state('global_trials_comparison')
            if comparison_data is not None and not comparison_data.empty and 'pediatric_percentage' in comparison_data.columns:
                st.subheader("Percentage of Pediatric Trials by Country")
                
                # Create pediatric percentage map
                fig_ped_map = plot_trials_by_country_map(
                    comparison_data, 
                    value_column='pediatric_percentage',
                    title="Global Distribution of Pediatric Clinical Trials (% of Total)"
                )
                if fig_ped_map:
                    st.plotly_chart(fig_ped_map, use_container_width=True)
        
        with tab3:
            st.subheader("Trial Trends Over Time")
            
            # Get year data
            year_data = safe_get_session_state('global_trials_years')
            
            if year_data is not None and not year_data.empty:
                # Create line chart
                fig_years = plot_trials_by_country_and_year(year_data)
                if fig_years:
                    st.plotly_chart(fig_years, use_container_width=True)
                
                # Allow download of the data
                st.markdown(get_download_link(year_data, filename="global_trials_by_year.csv"), unsafe_allow_html=True)
            else:
                st.info("No time trend data available.")
        
        with tab4:
            st.subheader("Trial Characteristics by Country")
            
            # Create sub-tabs for different characteristics
            char_tab1, char_tab2, char_tab3 = st.tabs(["Phases", "Pediatric vs. Adult", "Status"])
            
            with char_tab1:
                # Get phase data
                phase_data = safe_get_session_state('global_trials_phases')
                
                if phase_data is not None and not phase_data.empty:
                    # Create phase chart
                    fig_phases = plot_trials_by_country_and_phase(phase_data)
                    if fig_phases:
                        st.plotly_chart(fig_phases, use_container_width=True)
                    
                    # Add explanatory text
                    st.info("""
                    The phase distribution shows the maturity of trials in each country:
                    - **Phase 1**: Initial safety testing
                    - **Phase 2**: Efficacy and side effects
                    - **Phase 3**: Efficacy in large populations
                    - **Phase 4**: Post-marketing surveillance
                    """)
                else:
                    st.info("No phase data available.")
            
            with char_tab2:
                # Get comparison data with pediatric information
                comparison_data = safe_get_session_state('global_trials_comparison')
                
                if comparison_data is not None and not comparison_data.empty and 'pediatric_trials' in comparison_data.columns:
                    # Create pediatric vs. adult comparison chart
                    fig_ped_comp = plot_pediatric_comparison(comparison_data)
                    if fig_ped_comp:
                        st.plotly_chart(fig_ped_comp, use_container_width=True)
                    
                    # Create pediatric percentage chart
                    fig_ped_pct = plot_pediatric_percentage(comparison_data)
                    if fig_ped_pct:
                        st.plotly_chart(fig_ped_pct, use_container_width=True)
                        
                    # Display some informative text
                    st.info("""
                    This analysis shows the proportion of pediatric clinical trials (participants under 18 years old) 
                    compared to adult trials across different countries. Countries with higher percentages may have 
                    more robust pediatric research programs or specific initiatives to address childhood diseases.
                    """)
                else:
                    st.info("No pediatric comparison data available.")
            
            with char_tab3:
                # Status data would need to be fetched
                # Since it's not in session state yet, we need to fetch it
                if safe_get_session_state('global_trials_status') is None:
                    with st.spinner("Loading status data..."):
                        top_countries = country_data['country'].tolist()
                        status_data = fetch_trials_by_country_and_status(engine, countries=top_countries)
                        if not status_data.empty:
                            set_session_state('global_trials_status', status_data)
                
                status_data = safe_get_session_state('global_trials_status')
                
                if status_data is not None and not status_data.empty:
                    # Group the statuses into broader categories for clearer visualization
                    status_mapping = {
                        'RECRUITING': 'Recruiting',
                        'ACTIVE, NOT RECRUITING': 'Active',
                        'ENROLLING BY INVITATION': 'Active',
                        'NOT YET RECRUITING': 'Planned',
                        'COMPLETED': 'Completed',
                        'TERMINATED': 'Terminated/Withdrawn',
                        'WITHDRAWN': 'Terminated/Withdrawn',
                        'SUSPENDED': 'Terminated/Withdrawn',
                        'UNKNOWN STATUS': 'Unknown'
                    }
                    
                    # Apply the mapping, with a default category for any unmapped statuses
                    status_data['status_category'] = status_data['overall_status'].map(
                        lambda x: status_mapping.get(x, 'Other')
                    )
                    
                    # Aggregate by country and status category
                    status_agg = status_data.groupby(['country', 'status_category'])['trial_count'].sum().reset_index()
                    
                    # Create a grouped bar chart
                    fig_status = px.bar(
                        status_agg,
                        x='country',
                        y='trial_count',
                        color='status_category',
                        title="Trial Status Distribution by Country",
                        labels={'country': 'Country', 'trial_count': 'Number of Trials', 'status_category': 'Status'},
                        barmode='group',
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    
                    fig_status.update_layout(
                        height=600,
                        xaxis_title="Country",
                        yaxis_title="Number of Trials",
                        template="plotly_white",
                        legend=dict(
                            title="Status",
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        xaxis={'categoryorder': 'total descending'}
                    )
                    
                    st.plotly_chart(fig_status, use_container_width=True)
                    
                    # Display some explanatory text
                    st.info("""
                    The status distribution shows the current state of clinical trials in each country:
                    - **Recruiting/Active**: Trials that are currently enrolling or treating participants
                    - **Completed**: Trials that have concluded their primary outcome measures
                    - **Terminated/Withdrawn**: Trials that were stopped early
                    - **Planned**: Trials that are approved but haven't started recruitment yet
                    """)
                else:
                    st.info("No status data available.")
    else:
        st.info("No data available. Please check your database connection and try again.")


if __name__ == "__main__":
    main()