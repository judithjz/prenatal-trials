import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px

# Database connection parameters
DB_PARAMS = {
    "host": "aact-db.ctti-clinicaltrials.org",
    "port": 5432,
    "dbname": "aact",
    "user": "lricher",
    "password": "egr.pqy!TYK0qmv2dpt"
}

def connect_to_database():
    """Establish connection to the AACT database."""
    try:
        conn = psycopg2.connect(
            host=DB_PARAMS['host'],
            port=DB_PARAMS['port'],
            dbname=DB_PARAMS['dbname'],
            user=DB_PARAMS['user'],
            password=DB_PARAMS['password']
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None

def fetch_active_trials_data(conn):
    """
    Fetch data of active trials grouped by country and sorted by the top 10 countries.
    """
    query = """
    SELECT 
        countries.name AS country,
        COUNT(studies.nct_id) AS active_trials
    FROM studies
    JOIN countries ON studies.nct_id = countries.nct_id
    WHERE studies.overall_status = 'RECRUITING' -- Uncomment if filtering active trials only
    GROUP BY country
    ORDER BY active_trials DESC
    LIMIT 15;
    """
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Streamlit app
def main():
    st.title("Active Clinical Trials Per Country")
    st.write("This app displays the top 10 countries with the highest number of active clinical trials.")

    # Connect to the database
    conn = connect_to_database()
    if conn:
        with conn:
            # Fetch data
            st.info("Fetching data from the database...")
            active_trials_data = fetch_active_trials_data(conn)
            
            if not active_trials_data.empty:
                # Display data
                st.subheader("Top 10 Countries with Active Clinical Trials")
                st.dataframe(active_trials_data)

                # Plot data with Plotly
                st.subheader("Graph: Number of Active Trials Per Country")
                fig = px.bar(
                    active_trials_data,
                    x="active_trials",
                    y="country",
                    orientation="h",
                    labels={"active_trials": "Number of Active Trials", "country": "Country"},
                    title="Active Trials by Country (Top 10)",
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),  # Reverse y-axis for highest value at the top
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available to display.")
    else:
        st.error("Failed to connect to the database. Check your credentials and try again.")

if __name__ == "__main__":
    main()