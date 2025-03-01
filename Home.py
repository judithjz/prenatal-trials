import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px

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

def fetch_active_trials_data(conn):
    """
    Fetch data of active trials grouped by country and sorted by the top 15 countries.
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

def fetch_can_trials(conn):
    """
    Fetch data of all trials in Canada 
    """
    query = """
    SELECT 
    studies.overall_status,
    EXTRACT(YEAR FROM studies.start_date) AS start_year,
    COUNT(studies.nct_id) AS can_trials
    FROM studies
    JOIN countries ON studies.nct_id = countries.nct_id
    WHERE countries.name = 'Canada'
    AND EXTRACT(YEAR FROM studies.start_date) > 2010
    AND studies.overall_status NOT IN ('COMPLETED', 'TERMINATED', 'NO_LONGER_AVAILABLE', 
                                        'WITHDRAWN', 'APPROVED_FOR_MARKETING', 'SUSPENDED')
    GROUP BY start_year, studies.overall_status
    ORDER BY start_year, studies.overall_status DESC;
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
            st.info("Fetching data from the AACT database...")
            active_trials_data = fetch_active_trials_data(conn)
            
            if not active_trials_data.empty:
                
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

                can_trials_data = fetch_can_trials(conn)
                st.subheader("Active studies in Canada by Overall Status and Year")
                fig = px.bar(
                    can_trials_data,
                    x="start_year",
                    y="can_trials",
                    color="overall_status",
                    labels={"start_year": "Start Year", "can_trials": "Number of Trials", "overall_status": "Status"},
                    title="Number of Clinical Trials by Year and Status in Canada",
                )
                fig.update_layout(
                    xaxis_title="Start Year",
                    yaxis_title="Number of Trials",
                    barmode="group",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available to display.")
    else:
        st.error("Failed to connect to the database. Check your credentials and try again.")

if __name__ == "__main__":
    main()