import streamlit as st

def main():
    """
    Main function to render the landing page.
    """
    # Set page config
    st.set_page_config(
        page_title="Clinical Trials in Canada",
        page_icon="🧬",
        layout="wide"
    )
    
    # Header
    st.title("Clinical Trials Database Explorer - Canada")
    
    # Description
    st.markdown("""
    ## Welcome to the Clinical Trials Database Explorer
    
    This application provides comprehensive data analysis and visualization of clinical trials conducted in Canada,
    with specialized features for pediatric trials and rare disease classification using AI.
    
    ### Available Features:
    """)
    
    # Create cards for each page
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Pediatric Clinical Trials
        
        Explore clinical trials in Canada that include participants **under 18 years old**.
        
        Features:
        - Search and filter pediatric trials by status, phase, year, keywords, and conditions
        - View interactive visualizations by phase, status, and year
        - Explore geographic distribution with population-normalized metrics
        - View consolidated metropolitan area data (Greater Toronto Area and Metro Vancouver)
        - Access detailed trial information and condition analysis
        """)
        
        if st.button("Go to Pediatric Trials", key="pediatric_button", use_container_width=True):
            st.switch_page("pages/1_Pediatric_Clinical_Trials.py")
            
        st.markdown("---")
        
        st.markdown("""
        ### 🔍 Rare Disease Classifier
        
        Identify pediatric clinical trials that are focused on rare diseases using Anthropic's Claude AI.
        
        Features:
        - Automatically classify trials as rare or non-rare with detailed reasoning
        - View classification confidence levels and rare disease indicators
        - Export classification results for further analysis
        - Filter and search potential rare disease trials
        """)
        
        if st.button("Go to Rare Disease Classifier", key="rare_button", use_container_width=True):
            st.switch_page("pages/4_Rare_Disease_Classifier.py")
            
        st.markdown("---")
        
        st.markdown("""
        ### 🔬 Rare Disease Trials Analysis
        
        Explore clinical trials in Canada that have been classified as rare disease trials.
        
        Features:
        - View pre-classified rare disease trials with AI-generated reasoning
        - Filter rare disease trials by various criteria
        - Analyze condition categories (Oncology vs. Other)
        - Visualize intervention patterns in rare disease research
        - Explore condition co-occurrence networks
        """)
        
        if st.button("Go to Rare Disease Trials", key="rare_trials_button", use_container_width=True):
            st.switch_page("pages/3_Rare_Disease_Trials.py")
    
    with col2:
        st.markdown("""
        ### 👩‍⚕️ Adult Clinical Trials
        
        Explore clinical trials in Canada that include participants **18 years or older**.
        
        Features:
        - Search and filter adult trials using the same powerful interface
        - Compare visualization patterns between adult and pediatric research
        - Explore geographic distribution with population-normalization
        - View detailed trial information and metadata
        - Analyze intervention and condition patterns
        """)
        
        if st.button("Go to Adult Trials", key="adult_button", use_container_width=True):
            st.switch_page("pages/2_Adult_Clinical_Trials.py")
            
        st.markdown("---")
        
        st.markdown("""
        ### ℹ️ About This Application
        
        This application connects to the AACT (Aggregate Analysis of ClinicalTrials.gov) database,
        which provides comprehensive data on clinical trials registered on ClinicalTrials.gov.
        
        **Key Features:**
        - **Metropolitan Area Analysis**: Consolidates cities within the Greater Toronto Area and Metro Vancouver
        - **Population-Normalized Metrics**: Visualizes trial density per 100,000 residents
        - **AI-Powered Classification**: Uses Anthropic's Claude to identify rare disease trials
        - **Advanced Data Visualization**: Interactive charts, maps, and network diagrams
        
        **Technologies:**
        - Streamlit for the web interface
        - Plotly for interactive visualizations
        - NetworkX for condition network analysis
        - Anthropic's Claude AI for rare disease classification
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### How to Use This Application
    
    1. Navigate to any of the main modules using the buttons above
    2. Apply filters to focus on specific trial subsets
    3. Explore the various visualization tabs to gain insights
    4. View detailed information about specific trials of interest
    5. For rare disease research, use the classifier to identify relevant trials
    
    **Data Privacy Note**: This application processes clinical trial data locally and only uses the Anthropic API
    for rare disease classification when explicitly requested.
    
    **Source Code**: Find more information in the project README and documentation.
    """)

if __name__ == "__main__":
    main()