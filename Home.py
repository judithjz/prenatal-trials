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
    
    This application allows you to explore and analyze clinical trials conducted in Canada, 
    with a focus on different age groups and rare disease classification.
    
    ### Available Pages:
    """)
    
    # Create cards for each page
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Pediatric Clinical Trials
        
        Explore clinical trials in Canada that include participants **under 18 years old**.
        
        Features:
        - Search and filter pediatric trials
        - View interactive visualizations by phase, status, year
        - Explore geographic distribution across Canada
        - View detailed trial information
        """)
        
        if st.button("Go to Pediatric Trials", key="pediatric_button", use_container_width=True):
            st.switch_page("pages/Pediatric_Clinical_Trials.py")
            
        st.markdown("---")
        
        st.markdown("""
        ### 🔍 Rare Disease Classifier
        
        Identify pediatric clinical trials that are focused on rare diseases using AI-powered classification.
        
        Features:
        - Automatically classify trials as rare or non-rare
        - View classification reasoning
        - Export classification results
        - Filter and search rare disease trials
        """)
        
        if st.button("Go to Rare Disease Classifier", key="rare_button", use_container_width=True):
            st.switch_page("pages/Rare_Disease_Classifier.py")
            
        st.markdown("---")
        
        st.markdown("""
        ### 🔬 Rare Disease Trials
        
        Explore clinical trials in Canada that have been classified as rare disease trials.
        
        Features:
        - View pre-classified rare disease trials
        - See AI classification reasoning
        - Explore geographic distribution
        - View specialized rare disease metrics
        """)
        
        if st.button("Go to Rare Disease Trials", key="rare_trials_button", use_container_width=True):
            st.switch_page("pages/Rare_Disease_Trials.py")
    
    with col2:
        st.markdown("""
        ### 👩‍⚕️ Adult Clinical Trials
        
        Explore clinical trials in Canada that include participants **18 years or older**.
        
        Features:
        - Search and filter adult trials
        - View interactive visualizations by phase, status, year
        - Explore geographic distribution across Canada
        - View detailed trial information
        """)
        
        if st.button("Go to Adult Trials", key="adult_button", use_container_width=True):
            st.switch_page("pages/Adult_Trials.py")
            
        st.markdown("---")
        
        st.markdown("""
        ### ℹ️ About This Application
        
        This application connects to the AACT (Aggregate Analysis of ClinicalTrials.gov) database, 
        which contains information about clinical trials registered on ClinicalTrials.gov.
        
        **Data Source**: AACT Database
        
        **Technologies Used**:
        - Streamlit for the web interface
        - Plotly for interactive visualizations
        - Anthropic's Claude AI for rare disease classification
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### How to Use This Application
    
    1. Use the buttons above to navigate to the desired page
    2. Apply filters to narrow down the trials you're interested in
    3. Explore the visualizations to gain insights
    4. View detailed information about specific trials
    5. For rare disease classification, follow the instructions on the Rare Disease Classifier page
    
    This application is designed for researchers, healthcare professionals, and anyone interested 
    in exploring clinical trial data from Canada.
    """)

if __name__ == "__main__":
    main()