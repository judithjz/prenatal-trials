import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

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
    plot_phase_distribution,
    plot_status_distribution,
    plot_yearly_trends,
    render_city_visualization,
    render_summary_metrics
)

from utils.api_utils import (
    get_download_link
)


def classify_condition(condition):
    """
    Classify a condition into a category.
    Currently only distinguishes between oncology and other conditions.
    
    Args:
        condition: Condition name to classify
        
    Returns:
        Category name (Oncology or Other)
    """
    oncology_keywords = [
        'cancer', 'tumor', 'tumour', 'leukemia', 'lymphoma', 'oncology', 
        'sarcoma', 'carcinoma', 'melanoma', 'neoplasm', 'neoplastic', 
        'myeloma', 'adenocarcinoma', 'oncologic', 'malignant', 'metastatic',
        'metastasis', 'neuroendocrine', 'glioma', 'glioblastoma', 'neuroblastoma',
        'astrocytoma', 'ependymoma', 'craniopharyngioma', 'osteosarcoma'
    ]
    
    if condition:
        condition_lower = condition.lower()
        for keyword in oncology_keywords:
            if keyword in condition_lower:
                return 'Oncology'
    
    return 'Other'


def count_intervention_types(interventions_df):
    """Count the frequency of each intervention type."""
    return dict(Counter(interventions_df['intervention_type']))


def count_top_conditions(conditions_dict, limit=15, by_category=False):
    """
    Count the frequency of each condition across all trials.
    
    Args:
        conditions_dict: Dictionary mapping NCT IDs to lists of conditions
        limit: Max number of top conditions to return
        by_category: Whether to return data grouped by category
        
    Returns:
        DataFrame with condition counts, sorted by frequency
    """
    # Flatten the conditions list
    all_conditions = []
    for conditions in conditions_dict.values():
        all_conditions.extend(conditions)
    
    # Count and sort
    condition_counts = Counter(all_conditions)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'condition': list(condition_counts.keys()),
        'count': list(condition_counts.values())
    })
    
    # Add category
    df['category'] = df['condition'].apply(classify_condition)
    
    # Sort by count (descending)
    df = df.sort_values('count', ascending=False)
    
    if by_category:
        # Return all data grouped by category
        return df
    else:
        # Return top N conditions
        return df.head(limit)


def plot_intervention_distribution(interventions_df):
    """Create a pie chart showing the distribution of intervention types."""
    intervention_counts = count_intervention_types(interventions_df)
    
    df = pd.DataFrame({
        'intervention_type': list(intervention_counts.keys()),
        'count': list(intervention_counts.values())
    })
    
    fig = px.pie(
        df,
        values='count',
        names='intervention_type',
        title='Distribution of Intervention Types',
        hole=0.4
    )
    
    fig.update_layout(legend_title="Intervention Type")
    
    return fig


def plot_top_conditions(conditions_dict, limit=15, by_category=True):
    """Create a horizontal bar chart of the top conditions, colored by category."""
    # Get condition data with categories
    df = count_top_conditions(conditions_dict, limit, by_category=True)
    
    if by_category:
        # Create a separate chart for each category
        fig = go.Figure()
        
        # Get top conditions for each category
        oncology_df = df[df['category'] == 'Oncology'].head(limit)
        other_df = df[df['category'] == 'Other'].head(limit)
        
        # Add Oncology conditions
        if not oncology_df.empty:
            fig.add_trace(go.Bar(
                y=oncology_df['condition'],
                x=oncology_df['count'],
                orientation='h',
                name='Oncology',
                marker_color='#FF4136',  # Red
                customdata=oncology_df['category'],
                hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Category: %{customdata}<extra></extra>'
            ))
        
        # Add Other conditions
        if not other_df.empty:
            fig.add_trace(go.Bar(
                y=other_df['condition'],
                x=other_df['count'],
                orientation='h',
                name='Other',
                marker_color='#0074D9',  # Blue
                customdata=other_df['category'],
                hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Category: %{customdata}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Top Conditions in Rare Disease Trials by Category',
            xaxis_title='Number of Trials',
            yaxis_title='Condition',
            height=max(400, min(limit * 25 * 2, 800)),
            legend_title='Category',
            barmode='group'
        )
        
    else:
        # Limited to top N overall
        df = df.head(limit)
        
        # Use different colors for different categories
        color_map = {'Oncology': '#FF4136', 'Other': '#0074D9'}
        
        fig = px.bar(
            df,
            y='condition',
            x='count',
            orientation='h',
            title=f'Top {limit} Conditions in Rare Disease Trials',
            labels={'condition': 'Condition', 'count': 'Number of Trials', 'category': 'Category'},
            color='category',
            color_discrete_map=color_map,
            hover_data=['category']
        )
        
        fig.update_layout(
            height=max(400, min(limit * 25, 800)),
            yaxis={'categoryorder': 'total ascending'}
        )
    
    return fig


def plot_conditions_network(conditions_dict, min_co_occurrence=2):
    """
    Create a network diagram showing conditions that co-occur in trials.
    
    Args:
        conditions_dict: Dictionary mapping NCT IDs to lists of conditions
        min_co_occurrence: Minimum number of co-occurrences to include in the network
        
    Returns:
        Plotly graph object figure or None if networkx is not available
    """
    # Check if networkx is available
    try:
        import networkx as nx
    except ImportError:
        st.warning("The networkx library is required for the conditions network visualization. Please install it with `pip install networkx`.")
        return None
    
    # Count co-occurrences
    co_occurrences = {}
    
    for conditions in conditions_dict.values():
        if len(conditions) > 1:
            # For each pair of conditions in this trial
            for i, cond1 in enumerate(conditions):
                for cond2 in conditions[i+1:]:
                    pair = tuple(sorted([cond1, cond2]))
                    if pair in co_occurrences:
                        co_occurrences[pair] += 1
                    else:
                        co_occurrences[pair] = 1
    
    # Filter by minimum co-occurrence
    filtered_co_occurrences = {pair: count for pair, count in co_occurrences.items() 
                             if count >= min_co_occurrence}
    
    if not filtered_co_occurrences:
        return None
    
    # Get unique conditions
    unique_conditions = set()
    for pair in filtered_co_occurrences.keys():
        unique_conditions.add(pair[0])
        unique_conditions.add(pair[1])
    
    # Create node DataFrame
    nodes = pd.DataFrame({
        'id': list(range(len(unique_conditions))),
        'label': list(unique_conditions)
    })
    
    # Add category to nodes
    nodes['category'] = nodes['label'].apply(classify_condition)
    
    # Create mapping from condition to node ID
    condition_to_id = {cond: idx for idx, cond in enumerate(unique_conditions)}
    
    # Create edge DataFrame
    edges = []
    for pair, weight in filtered_co_occurrences.items():
        source_id = condition_to_id[pair[0]]
        target_id = condition_to_id[pair[1]]
        edges.append({
            'source': source_id,
            'target': target_id,
            'weight': weight
        })
    
    edges_df = pd.DataFrame(edges)
    
    # Create network visualization using plotly
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create network layout (using Fruchterman-Reingold algorithm)
    G = nx.Graph()
    
    # Add nodes
    for _, row in nodes.iterrows():
        G.add_node(row['id'], label=row['label'], category=row['category'])
    
    # Add edges
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    
    # Calculate layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Add edges to trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Create separate node traces for each category
    oncology_nodes = {'x': [], 'y': [], 'text': [], 'id': []}
    other_nodes = {'x': [], 'y': [], 'text': [], 'id': []}
    
    # Assign nodes to the appropriate category
    for node in G.nodes():
        x, y = pos[node]
        category = G.nodes[node]['category']
        node_info = G.nodes[node]['label']
        
        if category == 'Oncology':
            oncology_nodes['x'].append(x)
            oncology_nodes['y'].append(y)
            oncology_nodes['text'].append(node_info)
            oncology_nodes['id'].append(node)
        else:
            other_nodes['x'].append(x)
            other_nodes['y'].append(y)
            other_nodes['text'].append(node_info)
            other_nodes['id'].append(node)
    
    # Create oncology node trace
    oncology_node_trace = go.Scatter(
        x=oncology_nodes['x'],
        y=oncology_nodes['y'],
        text=oncology_nodes['text'],
        mode='markers',
        name='Oncology',
        marker=dict(
            color='#FF4136',  # Red
            size=15,
            line=dict(width=2)
        ),
        hovertemplate='<b>%{text}</b><br>Category: Oncology<extra></extra>'
    )
    
    # Create other node trace
    other_node_trace = go.Scatter(
        x=other_nodes['x'],
        y=other_nodes['y'],
        text=other_nodes['text'],
        mode='markers',
        name='Other',
        marker=dict(
            color='#0074D9',  # Blue
            size=15,
            line=dict(width=2)
        ),
        hovertemplate='<b>%{text}</b><br>Category: Other<extra></extra>'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, oncology_node_trace, other_node_trace],
                   layout=go.Layout(
                       title=dict(
                           text='Co-occurring Conditions Network (Colored by Category)',
                           font=dict(size=16)
                       ),
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       legend=dict(
                           title='Category',
                           yanchor="top",
                           y=0.99,
                           xanchor="right",
                           x=0.99
                       ),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig


def plot_interventions_by_condition(interventions_df, conditions_dict, top_n=10):
    """
    Create a heatmap showing the relationship between top conditions and intervention types.
    
    Args:
        interventions_df: DataFrame with intervention data
        conditions_dict: Dictionary mapping NCT IDs to lists of conditions
        top_n: Number of top conditions to include per category
        
    Returns:
        Plotly heatmap figure
    """
    # Get top conditions by category
    top_conditions_df = count_top_conditions(conditions_dict, top_n, by_category=True)
    
    # Get top conditions for each category
    oncology_top = top_conditions_df[top_conditions_df['category'] == 'Oncology'].head(top_n)['condition'].tolist()
    other_top = top_conditions_df[top_conditions_df['category'] == 'Other'].head(top_n)['condition'].tolist()
    
    # Combine top conditions from both categories
    top_conditions = oncology_top + other_top
    
    # Create mapping of nct_id to intervention types
    nct_to_interventions = {}
    for _, row in interventions_df.iterrows():
        if row['nct_id'] not in nct_to_interventions:
            nct_to_interventions[row['nct_id']] = []
        nct_to_interventions[row['nct_id']].append(row['intervention_type'])
    
    # Count intervention types for each top condition
    condition_intervention_counts = {}
    for nct_id, conditions in conditions_dict.items():
        if nct_id in nct_to_interventions:
            intervention_types = nct_to_interventions[nct_id]
            for condition in conditions:
                if condition in top_conditions:
                    if condition not in condition_intervention_counts:
                        condition_intervention_counts[condition] = Counter()
                    condition_intervention_counts[condition].update(intervention_types)
    
    # Get unique intervention types
    all_intervention_types = set()
    for counter in condition_intervention_counts.values():
        all_intervention_types.update(counter.keys())
    all_intervention_types = sorted(all_intervention_types)
    
    # Create heatmap data
    heatmap_data = []
    condition_labels = []
    categories = []
    
    for condition in top_conditions:
        if condition in condition_intervention_counts:
            counter = condition_intervention_counts[condition]
            row = [counter.get(int_type, 0) for int_type in all_intervention_types]
            heatmap_data.append(row)
            condition_labels.append(condition)
            categories.append('Oncology' if condition in oncology_top else 'Other')
    
    if not heatmap_data:
        return None
    
    # Create a custom colorscale that's different for each category
    # For Oncology (red colors) and Other (blue colors)
    customdata = [['Oncology' if cat == 'Oncology' else 'Other' for _ in all_intervention_types] for cat in categories]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=all_intervention_types,
        y=condition_labels,
        colorscale='Viridis',
        customdata=customdata,
        hovertemplate='<b>Condition:</b> %{y}<br><b>Intervention:</b> %{x}<br><b>Count:</b> %{z}<br><b>Category:</b> %{customdata}<extra></extra>'
    ))
    
    # Add category indicator (colored rectangles on the left)
    for i, (condition, category) in enumerate(zip(condition_labels, categories)):
        color = '#FF4136' if category == 'Oncology' else '#0074D9'  # Red for Oncology, Blue for Other
        fig.add_shape(
            type="rect",
            xref="paper", yref="y",
            x0=-0.08, x1=-0.02,
            y0=i-0.5, y1=i+0.5,
            fillcolor=color,
            line=dict(width=0)
        )
    
    # Add legend manually
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.01, x1=0.03,
        y0=0.97, y1=0.99,
        fillcolor='#FF4136',
        line=dict(width=0)
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.09, y=0.98,
        text="Oncology",
        showarrow=False,
        align="left"
    )
    
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.01, x1=0.03,
        y0=0.93, y1=0.95,
        fillcolor='#0074D9',
        line=dict(width=0)
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.08, y=0.94,
        text="Other",
        showarrow=False,
        align="left"
    )
    
    fig.update_layout(
        title='Intervention Types by Top Conditions (Categorized)',
        xaxis_title='Intervention Type',
        yaxis_title='Condition',
        height=max(500, len(condition_labels) * 25),
        margin=dict(l=120)  # Extra left margin for the category indicators
    )
    
    return fig


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
    
    with conn:
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
                    
                    # Visualization 1: Trials by Phase
                    st.write("### Trials by Phase")
                    fig_phase = plot_phase_distribution(filtered_df)
                    st.plotly_chart(fig_phase, use_container_width=True)
                    
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
                    render_city_visualization(filtered_df, conn)
                
                with tab4:
                    st.subheader("Interventions & Conditions Analysis")
                    
                    # Get the filtered NCT IDs
                    filtered_nct_ids = filtered_df['nct_id'].tolist()
                    
                    # Filter interventions and conditions data
                    interventions_df = safe_get_session_state('rare_interventions_df')
                    conditions_dict = safe_get_session_state('rare_conditions_dict', {})
                    
                    if interventions_df is not None and not interventions_df.empty:
                        # Filter interventions to only those in filtered trials
                        filtered_interventions = interventions_df[interventions_df['nct_id'].isin(filtered_nct_ids)]
                        
                        if not filtered_interventions.empty:
                            # Create intervention type distribution chart
                            st.write("### Intervention Types Distribution")
                            fig_intervention = plot_intervention_distribution(filtered_interventions)
                            st.plotly_chart(fig_intervention, use_container_width=True)
                            
                            # Summary statistics for interventions
                            int_counts = count_intervention_types(filtered_interventions)
                            
                            # Create 2-column layout for intervention stats
                            cols = st.columns(len(int_counts) if len(int_counts) <= 4 else 4)
                            for i, (int_type, count) in enumerate(sorted(int_counts.items(), key=lambda x: x[1], reverse=True)):
                                if i < len(cols):
                                    cols[i].metric(f"{int_type}", count)
                        else:
                            st.info("No intervention data available for the selected trials.")
                    
                    # Filter conditions dict to only include filtered trials
                    if conditions_dict:
                        filtered_conditions_dict = {nct_id: conditions for nct_id, conditions in conditions_dict.items() 
                                                 if nct_id in filtered_nct_ids}
                        
                        if filtered_conditions_dict:
                            # Create conditions bar chart with categories
                            st.write("### Top Conditions in Rare Disease Trials by Category")
                            
                            # Add radio button to choose visualization style
                            viz_style = st.radio(
                                "Visualization Style:",
                                ["Split by Category", "Combined"],
                                horizontal=True,
                                key="conditions_viz_style"
                            )
                            
                            # Get top conditions for display
                            top_n_conditions = st.slider(
                                "Number of Top Conditions Per Category",
                                min_value=5,
                                max_value=20,
                                value=10,
                                key="top_conditions_count"
                            )
                            
                            fig_conditions = plot_top_conditions(
                                filtered_conditions_dict, 
                                limit=top_n_conditions, 
                                by_category=(viz_style == "Split by Category")
                            )
                            st.plotly_chart(fig_conditions, use_container_width=True)
                            
                            # Show condition classification stats
                            all_conditions_df = count_top_conditions(filtered_conditions_dict, by_category=True)
                            oncology_count = len(all_conditions_df[all_conditions_df['category'] == 'Oncology'])
                            other_count = len(all_conditions_df[all_conditions_df['category'] == 'Other'])
                            total_count = len(all_conditions_df)
                            
                            # Display stats in columns
                            cols = st.columns(3)
                            cols[0].metric("Total Unique Conditions", total_count)
                            cols[1].metric("Oncology Conditions", oncology_count, 
                                          f"{oncology_count/total_count:.1%}" if total_count > 0 else "0%")
                            cols[2].metric("Other Conditions", other_count,
                                          f"{other_count/total_count:.1%}" if total_count > 0 else "0%")
                            
                            # Create conditions network visualization
                            st.write("### Conditions Co-occurrence Network")
                            st.info("This network shows conditions that frequently appear together in trials. Nodes are colored by category - red for Oncology and blue for Other conditions.")
                            
                            # Check if networkx is installed
                            try:
                                import networkx
                                
                                # Add slider for minimum co-occurrence
                                min_co_occurrence = st.slider(
                                    "Minimum Co-occurrence Threshold",
                                    min_value=1,
                                    max_value=10,
                                    value=2,
                                    help="Minimum number of times conditions must co-occur to be included in the network"
                                )
                                
                                fig_network = plot_conditions_network(filtered_conditions_dict, min_co_occurrence)
                                if fig_network:
                                    st.plotly_chart(fig_network, use_container_width=True)
                                else:
                                    st.info(f"No conditions co-occur at least {min_co_occurrence} times in the selected trials.")
                            except ImportError:
                                st.error("The networkx library is required for the conditions network visualization.")
                                st.info("Please install networkx with: `pip install networkx` and restart the application.")
                                
                                # Display alternative visualization - simple co-occurrence matrix
                                st.write("### Top Condition Co-occurrences (Alternative View)")
                                
                                # Count co-occurrences
                                co_occurrences = {}
                                for conditions in filtered_conditions_dict.values():
                                    if len(conditions) > 1:
                                        for i, cond1 in enumerate(conditions):
                                            for cond2 in conditions[i+1:]:
                                                pair = tuple(sorted([cond1, cond2]))
                                                if pair in co_occurrences:
                                                    co_occurrences[pair] += 1
                                                else:
                                                    co_occurrences[pair] = 1
                                
                                # Show top co-occurrences as a table
                                if co_occurrences:
                                    co_occur_df = pd.DataFrame([
                                        {
                                            "Condition 1": pair[0], 
                                            "Category 1": classify_condition(pair[0]),
                                            "Condition 2": pair[1], 
                                            "Category 2": classify_condition(pair[1]),
                                            "Co-occurrences": count
                                        }
                                        for pair, count in sorted(co_occurrences.items(), key=lambda x: x[1], reverse=True)[:15]
                                    ])
                                    st.dataframe(co_occur_df, use_container_width=True)
                                else:
                                    st.info("No condition co-occurrences found in the selected trials.")
                            
                            # Create intervention-condition heatmap if we have both datasets
                            if interventions_df is not None and not interventions_df.empty:
                                filtered_interventions = interventions_df[interventions_df['nct_id'].isin(filtered_nct_ids)]
                                
                                if not filtered_interventions.empty:
                                    st.write("### Intervention Types by Top Conditions")
                                    
                                    # Add slider for number of top conditions
                                    top_n = st.slider(
                                        "Number of Top Conditions per Category",
                                        min_value=5,
                                        max_value=20,
                                        value=8,
                                        help="Number of top conditions per category to include in the heatmap"
                                    )
                                    
                                    fig_heatmap = plot_interventions_by_condition(
                                        filtered_interventions, 
                                        filtered_conditions_dict,
                                        top_n
                                    )
                                    
                                    if fig_heatmap:
                                        st.plotly_chart(fig_heatmap, use_container_width=True)
                                    else:
                                        st.info("No data available for intervention-condition heatmap.")
                                    
                                    # Add summary of intervention types by category
                                    st.write("### Intervention Types by Condition Category")
                                    
                                    # Prepare data
                                    interventions_by_category = {}
                                    
                                    # Create mapping of nct_id to intervention types
                                    nct_to_interventions = {}
                                    for _, row in filtered_interventions.iterrows():
                                        if row['nct_id'] not in nct_to_interventions:
                                            nct_to_interventions[row['nct_id']] = []
                                        nct_to_interventions[row['nct_id']].append(row['intervention_type'])
                                    
                                    for nct_id, intervention_types in nct_to_interventions.items():
                                        if nct_id in filtered_nct_ids and nct_id in conditions_dict:
                                            trial_conditions = conditions_dict[nct_id]
                                            for condition in trial_conditions:
                                                category = classify_condition(condition)
                                                if category not in interventions_by_category:
                                                    interventions_by_category[category] = Counter()
                                                interventions_by_category[category].update(intervention_types)
                                    
                                    # Create data for bar chart
                                    categories = []
                                    intervention_types = []
                                    counts = []
                                    
                                    for category, counter in interventions_by_category.items():
                                        for int_type, count in counter.items():
                                            categories.append(category)
                                            intervention_types.append(int_type)
                                            counts.append(count)
                                    
                                    chart_df = pd.DataFrame({
                                        'Category': categories,
                                        'InterventionType': intervention_types,
                                        'Count': counts
                                    })
                                    
                                    if not chart_df.empty:
                                        # Create grouped bar chart
                                        fig = px.bar(
                                            chart_df,
                                            x='InterventionType',
                                            y='Count',
                                            color='Category',
                                            barmode='group',
                                            title='Intervention Types by Condition Category',
                                            color_discrete_map={'Oncology': '#FF4136', 'Other': '#0074D9'}
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No condition data available for the selected trials.")
                
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


if __name__ == "__main__":
    main()
