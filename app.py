"""
app.py - golf shot performance analyzer streamlit dashboard

this is the main application that ties together all modules into an
interactive dashboard. it demonstrates the full data science workflow:
data ingestion → cleaning → analysis → visualization → insight generation.

designed to showcase skills relevant to taylormade performance research:
- automated data pipelines
- statistical analysis (anova, regression)
- interactive dashboards
- clear communication of findings

author: eric bolander
project: golf shot performance analyzer (taylormade portfolio project)
deployment: streamlit cloud (https://streamlit.io/cloud)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import (
    load_data, clean_data, segment_by_skill, filter_by_ball_type,
    get_data_summary, THRESHOLDS, SKILL_LEVELS
)
from analysis import (
    calculate_consistency_metrics, calculate_performance_metrics,
    compare_skill_levels, run_anova_analysis, run_regression_analysis,
    calculate_dispersion_ellipse, generate_performance_summary
)
from visualizations import (
    create_dispersion_plot, create_skill_level_comparison,
    create_consistency_comparison, create_launch_condition_scatter,
    create_correlation_heatmap, create_multi_skill_dispersion,
    create_regression_visualization, create_performance_dashboard_summary
)
from insights import (
    generate_performance_insights, generate_skill_comparison_insights,
    generate_regression_insights, generate_optimal_launch_insights,
    generate_full_report
)


# =============================================================================
# page configuration
# =============================================================================
st.set_page_config(
    page_title="Golf Shot Performance Analyzer",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css for cleaner appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# data loading with caching
# =============================================================================
@st.cache_data
def load_and_process_data(filepath):
    """
    load and process data with streamlit caching for performance.
    caching ensures data is only loaded once, not on every rerun.
    """
    raw_df = load_data(filepath)
    clean_df, removed_df = clean_data(raw_df)
    segments = segment_by_skill(clean_df)
    return raw_df, clean_df, removed_df, segments


# =============================================================================
# sidebar: data source and filters
# =============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/3/3a/TaylorMade_Golf_logo.svg/220px-TaylorMade_Golf_logo.svg.png", width=150)
    st.title("⛳ Shot Analyzer")
    st.markdown("---")
    
    # data source selection
    st.subheader("📁 Data Source")
    data_option = st.radio(
        "Select data source:",
        ["Use Sample Data", "Upload CSV"],
        index=0
    )
    
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload Trackman CSV",
            type=['csv'],
            help="Upload a Trackman export CSV file"
        )
        if uploaded_file is not None:
            # save uploaded file temporarily
            temp_path = "/tmp/uploaded_data.csv"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_path = temp_path
        else:
            st.info("👆 Upload a file or use sample data")
            st.stop()
    else:
        # use sample data (from project or data folder)
        if os.path.exists('/mnt/project/data.csv'):
            data_path = '/mnt/project/data.csv'
        elif os.path.exists('data/sample_data.csv'):
            data_path = 'data/sample_data.csv'
        else:
            st.error("Sample data not found. Please upload a CSV file.")
            st.stop()
    
    st.markdown("---")
    
    # filters
    st.subheader("🔧 Filters")
    
    # load data to get filter options
    try:
        raw_df, clean_df, removed_df, segments = load_and_process_data(data_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # ball type filter
    ball_types = ['All'] + list(clean_df['ball_type'].dropna().unique())
    selected_ball = st.selectbox(
        "Ball Type:",
        ball_types,
        index=0
    )
    
    # skill level filter
    skill_options = {
        'All': 'all',
        'Elite (145+ mph)': 'elite',
        'Scratch (130-145 mph)': 'scratch',
        'Mid Handicap (110-130 mph)': 'mid',
        'High Handicap (<110 mph)': 'high'
    }
    selected_skill = st.selectbox(
        "Skill Level:",
        list(skill_options.keys()),
        index=0
    )
    
    # apply filters
    if selected_ball != 'All':
        for level in segments:
            segments[level] = filter_by_ball_type(segments[level], selected_ball)
    
    # select working dataset
    skill_key = skill_options[selected_skill]
    if skill_key in segments:
        working_df = segments[skill_key]
    else:
        working_df = clean_df
    
    # exclude mishits toggle
    exclude_mishits = st.checkbox(
        "Exclude Mishits (<80 mph ball speed)",
        value=True
    )
    
    if exclude_mishits and 'is_mishit' in working_df.columns:
        working_df = working_df[~working_df['is_mishit']]
    
    st.markdown("---")
    st.caption(f"📊 {len(working_df):,} shots displayed")
    
    # about section
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    **Author:** Eric Bolander  
    **Project:** TaylorMade Portfolio
    
    Built to demonstrate:
    - Python data pipelines
    - Statistical analysis
    - Interactive dashboards
    - Golf domain expertise
    
    [GitHub](https://github.com/Ebolander23) | [LinkedIn](https://www.linkedin.com/in/eric-bolander-b1333b176/)
    """)


# =============================================================================
# main content: tabbed interface
# =============================================================================
st.markdown('<p class="main-header">⛳ Golf Shot Performance Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Trackman data analysis for equipment testing and player fitting</p>', unsafe_allow_html=True)

# create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🎯 Dispersion Analysis",
    "📈 Statistical Analysis",
    "🔬 Skill Comparison",
    "📋 Full Report"
])


# =============================================================================
# tab 1: overview dashboard
# =============================================================================
with tab1:
    st.header("Performance Overview")
    
    if len(working_df) == 0:
        st.warning("No data available with current filters. Try adjusting your selections.")
        st.stop()
    
    # key metrics in columns
    perf = calculate_performance_metrics(working_df)
    cons = calculate_consistency_metrics(working_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🚀 Ball Speed",
            value=f"{perf.ball_speed_mean:.1f} mph",
            delta=f"±{working_df['ball_speed'].std():.1f} std"
        )
    
    with col2:
        st.metric(
            label="📏 Carry Distance",
            value=f"{perf.carry_mean:.1f} yds",
            delta=f"±{cons.carry_std:.1f} std"
        )
    
    with col3:
        st.metric(
            label="🔄 Spin Rate",
            value=f"{perf.spin_rate_mean:.0f} rpm",
            delta=f"±{cons.spin_std:.0f} std"
        )
    
    with col4:
        st.metric(
            label="🎯 Smash Factor",
            value=f"{perf.smash_factor_mean:.3f}",
            delta=f"±{cons.smash_factor_std:.3f} std"
        )
    
    st.markdown("---")
    
    # two-column layout for main visualizations
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("Shot Dispersion")
        dispersion_fig = create_dispersion_plot(
            working_df,
            title=f"Shot Pattern ({len(working_df):,} shots)",
            show_ellipse=True
        )
        st.plotly_chart(dispersion_fig, use_container_width=True)
    
    with col_right:
        st.subheader("Key Insights")
        insights = generate_performance_insights(working_df, "Selected Data")
        for insight in insights:
            st.markdown(insight)
    
    # distribution histograms
    st.markdown("---")
    st.subheader("Distributions")
    
    dist_col1, dist_col2, dist_col3 = st.columns(3)
    
    with dist_col1:
        fig_carry = px.histogram(
            working_df, x='carry', nbins=30,
            title="Carry Distance",
            labels={'carry': 'Carry (yards)', 'count': 'Count'}
        )
        fig_carry.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_carry, use_container_width=True)
    
    with dist_col2:
        fig_speed = px.histogram(
            working_df, x='ball_speed', nbins=30,
            title="Ball Speed",
            labels={'ball_speed': 'Ball Speed (mph)', 'count': 'Count'}
        )
        fig_speed.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_speed, use_container_width=True)
    
    with dist_col3:
        fig_spin = px.histogram(
            working_df, x='spin_rate', nbins=30,
            title="Spin Rate",
            labels={'spin_rate': 'Spin Rate (rpm)', 'count': 'Count'}
        )
        fig_spin.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_spin, use_container_width=True)


# =============================================================================
# tab 2: dispersion analysis
# =============================================================================
with tab2:
    st.header("Dispersion Analysis")
    st.markdown("""
    Dispersion analysis shows where shots land relative to the target line. 
    This is critical for equipment fitting — a tighter pattern means more predictable results.
    """)
    
    # dispersion with color options
    color_option = st.selectbox(
        "Color shots by:",
        ['None', 'Spin Rate', 'Ball Speed', 'Smash Factor', 'Launch Angle']
    )
    
    color_map = {
        'None': None,
        'Spin Rate': 'spin_rate',
        'Ball Speed': 'ball_speed',
        'Smash Factor': 'smash_factor',
        'Launch Angle': 'launch_angle'
    }
    
    dispersion_color_fig = create_dispersion_plot(
        working_df,
        title="Shot Dispersion Pattern",
        show_ellipse=True,
        color_by=color_map[color_option]
    )
    st.plotly_chart(dispersion_color_fig, use_container_width=True)
    
    # dispersion statistics
    st.markdown("---")
    st.subheader("Dispersion Statistics")
    
    ellipse = calculate_dispersion_ellipse(working_df)
    
    disp_col1, disp_col2, disp_col3, disp_col4 = st.columns(4)
    
    with disp_col1:
        st.metric("Center Carry", f"{ellipse['center_carry']:.1f} yds")
    with disp_col2:
        st.metric("Center Offline", f"{ellipse['center_offline']:.1f} yds")
    with disp_col3:
        st.metric("68% Width", f"±{ellipse['offline_std']:.1f} yds")
    with disp_col4:
        st.metric("68% Depth", f"±{ellipse['carry_std']:.1f} yds")
    
    st.info(f"""
    **Interpretation**: 68% of shots land within ±{ellipse['offline_std']:.0f} yards of the 
    target line and within ±{ellipse['carry_std']:.0f} yards of the average carry distance.
    """)


# =============================================================================
# tab 3: statistical analysis
# =============================================================================
with tab3:
    st.header("Statistical Analysis")
    
    analysis_type = st.radio(
        "Select analysis type:",
        ["Correlation Analysis", "Regression Model", "ANOVA (Skill Comparison)"],
        horizontal=True
    )
    
    if analysis_type == "Correlation Analysis":
        st.subheader("Correlation Matrix")
        st.markdown("""
        Correlation shows how variables relate: +1 = perfect positive, -1 = perfect negative, 0 = no relationship.
        """)
        
        corr_fig = create_correlation_heatmap(working_df)
        st.plotly_chart(corr_fig, use_container_width=True)
        
        st.markdown("""
        **Key Observations:**
        - Ball speed and carry have the strongest correlation (~0.9) — speed is king
        - Spin rate and carry are negatively correlated — more spin = less distance
        - Smash factor and carry are positively correlated — strike quality matters
        """)
    
    elif analysis_type == "Regression Model":
        st.subheader("Predicting Carry Distance")
        st.markdown("""
        Linear regression quantifies how launch conditions affect distance.
        The model predicts carry from ball speed, launch angle, and spin rate.
        """)
        
        # run regression
        reg_result = run_regression_analysis(
            working_df, 'carry',
            ['ball_speed', 'launch_angle', 'spin_rate']
        )
        
        if 'error' not in reg_result:
            # display r-squared
            r2_col, n_col = st.columns(2)
            with r2_col:
                st.metric("R-Squared", f"{reg_result['r_squared']:.3f}")
            with n_col:
                st.metric("Observations", f"{reg_result['n_observations']:,}")
            
            st.markdown(f"**Model Quality**: {reg_result['model_quality'].title()}")
            
            # coefficient table
            st.markdown("#### Coefficients")
            coef_df = pd.DataFrame(reg_result['coefficients'])
            st.dataframe(
                coef_df[['variable', 'coefficient', 'p_value', 'significant']],
                use_container_width=True,
                hide_index=True
            )
            
            # interpretations
            st.markdown("#### Key Takeaways")
            for insight in generate_regression_insights(reg_result):
                st.markdown(insight)
            
            # regression visualization
            st.markdown("---")
            reg_viz = create_regression_visualization(
                working_df,
                x_var='ball_speed',
                y_var='carry',
                title="Ball Speed vs. Carry Distance"
            )
            st.plotly_chart(reg_viz, use_container_width=True)
        else:
            st.error(reg_result['error'])
    
    else:  # ANOVA
        st.subheader("ANOVA: Comparing Skill Levels")
        st.markdown("""
        ANOVA (Analysis of Variance) tests whether performance differs significantly 
        across skill levels. A low p-value (<0.05) indicates the groups are truly different.
        """)
        
        # metric selection for ANOVA
        anova_metric = st.selectbox(
            "Select metric to compare:",
            ['carry', 'ball_speed', 'spin_rate', 'smash_factor', 'offline']
        )
        
        # run ANOVA across skill segments
        anova_result = run_anova_analysis(segments, anova_metric)
        
        # display results
        anova_col1, anova_col2, anova_col3 = st.columns(3)
        with anova_col1:
            st.metric("F-Statistic", f"{anova_result['f_statistic']:.2f}")
        with anova_col2:
            st.metric("P-Value", anova_result['p_value_formatted'])
        with anova_col3:
            st.metric("Effect Size (η²)", f"{anova_result['eta_squared']:.4f}")
        
        st.markdown(f"**Interpretation**: {anova_result['interpretation']}")
        
        # box plot comparison
        st.markdown("---")
        box_fig = create_skill_level_comparison(segments, anova_metric)
        st.plotly_chart(box_fig, use_container_width=True)


# =============================================================================
# tab 4: skill comparison
# =============================================================================
with tab4:
    st.header("Skill Level Comparison")
    st.markdown("""
    This analysis compares performance across skill levels, revealing how equipment 
    needs differ for elite vs. amateur players.
    """)
    
    # comparison table
    st.subheader("📊 Comparison Table")
    comparison_df = compare_skill_levels(segments)
    
    # format the dataframe for display
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )
    
    # multi-skill dispersion overlay
    st.markdown("---")
    st.subheader("🎯 Dispersion Overlay")
    multi_disp_fig = create_multi_skill_dispersion(segments, max_points=300)
    st.plotly_chart(multi_disp_fig, use_container_width=True)
    
    # consistency comparison
    st.markdown("---")
    st.subheader("📈 Consistency Metrics")
    cons_fig = create_consistency_comparison(segments)
    st.plotly_chart(cons_fig, use_container_width=True)
    
    # insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    for insight in generate_skill_comparison_insights(segments):
        st.markdown(insight)


# =============================================================================
# tab 5: full report
# =============================================================================
with tab5:
    st.header("Full Analysis Report")
    st.markdown("""
    A comprehensive report suitable for player feedback, equipment testing documentation,
    or internal analysis review.
    """)
    
    # generate report
    report = generate_full_report(
        working_df,
        segments=segments,
        player_name="Selected Data"
    )
    
    # display in code block for clean formatting
    st.code(report, language=None)
    
    # download button
    st.download_button(
        label="📥 Download Report (TXT)",
        data=report,
        file_name="golf_performance_report.txt",
        mime="text/plain"
    )
    
    # data export
    st.markdown("---")
    st.subheader("📤 Export Data")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        csv_data = working_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name="filtered_shot_data.csv",
            mime="text/csv"
        )
    
    with export_col2:
        comparison_csv = compare_skill_levels(segments).to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Table (CSV)",
            data=comparison_csv,
            file_name="skill_comparison.csv",
            mime="text/csv"
        )


# =============================================================================
# footer
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <strong>Golf Shot Performance Analyzer</strong> | Built by Eric Bolander<br>
    Demonstrating: Python • Statistical Analysis • Interactive Dashboards • Golf Domain Expertise<br>
    Portfolio project for TaylorMade Performance Research Engineer role
</div>
""", unsafe_allow_html=True)
