"""
visualizations.py - golf shot performance visualization module

this module creates the visualizations that taylormade's performance research
team uses to communicate findings: dispersion plots, comparison charts, and
statistical overlays. all plots are designed to be presentation-ready.

author: eric bolander
project: golf shot performance analyzer (taylormade portfolio project)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import sys
sys.path.insert(0, '/home/claude/golf-shot-analyzer/src')
from analysis import (
    calculate_consistency_metrics,
    calculate_performance_metrics,
    calculate_dispersion_ellipse
)


# =============================================================================
# color scheme: taylormade-inspired palette
# =============================================================================
# using a professional color scheme that's colorblind-friendly
COLORS = {
    'elite': '#1f77b4',      # blue
    'scratch': '#2ca02c',    # green
    'mid': '#ff7f0e',        # orange
    'high': '#d62728',       # red
    'primary': '#1a1a2e',    # dark navy
    'secondary': '#16213e',  # medium navy
    'accent': '#e94560',     # taylormade red accent
    'grid': '#e0e0e0',       # light gray for grid
}


def create_dispersion_plot(
    df: pd.DataFrame,
    title: str = "Shot Dispersion Pattern",
    show_ellipse: bool = True,
    color_by: Optional[str] = None
) -> go.Figure:
    """
    create a bird's-eye view dispersion plot showing where shots land.
    
    this is the most common visualization in equipment fitting because it
    immediately shows a player's accuracy and consistency. the view is from
    above, looking down at the landing area:
    - x-axis: offline distance (negative = left, positive = right)
    - y-axis: carry distance
    
    the dispersion ellipse overlay shows where 68% of shots land (1σ).
    
    args:
        df: dataframe with 'carry' and 'offline' columns
        title: plot title
        show_ellipse: whether to overlay the 68% dispersion ellipse
        color_by: optional column to color points by (e.g., 'spin_rate')
        
    returns:
        plotly figure object
    """
    # create scatter plot
    if color_by and color_by in df.columns:
        fig = px.scatter(
            df,
            x='offline',
            y='carry',
            color=color_by,
            color_continuous_scale='Viridis',
            opacity=0.6,
            title=title
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['offline'],
            y=df['carry'],
            mode='markers',
            marker=dict(
                size=6,
                color=COLORS['primary'],
                opacity=0.5,
            ),
            name='Shots'
        ))
    
    # add dispersion ellipse if requested
    if show_ellipse:
        ellipse = calculate_dispersion_ellipse(df, confidence=0.68)
        
        # generate ellipse points
        theta = np.linspace(0, 2 * np.pi, 100)
        
        # ellipse before rotation
        x_ellipse = ellipse['semi_minor_axis'] * np.cos(theta)
        y_ellipse = ellipse['semi_major_axis'] * np.sin(theta)
        
        # rotate ellipse
        angle_rad = np.radians(ellipse['rotation_angle'])
        x_rotated = x_ellipse * np.cos(angle_rad) - y_ellipse * np.sin(angle_rad)
        y_rotated = x_ellipse * np.sin(angle_rad) + y_ellipse * np.cos(angle_rad)
        
        # translate to center
        x_final = x_rotated + ellipse['center_offline']
        y_final = y_rotated + ellipse['center_carry']
        
        fig.add_trace(go.Scatter(
            x=x_final,
            y=y_final,
            mode='lines',
            line=dict(color=COLORS['accent'], width=2, dash='dash'),
            name=f"68% Dispersion (±{ellipse['offline_std']:.0f}yds)"
        ))
        
        # add center point
        fig.add_trace(go.Scatter(
            x=[ellipse['center_offline']],
            y=[ellipse['center_carry']],
            mode='markers',
            marker=dict(size=12, color=COLORS['accent'], symbol='x'),
            name=f"Center ({ellipse['center_carry']:.0f}yds)"
        ))
    
    # style the plot
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title="Offline (yards) ← Left | Right →",
        yaxis_title="Carry Distance (yards)",
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        height=600,
        width=800,
    )
    
    # set axis ranges to be symmetric around zero for offline
    max_offline = max(abs(df['offline'].min()), abs(df['offline'].max())) * 1.1
    fig.update_xaxes(range=[-max_offline, max_offline])
    
    return fig


def create_skill_level_comparison(
    segments: Dict[str, pd.DataFrame],
    metric: str = 'carry'
) -> go.Figure:
    """
    create box plots comparing a metric across skill levels.
    
    box plots show the distribution of values: median, quartiles, and outliers.
    this is useful for showing how performance varies within and between groups.
    
    args:
        segments: dictionary with skill level dataframes
        metric: column to compare (e.g., 'carry', 'ball_speed', 'spin_rate')
        
    returns:
        plotly figure object
    """
    fig = go.Figure()
    
    skill_order = ['elite', 'scratch', 'mid', 'high']
    labels = {
        'elite': 'Elite<br>(145+ mph)',
        'scratch': 'Scratch<br>(130-145)',
        'mid': 'Mid HCP<br>(110-130)',
        'high': 'High HCP<br>(<110)'
    }
    
    for level in skill_order:
        if level not in segments or len(segments[level]) == 0:
            continue
            
        fig.add_trace(go.Box(
            y=segments[level][metric],
            name=labels[level],
            marker_color=COLORS[level],
            boxmean=True,  # show mean as dashed line
        ))
    
    # metric-specific labeling
    metric_labels = {
        'carry': 'Carry Distance (yards)',
        'ball_speed': 'Ball Speed (mph)',
        'spin_rate': 'Spin Rate (rpm)',
        'launch_angle': 'Launch Angle (degrees)',
        'smash_factor': 'Smash Factor',
        'offline': 'Offline Distance (yards)',
    }
    
    fig.update_layout(
        title=dict(
            text=f"{metric_labels.get(metric, metric)} by Skill Level",
            x=0.5,
            font=dict(size=16)
        ),
        yaxis_title=metric_labels.get(metric, metric),
        xaxis_title="Skill Level (by Ball Speed)",
        template='plotly_white',
        showlegend=False,
        height=500,
        width=700,
    )
    
    return fig


def create_consistency_comparison(
    segments: Dict[str, pd.DataFrame]
) -> go.Figure:
    """
    create grouped bar chart comparing consistency metrics across skill levels.
    
    this visualization answers: "who is more consistent - elite or amateur players?"
    spoiler: elite players are more consistent in everything, but the magnitude
    varies by metric.
    
    args:
        segments: dictionary with skill level dataframes
        
    returns:
        plotly figure object
    """
    # calculate consistency metrics for each skill level
    data = []
    skill_order = ['elite', 'scratch', 'mid', 'high']
    
    for level in skill_order:
        if level not in segments or len(segments[level]) == 0:
            continue
        metrics = calculate_consistency_metrics(segments[level])
        data.append({
            'skill_level': level,
            'Carry CV (%)': metrics.carry_cv,
            'Offline Std (yds)': metrics.offline_std,
            'Spin Std (rpm/10)': metrics.spin_std / 10,  # scale for visualization
            'Smash Factor Std (×100)': metrics.smash_factor_std * 100,
        })
    
    df_metrics = pd.DataFrame(data)
    
    # create grouped bar chart
    fig = go.Figure()
    
    metrics_to_plot = ['Carry CV (%)', 'Offline Std (yds)', 'Spin Std (rpm/10)', 'Smash Factor Std (×100)']
    
    for i, metric in enumerate(metrics_to_plot):
        fig.add_trace(go.Bar(
            name=metric,
            x=df_metrics['skill_level'],
            y=df_metrics[metric],
            text=[f"{v:.1f}" for v in df_metrics[metric]],
            textposition='auto',
        ))
    
    fig.update_layout(
        title=dict(
            text="Consistency Metrics by Skill Level (Lower is Better)",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Skill Level",
        yaxis_title="Metric Value (see legend for units)",
        barmode='group',
        template='plotly_white',
        height=500,
        width=900,
        legend=dict(x=0.7, y=0.98),
    )
    
    return fig


def create_launch_condition_scatter(
    df: pd.DataFrame,
    x_var: str = 'launch_angle',
    y_var: str = 'spin_rate',
    color_var: str = 'carry',
    title: str = "Launch Conditions vs. Carry Distance"
) -> go.Figure:
    """
    create scatter plot showing relationship between launch conditions and outcome.
    
    this type of plot is used during fitting to show:
    - current launch conditions (where you are)
    - optimal launch conditions (where you want to be)
    - how changes in launch affect distance
    
    the color gradient shows the outcome (e.g., carry distance), making it
    easy to see the "hot zone" of optimal launch conditions.
    
    args:
        df: dataframe with shot data
        x_var: column for x-axis (e.g., 'launch_angle')
        y_var: column for y-axis (e.g., 'spin_rate')
        color_var: column for color scale (e.g., 'carry')
        title: plot title
        
    returns:
        plotly figure object
    """
    fig = px.scatter(
        df,
        x=x_var,
        y=y_var,
        color=color_var,
        color_continuous_scale='RdYlGn',  # red=low, green=high
        opacity=0.7,
        title=title,
    )
    
    # axis labels
    axis_labels = {
        'launch_angle': 'Launch Angle (°)',
        'spin_rate': 'Spin Rate (rpm)',
        'ball_speed': 'Ball Speed (mph)',
        'carry': 'Carry Distance (yds)',
        'smash_factor': 'Smash Factor',
    }
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title=axis_labels.get(x_var, x_var),
        yaxis_title=axis_labels.get(y_var, y_var),
        coloraxis_colorbar_title=axis_labels.get(color_var, color_var),
        template='plotly_white',
        height=600,
        width=800,
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    create correlation heatmap for key metrics.
    
    correlation analysis reveals which variables move together:
    - strong positive: as one increases, the other increases
    - strong negative: as one increases, the other decreases
    - near zero: no linear relationship
    
    this helps identify which launch parameters most strongly affect distance.
    
    args:
        df: dataframe with shot data
        
    returns:
        plotly figure object
    """
    # select key metrics for correlation
    cols = ['ball_speed', 'club_speed', 'launch_angle', 'spin_rate', 
            'smash_factor', 'carry', 'total_distance', 'offline']
    
    # filter to columns that exist in dataframe
    cols = [c for c in cols if c in df.columns]
    
    # calculate correlation matrix
    corr_matrix = df[cols].corr()
    
    # create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=cols,
        y=cols,
        colorscale='RdBu_r',
        zmid=0,  # center color scale at zero
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=dict(
            text="Correlation Matrix: Launch Conditions & Outcomes",
            x=0.5,
            font=dict(size=16)
        ),
        template='plotly_white',
        height=600,
        width=700,
    )
    
    return fig


def create_multi_skill_dispersion(
    segments: Dict[str, pd.DataFrame],
    max_points: int = 500
) -> go.Figure:
    """
    create overlay dispersion plot comparing all skill levels.
    
    this plot shows how shot patterns differ across skill levels on the
    same axes, making it easy to see that elite players have:
    - longer carry (higher on y-axis)
    - tighter dispersion (smaller spread)
    
    args:
        segments: dictionary with skill level dataframes
        max_points: max points per skill level to avoid overplotting
        
    returns:
        plotly figure object
    """
    fig = go.Figure()
    
    skill_order = ['high', 'mid', 'scratch', 'elite']  # plot in this order so elite is on top
    labels = {
        'elite': 'Elite (145+ mph)',
        'scratch': 'Scratch (130-145)',
        'mid': 'Mid HCP (110-130)',
        'high': 'High HCP (<110)'
    }
    
    for level in skill_order:
        if level not in segments or len(segments[level]) == 0:
            continue
        
        # sample if too many points
        df = segments[level]
        if len(df) > max_points:
            df = df.sample(n=max_points, random_state=42)
        
        fig.add_trace(go.Scatter(
            x=df['offline'],
            y=df['carry'],
            mode='markers',
            marker=dict(
                size=5,
                color=COLORS[level],
                opacity=0.5,
            ),
            name=labels[level]
        ))
    
    fig.update_layout(
        title=dict(
            text="Shot Dispersion by Skill Level",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Offline (yards) ← Left | Right →",
        yaxis_title="Carry Distance (yards)",
        template='plotly_white',
        legend=dict(x=0.02, y=0.98),
        height=700,
        width=900,
    )
    
    return fig


def create_regression_visualization(
    df: pd.DataFrame,
    x_var: str = 'ball_speed',
    y_var: str = 'carry',
    title: str = "Ball Speed vs. Carry Distance"
) -> go.Figure:
    """
    create scatter plot with regression line overlay.
    
    this visualization shows the linear relationship between two variables
    with the best-fit line, helping players understand:
    - how much more distance they'd gain from increased ball speed
    - the strength of the relationship (how tight points cluster around line)
    
    args:
        df: dataframe with shot data
        x_var: column for x-axis
        y_var: column for y-axis
        title: plot title
        
    returns:
        plotly figure object
    """
    # remove missing values
    plot_df = df[[x_var, y_var]].dropna()
    
    # calculate regression line
    slope, intercept, r_value, p_value, std_err = \
        __import__('scipy.stats', fromlist=['linregress']).linregress(plot_df[x_var], plot_df[y_var])
    
    # generate line points
    x_line = np.array([plot_df[x_var].min(), plot_df[x_var].max()])
    y_line = slope * x_line + intercept
    
    # create figure
    fig = go.Figure()
    
    # scatter points
    fig.add_trace(go.Scatter(
        x=plot_df[x_var],
        y=plot_df[y_var],
        mode='markers',
        marker=dict(size=5, color=COLORS['primary'], opacity=0.4),
        name='Shots'
    ))
    
    # regression line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color=COLORS['accent'], width=3),
        name=f'Fit: R² = {r_value**2:.3f}'
    ))
    
    # axis labels
    axis_labels = {
        'ball_speed': 'Ball Speed (mph)',
        'carry': 'Carry Distance (yds)',
        'launch_angle': 'Launch Angle (°)',
        'spin_rate': 'Spin Rate (rpm)',
    }
    
    # add annotation with equation
    equation = f"y = {slope:.2f}x + {intercept:.1f}"
    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f"{equation}<br>+1 mph = +{slope:.1f} yds",
        showarrow=False,
        font=dict(size=12),
        bgcolor='white',
        bordercolor=COLORS['grid'],
        borderwidth=1,
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title=axis_labels.get(x_var, x_var),
        yaxis_title=axis_labels.get(y_var, y_var),
        template='plotly_white',
        height=500,
        width=700,
        showlegend=True,
        legend=dict(x=0.75, y=0.05),
    )
    
    return fig


def create_performance_dashboard_summary(
    df: pd.DataFrame,
    skill_level: str = 'All'
) -> go.Figure:
    """
    create a multi-panel summary dashboard for a single skill level.
    
    this is the "one-pager" view that summarizes performance:
    - dispersion plot (top left)
    - key metrics table (top right)
    - distribution histograms (bottom)
    
    args:
        df: dataframe with shot data
        skill_level: label for the data
        
    returns:
        plotly figure object
    """
    # calculate metrics
    perf = calculate_performance_metrics(df)
    cons = calculate_consistency_metrics(df)
    
    # create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Shot Dispersion',
            'Key Metrics',
            'Carry Distribution',
            'Ball Speed Distribution'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'table'}],
            [{'type': 'histogram'}, {'type': 'histogram'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # dispersion scatter (row 1, col 1)
    fig.add_trace(
        go.Scatter(
            x=df['offline'],
            y=df['carry'],
            mode='markers',
            marker=dict(size=4, color=COLORS['primary'], opacity=0.5),
            showlegend=False,
        ),
        row=1, col=1
    )
    
    # metrics table (row 1, col 2)
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color=COLORS['secondary'],
                font=dict(color='white', size=11),
                align='left'
            ),
            cells=dict(
                values=[
                    ['Shots', 'Ball Speed', 'Carry', 'Total', 'Spin', 'Launch', 'Smash Factor', 
                     'Carry Std', 'Dispersion'],
                    [f"{len(df):,}", f"{perf.ball_speed_mean:.1f} mph", f"{perf.carry_mean:.1f} yds",
                     f"{perf.total_mean:.1f} yds", f"{perf.spin_rate_mean:.0f} rpm",
                     f"{perf.launch_angle_mean:.1f}°", f"{perf.smash_factor_mean:.3f}",
                     f"±{cons.carry_std:.1f} yds", f"±{cons.offline_std:.1f} yds"]
                ],
                fill_color='white',
                font=dict(size=10),
                align='left',
                height=25,
            )
        ),
        row=1, col=2
    )
    
    # carry histogram (row 2, col 1)
    fig.add_trace(
        go.Histogram(
            x=df['carry'],
            nbinsx=30,
            marker_color=COLORS['elite'],
            showlegend=False,
        ),
        row=2, col=1
    )
    
    # ball speed histogram (row 2, col 2)
    fig.add_trace(
        go.Histogram(
            x=df['ball_speed'],
            nbinsx=30,
            marker_color=COLORS['scratch'],
            showlegend=False,
        ),
        row=2, col=2
    )
    
    # update axes
    fig.update_xaxes(title_text="Offline (yds)", row=1, col=1)
    fig.update_yaxes(title_text="Carry (yds)", row=1, col=1)
    fig.update_xaxes(title_text="Carry (yds)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Ball Speed (mph)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    fig.update_layout(
        title=dict(
            text=f"Performance Dashboard: {skill_level}",
            x=0.5,
            font=dict(size=18)
        ),
        template='plotly_white',
        height=700,
        width=1000,
        showlegend=False,
    )
    
    return fig


# =============================================================================
# main execution: demonstrate visualization capabilities
# =============================================================================
if __name__ == "__main__":
    from data_loader import load_data, clean_data, segment_by_skill, filter_by_ball_type
    
    print("=" * 60)
    print("golf shot performance analyzer - visualization demo")
    print("=" * 60)
    
    # load and clean data
    raw_df = load_data('/mnt/project/data.csv')
    clean_df, _ = clean_data(raw_df)
    
    # segment by skill and filter to premium balls
    segments = segment_by_skill(clean_df)
    for level in segments:
        segments[level] = filter_by_ball_type(segments[level], 'Premium')
    
    # create sample visualizations
    print("\ngenerating visualizations...")
    
    # 1. elite dispersion plot
    elite_dispersion = create_dispersion_plot(
        segments['elite'],
        title="Elite Player Dispersion (Ball Speed ≥145 mph)"
    )
    elite_dispersion.write_html('/home/claude/golf-shot-analyzer/elite_dispersion.html')
    print("  saved: elite_dispersion.html")
    
    # 2. skill level comparison
    skill_comparison = create_skill_level_comparison(segments, 'carry')
    skill_comparison.write_html('/home/claude/golf-shot-analyzer/skill_comparison.html')
    print("  saved: skill_comparison.html")
    
    # 3. multi-skill dispersion overlay
    multi_dispersion = create_multi_skill_dispersion(segments)
    multi_dispersion.write_html('/home/claude/golf-shot-analyzer/multi_dispersion.html')
    print("  saved: multi_dispersion.html")
    
    # 4. correlation heatmap
    corr_heatmap = create_correlation_heatmap(clean_df)
    corr_heatmap.write_html('/home/claude/golf-shot-analyzer/correlation.html')
    print("  saved: correlation.html")
    
    # 5. regression visualization
    regression_plot = create_regression_visualization(
        segments['elite'],
        x_var='ball_speed',
        y_var='carry'
    )
    regression_plot.write_html('/home/claude/golf-shot-analyzer/regression.html')
    print("  saved: regression.html")
    
    # 6. dashboard summary
    dashboard = create_performance_dashboard_summary(segments['elite'], 'Elite')
    dashboard.write_html('/home/claude/golf-shot-analyzer/dashboard.html')
    print("  saved: dashboard.html")
    
    print("\nall visualizations saved!")
