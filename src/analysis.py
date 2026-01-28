"""
analysis.py - statistical analysis engine for golf shot performance

this module implements the core analytics that a taylormade performance research
engineer would use: consistency metrics, comparative statistics (anova), and
predictive modeling (regression). each function is documented with the golf
context explaining why these metrics matter for equipment testing.

author: eric bolander
project: golf shot performance analyzer (taylormade portfolio project)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConsistencyMetrics:
    """
    container for shot consistency metrics.
    
    consistency is often more important than peak performance in golf equipment
    testing. a driver that produces 260 yards ± 8 yards is more valuable than
    one that produces 265 yards ± 20 yards for most players.
    """
    carry_std: float          # yards - distance consistency
    carry_cv: float           # coefficient of variation (std/mean * 100)
    offline_std: float        # yards - dispersion (accuracy)
    spin_std: float           # rpm - spin consistency
    launch_angle_std: float   # degrees - launch consistency
    smash_factor_std: float   # strike quality consistency
    ball_speed_cv: float      # cv for ball speed (clubhead delivery)


@dataclass
class PerformanceMetrics:
    """
    container for average performance metrics.
    
    these represent "what you can expect" from a player or equipment setup.
    taylormade uses these to characterize equipment across skill levels.
    """
    ball_speed_mean: float    # mph - primary power metric
    carry_mean: float         # yards - primary distance metric
    total_mean: float         # yards - carry + roll
    spin_rate_mean: float     # rpm - affects trajectory and roll
    launch_angle_mean: float  # degrees - initial ball flight
    smash_factor_mean: float  # efficiency metric
    apex_height_mean: float   # feet - trajectory peak


def calculate_consistency_metrics(df: pd.DataFrame) -> ConsistencyMetrics:
    """
    calculate consistency metrics for a set of shots.
    
    consistency metrics tell us how repeatable a player's performance is,
    which is critical for:
    1. fitting equipment (consistent players can use tighter specs)
    2. comparing equipment (did driver a produce tighter dispersion?)
    3. tracking improvement (is the player getting more consistent?)
    
    args:
        df: cleaned dataframe with shot data
        
    returns:
        ConsistencyMetrics dataclass with all calculated values
    """
    # carry distance standard deviation
    # this is the primary "distance consistency" metric
    # tour players typically have 8-12 yard std, amateurs 15-25 yards
    carry_std = df['carry'].std()
    
    # coefficient of variation for carry
    # cv normalizes std by the mean, allowing fair comparison across skill levels
    # a 10-yard std on 280 yards (3.6% cv) is more consistent than
    # 10-yard std on 200 yards (5% cv)
    carry_cv = (df['carry'].std() / df['carry'].mean()) * 100
    
    # offline standard deviation (dispersion)
    # this is the "accuracy" metric - how far left/right of target
    # smaller is better. elite players: 15-20 yds, amateurs: 25-40 yds
    # this value represents 1 standard deviation, so ~68% of shots fall within ± this value
    offline_std = df['offline'].std()
    
    # spin rate standard deviation
    # spin consistency affects trajectory predictability
    # important for ball fitting - some balls produce more consistent spin
    # typical range: 300-600 rpm std for consistent players
    spin_std = df['spin_rate'].std()
    
    # launch angle standard deviation
    # measures how repeatable the impact conditions are
    # tight launch = consistent low point and attack angle
    # good players: 1.5-2.5 degrees, amateurs: 3-5 degrees
    launch_angle_std = df['launch_angle'].std()
    
    # smash factor standard deviation
    # measures strike quality consistency (center vs off-center hits)
    # tour players: 0.02-0.03, amateurs: 0.05-0.10
    smash_factor_std = df['smash_factor'].std()
    
    # ball speed cv
    # combines club speed consistency with strike quality
    # lower is better - indicates repeatable power delivery
    ball_speed_cv = (df['ball_speed'].std() / df['ball_speed'].mean()) * 100
    
    return ConsistencyMetrics(
        carry_std=carry_std,
        carry_cv=carry_cv,
        offline_std=offline_std,
        spin_std=spin_std,
        launch_angle_std=launch_angle_std,
        smash_factor_std=smash_factor_std,
        ball_speed_cv=ball_speed_cv
    )


def calculate_performance_metrics(df: pd.DataFrame) -> PerformanceMetrics:
    """
    calculate average performance metrics for a set of shots.
    
    these metrics characterize "typical" output for a player or equipment setup.
    
    args:
        df: cleaned dataframe with shot data
        
    returns:
        PerformanceMetrics dataclass with all calculated values
    """
    return PerformanceMetrics(
        ball_speed_mean=df['ball_speed'].mean(),
        carry_mean=df['carry'].mean(),
        total_mean=df['total_distance'].mean(),
        spin_rate_mean=df['spin_rate'].mean(),
        launch_angle_mean=df['launch_angle'].mean(),
        smash_factor_mean=df['smash_factor'].mean(),
        apex_height_mean=df['apex_height'].mean() if 'apex_height' in df.columns else np.nan
    )


def compare_skill_levels(segments: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    generate comparative statistics across skill level segments.
    
    this analysis reveals how performance and consistency differ by skill level,
    which is essential for taylormade's product line strategy:
    - qi10 max: designed for mid/high handicaps (higher moi, more forgiveness)
    - qi10 ls: designed for elite players (lower spin, more workability)
    
    args:
        segments: dictionary from segment_by_skill() with 'elite', 'scratch', 'mid', 'high' keys
        
    returns:
        dataframe with metrics compared across skill levels
    """
    comparison_data = []
    
    for level in ['elite', 'scratch', 'mid', 'high']:
        if level not in segments or len(segments[level]) == 0:
            continue
            
        df = segments[level]
        consistency = calculate_consistency_metrics(df)
        performance = calculate_performance_metrics(df)
        
        comparison_data.append({
            'skill_level': level,
            'shot_count': len(df),
            # performance metrics
            'ball_speed_mean': round(performance.ball_speed_mean, 1),
            'carry_mean': round(performance.carry_mean, 1),
            'spin_rate_mean': round(performance.spin_rate_mean, 0),
            'launch_angle_mean': round(performance.launch_angle_mean, 1),
            'smash_factor_mean': round(performance.smash_factor_mean, 3),
            # consistency metrics
            'carry_std': round(consistency.carry_std, 1),
            'carry_cv': round(consistency.carry_cv, 2),
            'offline_std': round(consistency.offline_std, 1),
            'spin_std': round(consistency.spin_std, 0),
            'smash_factor_std': round(consistency.smash_factor_std, 3),
        })
    
    return pd.DataFrame(comparison_data)


def run_anova_analysis(
    segments: Dict[str, pd.DataFrame],
    metric: str = 'carry'
) -> Dict:
    """
    perform one-way anova to test if metric differs significantly across skill levels.
    
    anova (analysis of variance) is a key statistical tool for equipment testing.
    it answers questions like:
    - "does carry distance differ significantly between skill levels?"
    - "is the difference in spin rate between ball types statistically significant?"
    
    for taylormade, this is used to validate that equipment differences are real
    and not just random variation.
    
    args:
        segments: dictionary from segment_by_skill()
        metric: column name to compare (e.g., 'carry', 'spin_rate', 'offline')
        
    returns:
        dictionary with f-statistic, p-value, and interpretation
    """
    # extract the metric from each skill level
    groups = []
    group_labels = []
    
    for level in ['elite', 'scratch', 'mid', 'high']:
        if level in segments and len(segments[level]) > 0:
            groups.append(segments[level][metric].dropna().values)
            group_labels.append(level)
    
    # perform one-way anova
    f_statistic, p_value = stats.f_oneway(*groups)
    
    # calculate effect size (eta-squared)
    # eta-squared tells us what proportion of variance is explained by group membership
    # small: 0.01, medium: 0.06, large: 0.14
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # determine significance and practical interpretation
    if p_value < 0.001:
        significance = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        significance = "very significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    
    if eta_squared >= 0.14:
        effect = "large effect"
    elif eta_squared >= 0.06:
        effect = "medium effect"
    elif eta_squared >= 0.01:
        effect = "small effect"
    else:
        effect = "negligible effect"
    
    return {
        'metric': metric,
        'groups': group_labels,
        'f_statistic': round(f_statistic, 2),
        'p_value': p_value,
        'p_value_formatted': f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
        'eta_squared': round(eta_squared, 4),
        'significance': significance,
        'effect_size': effect,
        'interpretation': f"{metric} differs across skill levels with {significance} ({effect})"
    }


def run_regression_analysis(
    df: pd.DataFrame,
    target: str = 'carry',
    predictors: List[str] = None
) -> Dict:
    """
    perform multiple linear regression to predict target variable.
    
    regression analysis reveals which launch conditions most strongly predict
    performance outcomes. for taylormade, this informs:
    - which parameters to optimize during fitting
    - how much carry gain to expect from specific changes
    - which metrics explain the most variance in outcomes
    
    common use cases:
    - predict carry from ball_speed, launch_angle, spin_rate
    - predict total_distance from carry, spin_rate, land_angle
    
    args:
        df: cleaned dataframe with shot data
        target: column name to predict (e.g., 'carry')
        predictors: list of column names as predictors. defaults to
                   ['ball_speed', 'launch_angle', 'spin_rate']
                   
    returns:
        dictionary with regression results including coefficients and r-squared
    """
    if predictors is None:
        # default predictors for carry distance
        # these are the primary launch conditions that determine ball flight
        predictors = ['ball_speed', 'launch_angle', 'spin_rate']
    
    # prepare data, dropping rows with missing values in relevant columns
    cols_needed = [target] + predictors
    analysis_df = df[cols_needed].dropna()
    
    if len(analysis_df) < 30:
        return {'error': 'insufficient data for regression (need at least 30 rows)'}
    
    # extract variables
    y = analysis_df[target].values
    X = analysis_df[predictors].values
    
    # add intercept (constant term)
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # calculate regression coefficients using ordinary least squares
    # β = (X'X)^-1 X'y
    try:
        coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {'error': 'regression calculation failed (singular matrix)'}
    
    # calculate predictions and residuals
    y_pred = X_with_intercept @ coefficients
    residuals = y - y_pred
    
    # calculate r-squared (coefficient of determination)
    # r² tells us what proportion of variance in the target is explained by predictors
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # calculate adjusted r-squared (penalizes adding more predictors)
    n = len(y)
    p = len(predictors)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    
    # calculate standard errors and t-statistics for coefficients
    mse = ss_res / (n - p - 1)
    var_coef = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept).diagonal()
    se_coef = np.sqrt(var_coef)
    t_stats = coefficients / se_coef
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    # build coefficient table
    coef_table = []
    coef_names = ['intercept'] + predictors
    for i, name in enumerate(coef_names):
        coef_table.append({
            'variable': name,
            'coefficient': round(coefficients[i], 4),
            'std_error': round(se_coef[i], 4),
            't_statistic': round(t_stats[i], 2),
            'p_value': round(p_values[i], 4),
            'significant': p_values[i] < 0.05
        })
    
    # generate interpretation
    # for carry distance prediction from ball speed, launch angle, spin rate:
    # - ball_speed coefficient ~2.0 means +1 mph ball speed = +2 yards carry
    # - launch_angle coefficient ~1.5 means +1 degree launch = +1.5 yards (up to optimal)
    # - spin_rate coefficient ~ -0.01 means +100 rpm = -1 yard (high spin costs distance)
    
    interpretations = []
    for row in coef_table:
        if row['variable'] == 'intercept':
            continue
        if row['significant']:
            direction = "increases" if row['coefficient'] > 0 else "decreases"
            interpretations.append(
                f"{row['variable']}: +1 unit {direction} {target} by {abs(row['coefficient']):.2f}"
            )
    
    return {
        'target': target,
        'predictors': predictors,
        'n_observations': n,
        'r_squared': round(r_squared, 4),
        'adj_r_squared': round(adj_r_squared, 4),
        'coefficients': coef_table,
        'interpretations': interpretations,
        'model_quality': 'excellent' if r_squared > 0.9 else 'good' if r_squared > 0.7 else 'moderate' if r_squared > 0.5 else 'weak'
    }


def calculate_dispersion_ellipse(
    df: pd.DataFrame,
    confidence: float = 0.68
) -> Dict:
    """
    calculate dispersion ellipse parameters for shot pattern visualization.
    
    the dispersion ellipse shows where a player's shots land, which is critical
    for equipment fitting and course management. the ellipse captures:
    - center point (where shots tend to cluster)
    - spread in carry direction (distance consistency)
    - spread in lateral direction (accuracy)
    - orientation (does the pattern tilt left or right?)
    
    a 68% ellipse (1 standard deviation) means ~68% of shots fall inside.
    this is the standard used in trackman fitting sessions.
    
    args:
        df: cleaned dataframe with 'carry' and 'offline' columns
        confidence: proportion of shots to capture (0.68 = 1σ, 0.95 = 2σ)
        
    returns:
        dictionary with ellipse parameters for plotting
    """
    # extract carry and offline data
    carry = df['carry'].values
    offline = df['offline'].values
    
    # calculate center (mean position)
    center_carry = np.mean(carry)
    center_offline = np.mean(offline)
    
    # calculate covariance matrix
    # this captures how carry and offline vary together
    cov_matrix = np.cov(carry, offline)
    
    # eigendecomposition to get ellipse axes
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # sort by eigenvalue (largest first)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # calculate ellipse parameters
    # chi-squared value for the confidence level with 2 degrees of freedom
    chi2_val = stats.chi2.ppf(confidence, df=2)
    
    # semi-axes lengths (scaled by chi-squared for confidence level)
    semi_major = np.sqrt(eigenvalues[0] * chi2_val)
    semi_minor = np.sqrt(eigenvalues[1] * chi2_val)
    
    # rotation angle (in degrees)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    return {
        'center_carry': round(center_carry, 1),
        'center_offline': round(center_offline, 1),
        'semi_major_axis': round(semi_major, 1),
        'semi_minor_axis': round(semi_minor, 1),
        'rotation_angle': round(angle, 1),
        'confidence_level': confidence,
        'carry_std': round(np.std(carry), 1),
        'offline_std': round(np.std(offline), 1),
    }


def identify_optimal_launch_conditions(df: pd.DataFrame) -> Dict:
    """
    identify launch conditions that produce optimal carry distance.
    
    this analysis finds the "sweet spot" for launch angle and spin rate
    at a given ball speed. taylormade uses this to:
    - set target specs during fitting
    - design equipment that promotes optimal launch
    - show players what they should aim for
    
    the optimal conditions vary by ball speed:
    - higher ball speed (elite): lower launch, lower spin (12°, 2200 rpm)
    - lower ball speed (amateur): higher launch, more spin (14°, 2800 rpm)
    
    args:
        df: cleaned dataframe with shot data
        
    returns:
        dictionary with optimal launch conditions for max carry
    """
    # bin shots by ball speed to find optimal conditions at each speed
    df = df.copy()
    
    # create ball speed bins (5 mph increments)
    df['speed_bin'] = pd.cut(
        df['ball_speed'],
        bins=range(100, 180, 5),
        labels=[f"{i}-{i+5}" for i in range(100, 175, 5)]
    )
    
    # for each speed bin, find the shots in the top 10% of carry distance
    optimal_conditions = []
    
    for speed_bin in df['speed_bin'].dropna().unique():
        bin_data = df[df['speed_bin'] == speed_bin]
        
        if len(bin_data) < 20:  # need enough shots for reliable analysis
            continue
        
        # top 10% of carry distances in this speed bin
        carry_threshold = bin_data['carry'].quantile(0.90)
        top_shots = bin_data[bin_data['carry'] >= carry_threshold]
        
        optimal_conditions.append({
            'speed_range': speed_bin,
            'n_shots': len(bin_data),
            'n_top_shots': len(top_shots),
            'avg_ball_speed': round(bin_data['ball_speed'].mean(), 1),
            'optimal_launch': round(top_shots['launch_angle'].mean(), 1),
            'optimal_spin': round(top_shots['spin_rate'].mean(), 0),
            'optimal_carry': round(top_shots['carry'].mean(), 1),
            'avg_carry': round(bin_data['carry'].mean(), 1),
            'carry_gain': round(top_shots['carry'].mean() - bin_data['carry'].mean(), 1)
        })
    
    return {
        'by_speed_bin': optimal_conditions,
        'insight': "shots in the top 10% of carry distance at each ball speed"
    }


def generate_performance_summary(
    df: pd.DataFrame,
    skill_level: str = 'all'
) -> str:
    """
    generate a human-readable performance summary.
    
    this produces the kind of summary a taylormade fitter would discuss
    with a player after a testing session.
    
    args:
        df: cleaned dataframe with shot data
        skill_level: label for the data being summarized
        
    returns:
        formatted string summary
    """
    if len(df) == 0:
        return "no data available for analysis"
    
    perf = calculate_performance_metrics(df)
    cons = calculate_consistency_metrics(df)
    
    summary = f"""
{'='*60}
performance summary: {skill_level.upper()}
{'='*60}
shots analyzed: {len(df):,}

performance averages:
  ball speed:     {perf.ball_speed_mean:.1f} mph
  carry distance: {perf.carry_mean:.1f} yds
  total distance: {perf.total_mean:.1f} yds
  launch angle:   {perf.launch_angle_mean:.1f}°
  spin rate:      {perf.spin_rate_mean:.0f} rpm
  smash factor:   {perf.smash_factor_mean:.3f}

consistency metrics:
  carry std dev:    {cons.carry_std:.1f} yds (68% of shots within ±{cons.carry_std:.1f} yds)
  carry cv:         {cons.carry_cv:.1f}% (lower is more consistent)
  dispersion (1σ):  {cons.offline_std:.1f} yds left/right
  spin std dev:     {cons.spin_std:.0f} rpm
  smash factor std: {cons.smash_factor_std:.3f}

insights:
"""
    
    # add contextual insights based on the data
    if cons.offline_std < 20:
        summary += "  • excellent accuracy - tight dispersion pattern\n"
    elif cons.offline_std < 30:
        summary += "  • good accuracy - moderate dispersion\n"
    else:
        summary += "  • dispersion is wide - focus on face control\n"
    
    if perf.smash_factor_mean > 1.48:
        summary += "  • elite strike quality (smash factor > 1.48)\n"
    elif perf.smash_factor_mean > 1.44:
        summary += "  • good strike quality (smash factor 1.44-1.48)\n"
    else:
        summary += "  • strike quality has room for improvement\n"
    
    if perf.spin_rate_mean < 2500 and perf.ball_speed_mean > 150:
        summary += "  • low spin for speed - optimized for distance\n"
    elif perf.spin_rate_mean > 3500:
        summary += "  • high spin - may benefit from lower-spin shaft or head\n"
    
    return summary


# =============================================================================
# main execution: demonstrate all analysis capabilities
# =============================================================================
if __name__ == "__main__":
    # import data loader
    import sys
    sys.path.insert(0, '/home/claude/golf-shot-analyzer/src')
    from data_loader import load_data, clean_data, segment_by_skill, filter_by_ball_type
    
    print("=" * 60)
    print("golf shot performance analyzer - analysis demo")
    print("=" * 60)
    
    # load and clean data
    raw_df = load_data('/mnt/project/data.csv')
    clean_df, _ = clean_data(raw_df)
    
    # segment by skill
    segments = segment_by_skill(clean_df)
    
    # filter to premium balls for consistent comparison
    for level in segments:
        segments[level] = filter_by_ball_type(segments[level], 'Premium')
    
    # compare across skill levels
    print("\n" + "=" * 60)
    print("skill level comparison (premium balls only)")
    print("=" * 60)
    comparison = compare_skill_levels(segments)
    print(comparison.to_string(index=False))
    
    # anova analysis
    print("\n" + "=" * 60)
    print("anova: carry distance by skill level")
    print("=" * 60)
    anova_result = run_anova_analysis(segments, 'carry')
    print(f"f-statistic: {anova_result['f_statistic']}")
    print(f"p-value: {anova_result['p_value_formatted']}")
    print(f"effect size (η²): {anova_result['eta_squared']}")
    print(f"interpretation: {anova_result['interpretation']}")
    
    # regression analysis
    print("\n" + "=" * 60)
    print("regression: predicting carry from launch conditions")
    print("=" * 60)
    elite_df = segments['elite']
    reg_result = run_regression_analysis(elite_df, 'carry', ['ball_speed', 'launch_angle', 'spin_rate'])
    print(f"r-squared: {reg_result['r_squared']} ({reg_result['model_quality']} fit)")
    print(f"observations: {reg_result['n_observations']}")
    print("\ncoefficients:")
    for coef in reg_result['coefficients']:
        sig = "*" if coef['significant'] else ""
        print(f"  {coef['variable']}: {coef['coefficient']:.4f} {sig}")
    print("\ninterpretations:")
    for interp in reg_result['interpretations']:
        print(f"  • {interp}")
    
    # generate summary for elite players
    print(generate_performance_summary(elite_df, 'elite'))
