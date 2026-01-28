"""
insights.py - automated insight generation for golf performance data

this module automatically generates human-readable insights from shot data,
mimicking how a taylormade performance research engineer would communicate
findings to players, coaches, and product teams.

the ability to translate complex data into actionable insights is a key
requirement for the performance research engineer role.

author: eric bolander
project: golf shot performance analyzer (taylormade portfolio project)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
sys.path.insert(0, '/home/claude/golf-shot-analyzer/src')
from analysis import (
    calculate_consistency_metrics,
    calculate_performance_metrics,
    run_anova_analysis,
    run_regression_analysis,
    calculate_dispersion_ellipse,
    ConsistencyMetrics,
    PerformanceMetrics
)


# =============================================================================
# benchmark values for contextualizing performance
# =============================================================================
# these benchmarks are based on trackman combine data and pga tour statistics.
# they provide context for whether a player's metrics are "good" or "needs work".

BENCHMARKS = {
    # pga tour driver averages (2024 season)
    'tour': {
        'ball_speed': 171.0,      # mph
        'carry': 275.0,           # yards
        'spin_rate': 2500,        # rpm
        'launch_angle': 10.5,     # degrees
        'smash_factor': 1.49,     # ratio
        'carry_std': 10.0,        # yards (estimated)
        'offline_std': 18.0,      # yards (estimated)
    },
    # scratch golfer benchmarks (~0 handicap)
    'scratch': {
        'ball_speed': 155.0,
        'carry': 250.0,
        'spin_rate': 2800,
        'launch_angle': 12.0,
        'smash_factor': 1.47,
        'carry_std': 15.0,
        'offline_std': 22.0,
    },
    # 15 handicap benchmarks
    'amateur': {
        'ball_speed': 130.0,
        'carry': 200.0,
        'spin_rate': 3200,
        'launch_angle': 14.0,
        'smash_factor': 1.42,
        'carry_std': 25.0,
        'offline_std': 35.0,
    }
}


def generate_performance_insights(
    df: pd.DataFrame,
    player_name: str = "Player"
) -> List[str]:
    """
    generate insights about player performance compared to benchmarks.
    
    this function produces the kind of feedback a taylormade fitter would
    give during a driver fitting session.
    
    args:
        df: cleaned dataframe with shot data
        player_name: name for personalized insights
        
    returns:
        list of insight strings
    """
    insights = []
    perf = calculate_performance_metrics(df)
    cons = calculate_consistency_metrics(df)
    
    # determine player's skill tier based on ball speed
    if perf.ball_speed_mean >= 160:
        tier = 'tour'
        tier_label = 'Tour-level'
    elif perf.ball_speed_mean >= 145:
        tier = 'scratch'
        tier_label = 'Scratch-level'
    else:
        tier = 'amateur'
        tier_label = 'Amateur'
    
    bench = BENCHMARKS[tier]
    
    # ball speed insight
    speed_diff = perf.ball_speed_mean - bench['ball_speed']
    if speed_diff > 5:
        insights.append(
            f"🚀 **Ball Speed**: {perf.ball_speed_mean:.1f} mph is {speed_diff:.1f} mph "
            f"above {tier_label} average ({bench['ball_speed']:.0f} mph). Excellent clubhead speed!"
        )
    elif speed_diff < -5:
        insights.append(
            f"📊 **Ball Speed**: {perf.ball_speed_mean:.1f} mph is {abs(speed_diff):.1f} mph "
            f"below {tier_label} average. Focus on speed training or consider lighter shaft."
        )
    else:
        insights.append(
            f"✅ **Ball Speed**: {perf.ball_speed_mean:.1f} mph is right at {tier_label} average."
        )
    
    # smash factor insight (strike quality)
    smash_diff = perf.smash_factor_mean - bench['smash_factor']
    if smash_diff > 0.02:
        insights.append(
            f"🎯 **Strike Quality**: Smash factor of {perf.smash_factor_mean:.3f} is excellent "
            f"(benchmark: {bench['smash_factor']:.2f}). Consistent center contact."
        )
    elif smash_diff < -0.03:
        insights.append(
            f"⚠️ **Strike Quality**: Smash factor of {perf.smash_factor_mean:.3f} suggests "
            f"off-center hits. Consider impact tape to identify miss pattern, or larger "
            f"head (higher MOI) for more forgiveness."
        )
    else:
        insights.append(
            f"✅ **Strike Quality**: Smash factor of {perf.smash_factor_mean:.3f} indicates solid contact."
        )
    
    # spin rate insight
    if perf.spin_rate_mean < 2200 and perf.ball_speed_mean > 155:
        insights.append(
            f"🏌️ **Spin Rate**: {perf.spin_rate_mean:.0f} rpm is optimally low for your speed. "
            f"This maximizes roll and total distance."
        )
    elif perf.spin_rate_mean > 3500:
        insights.append(
            f"🔄 **Spin Rate**: {perf.spin_rate_mean:.0f} rpm is high. Consider a lower-spin "
            f"shaft or head (e.g., Qi10 LS) to reduce spin by 300-500 rpm and add 8-15 yards."
        )
    else:
        insights.append(
            f"✅ **Spin Rate**: {perf.spin_rate_mean:.0f} rpm is in the optimal range for carry."
        )
    
    # dispersion insight
    if cons.offline_std < bench['offline_std'] - 5:
        insights.append(
            f"🎯 **Dispersion**: ±{cons.offline_std:.1f} yards is tighter than {tier_label} "
            f"average (±{bench['offline_std']:.0f} yds). Excellent accuracy!"
        )
    elif cons.offline_std > bench['offline_std'] + 10:
        insights.append(
            f"📍 **Dispersion**: ±{cons.offline_std:.1f} yards is wider than average. "
            f"Focus on face angle control or consider a draw-biased head."
        )
    else:
        insights.append(
            f"✅ **Dispersion**: ±{cons.offline_std:.1f} yards is typical for your skill level."
        )
    
    # distance consistency insight
    if cons.carry_cv < 8:
        insights.append(
            f"📏 **Distance Consistency**: CV of {cons.carry_cv:.1f}% is very consistent. "
            f"You can confidently commit to carry numbers."
        )
    elif cons.carry_cv > 15:
        insights.append(
            f"📏 **Distance Consistency**: CV of {cons.carry_cv:.1f}% shows variability. "
            f"This may be strike quality (check smash factor) or inconsistent attack angle."
        )
    
    return insights


def generate_skill_comparison_insights(
    segments: Dict[str, pd.DataFrame]
) -> List[str]:
    """
    generate insights comparing performance across skill levels.
    
    this type of analysis is used by taylormade to:
    - understand how equipment should differ for different skill levels
    - set realistic expectations for players
    - inform product line decisions (qi10 vs qi10 max vs qi10 ls)
    
    args:
        segments: dictionary with skill level dataframes
        
    returns:
        list of insight strings
    """
    insights = []
    
    # calculate metrics for each level
    metrics = {}
    for level in ['elite', 'scratch', 'mid', 'high']:
        if level in segments and len(segments[level]) > 0:
            metrics[level] = {
                'perf': calculate_performance_metrics(segments[level]),
                'cons': calculate_consistency_metrics(segments[level]),
                'n': len(segments[level])
            }
    
    if 'elite' in metrics and 'high' in metrics:
        # distance gap
        elite_carry = metrics['elite']['perf'].carry_mean
        high_carry = metrics['high']['perf'].carry_mean
        distance_gap = elite_carry - high_carry
        
        insights.append(
            f"📊 **Distance Gap**: Elite players carry {distance_gap:.0f} yards farther "
            f"than high handicappers ({elite_carry:.0f} vs {high_carry:.0f} yards). "
            f"Ball speed is the primary driver."
        )
        
        # consistency gap
        elite_disp = metrics['elite']['cons'].offline_std
        high_disp = metrics['high']['cons'].offline_std
        
        # note: interesting finding - high handicappers sometimes have tighter dispersion
        # because they hit it shorter (less time for the ball to curve)
        if elite_disp > high_disp:
            insights.append(
                f"🎯 **Dispersion Pattern**: Interestingly, high handicappers show tighter "
                f"dispersion (±{high_disp:.0f} yds) than elite (±{elite_disp:.0f} yds). "
                f"This is because shorter shots have less time to curve offline."
            )
        else:
            insights.append(
                f"🎯 **Dispersion**: Elite players have tighter dispersion "
                f"(±{elite_disp:.0f} yds vs ±{high_disp:.0f} yds)."
            )
        
        # spin pattern
        elite_spin = metrics['elite']['perf'].spin_rate_mean
        high_spin = metrics['high']['perf'].spin_rate_mean
        
        insights.append(
            f"🔄 **Spin Pattern**: High handicappers average {high_spin:.0f} rpm vs "
            f"elite's {elite_spin:.0f} rpm. Higher spin often indicates steeper attack "
            f"angle or higher loft at impact - areas where game-improvement drivers help."
        )
    
    # product recommendation context
    insights.append(
        f"\n💡 **Equipment Insight**: This data explains TaylorMade's product line strategy:\n"
        f"   • **Qi10 LS**: For elite players (low spin, workability)\n"
        f"   • **Qi10**: For scratch/low single-digit (balanced performance)\n"
        f"   • **Qi10 Max**: For mid/high handicaps (forgiveness, draw bias)"
    )
    
    return insights


def generate_regression_insights(
    regression_result: Dict
) -> List[str]:
    """
    translate regression analysis into actionable insights.
    
    regression tells us "what predicts distance" - this function converts
    the statistical output into practical advice.
    
    args:
        regression_result: output from run_regression_analysis()
        
    returns:
        list of insight strings
    """
    insights = []
    
    if 'error' in regression_result:
        return [f"⚠️ Regression analysis failed: {regression_result['error']}"]
    
    # r-squared interpretation
    r2 = regression_result['r_squared']
    insights.append(
        f"📈 **Model Fit**: Launch conditions explain {r2*100:.1f}% of carry distance variation. "
        f"This is {'excellent' if r2 > 0.8 else 'good' if r2 > 0.6 else 'moderate'}."
    )
    
    # coefficient interpretations
    for coef in regression_result['coefficients']:
        if coef['variable'] == 'intercept':
            continue
        
        if not coef['significant']:
            continue
        
        var = coef['variable']
        val = coef['coefficient']
        
        if var == 'ball_speed':
            # ball speed coefficient tells us yards per mph
            insights.append(
                f"🚀 **Ball Speed Impact**: Each +1 mph ball speed adds {val:.1f} yards carry. "
                f"A 5 mph gain = {val*5:.0f} extra yards."
            )
        
        elif var == 'launch_angle':
            if val > 0:
                insights.append(
                    f"📐 **Launch Angle**: Each +1° launch adds {val:.1f} yards "
                    f"(up to optimal ~12° for elite speed)."
                )
            else:
                insights.append(
                    f"📐 **Launch Angle**: Data shows higher launch costs distance "
                    f"({val:.1f} yds per degree). Players may be launching too high."
                )
        
        elif var == 'spin_rate':
            # spin coefficient is usually negative (more spin = less distance)
            yards_per_100rpm = val * 100
            insights.append(
                f"🔄 **Spin Impact**: Each +100 rpm costs {abs(yards_per_100rpm):.1f} yards. "
                f"Reducing spin 500 rpm could add {abs(val*500):.0f} yards."
            )
    
    return insights


def generate_optimal_launch_insights(
    df: pd.DataFrame
) -> List[str]:
    """
    identify and explain optimal launch conditions for the dataset.
    
    this analysis finds the "sweet spot" where players maximize carry
    and provides target numbers for fitting.
    
    args:
        df: cleaned dataframe with shot data
        
    returns:
        list of insight strings
    """
    insights = []
    
    # find shots in top 10% of carry distance
    carry_90th = df['carry'].quantile(0.90)
    top_shots = df[df['carry'] >= carry_90th]
    
    if len(top_shots) < 10:
        return ["⚠️ Not enough data to determine optimal launch conditions."]
    
    avg_launch = top_shots['launch_angle'].mean()
    avg_spin = top_shots['spin_rate'].mean()
    avg_carry = top_shots['carry'].mean()
    avg_speed = top_shots['ball_speed'].mean()
    
    insights.append(
        f"🎯 **Optimal Launch Window** (top 10% of shots):\n"
        f"   • Ball Speed: {avg_speed:.1f} mph\n"
        f"   • Launch Angle: {avg_launch:.1f}°\n"
        f"   • Spin Rate: {avg_spin:.0f} rpm\n"
        f"   • Result: {avg_carry:.0f} yards carry"
    )
    
    # compare to overall average
    all_carry = df['carry'].mean()
    all_launch = df['launch_angle'].mean()
    all_spin = df['spin_rate'].mean()
    
    carry_gain = avg_carry - all_carry
    
    insights.append(
        f"💡 **Potential Gain**: Achieving optimal launch conditions consistently "
        f"would add {carry_gain:.0f} yards vs. current average "
        f"({avg_carry:.0f} vs {all_carry:.0f} yards)."
    )
    
    # fitting recommendation
    if avg_spin < all_spin:
        spin_reduction = all_spin - avg_spin
        insights.append(
            f"🔧 **Fitting Focus**: Top shots have {spin_reduction:.0f} rpm less spin. "
            f"Consider testing a lower-spin shaft or adjusting loft."
        )
    
    if avg_launch > all_launch:
        insights.append(
            f"🔧 **Launch Optimization**: Top shots launch {avg_launch - all_launch:.1f}° "
            f"higher. May benefit from positive attack angle adjustment."
        )
    
    return insights


def generate_full_report(
    df: pd.DataFrame,
    segments: Optional[Dict[str, pd.DataFrame]] = None,
    player_name: str = "Player"
) -> str:
    """
    generate a comprehensive analysis report.
    
    this produces a full report suitable for:
    - player feedback after a fitting session
    - internal taylormade performance review
    - portfolio demonstration
    
    args:
        df: cleaned dataframe with shot data
        segments: optional skill-segmented data for comparison
        player_name: name for personalization
        
    returns:
        formatted report string
    """
    report = []
    
    # header
    report.append("=" * 70)
    report.append("GOLF SHOT PERFORMANCE ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"\nData: {len(df):,} shots analyzed")
    report.append("-" * 70)
    
    # performance insights
    report.append("\n## PERFORMANCE ANALYSIS\n")
    for insight in generate_performance_insights(df, player_name):
        report.append(insight)
    
    # optimal launch
    report.append("\n" + "-" * 70)
    report.append("\n## OPTIMAL LAUNCH CONDITIONS\n")
    for insight in generate_optimal_launch_insights(df):
        report.append(insight)
    
    # regression analysis
    report.append("\n" + "-" * 70)
    report.append("\n## PREDICTIVE MODEL\n")
    reg_result = run_regression_analysis(df, 'carry', ['ball_speed', 'launch_angle', 'spin_rate'])
    for insight in generate_regression_insights(reg_result):
        report.append(insight)
    
    # skill comparison (if provided)
    if segments:
        report.append("\n" + "-" * 70)
        report.append("\n## SKILL LEVEL COMPARISON\n")
        for insight in generate_skill_comparison_insights(segments):
            report.append(insight)
    
    # footer
    report.append("\n" + "=" * 70)
    report.append("Generated by Golf Shot Performance Analyzer")
    report.append("Author: Eric Bolander | Portfolio Project")
    report.append("=" * 70)
    
    return "\n".join(report)


# =============================================================================
# main execution: demonstrate insight generation
# =============================================================================
if __name__ == "__main__":
    from data_loader import load_data, clean_data, segment_by_skill, filter_by_ball_type
    
    print("=" * 60)
    print("golf shot performance analyzer - insights demo")
    print("=" * 60)
    
    # load and clean data
    raw_df = load_data('/mnt/project/data.csv')
    clean_df, _ = clean_data(raw_df)
    
    # segment by skill
    segments = segment_by_skill(clean_df)
    for level in segments:
        segments[level] = filter_by_ball_type(segments[level], 'Premium')
    
    # generate full report for elite segment
    report = generate_full_report(
        segments['elite'],
        segments=segments,
        player_name="Elite Golfer"
    )
    
    print(report)
