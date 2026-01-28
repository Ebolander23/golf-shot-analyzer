"""
data_loader.py - trackman data ingestion and cleaning module

this module handles loading, cleaning, and filtering trackman launch monitor data
for golf shot performance analysis. every filter and threshold is documented with
golf-specific reasoning to demonstrate domain expertise.

author: eric bolander
project: golf shot performance analyzer (taylormade portfolio project)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# configuration: data quality thresholds
# =============================================================================
# these thresholds are based on physical limitations of golf ball flight and
# trackman measurement capabilities. each value is documented with reasoning.

THRESHOLDS = {
    # smash factor is ball speed / club speed. the theoretical maximum for a
    # driver is ~1.50 due to coefficient of restitution (COR) limits set by
    # the usga. values above 1.55 indicate measurement error (e.g., club speed
    # misread or ball speed inflated by wind). we use 1.55 to allow for minor
    # measurement variance while catching clear errors.
    'smash_factor_max': 1.55,
    
    # minimum smash factor for a "real" golf swing. anything below 0.8 suggests
    # the club barely contacted the ball (extreme shank, top, or whiff tracked
    # incorrectly). tour average is ~1.48-1.50, amateur average ~1.44-1.48.
    'smash_factor_min': 0.8,
    
    # spin rate bounds for driver shots:
    # - minimum: 500 rpm is extremely low but possible with delofted impact
    # - maximum: 10,000 rpm is a severe pop-up/sky ball. normal driver range
    #   is 2000-3500 rpm for most players. values above 10k often indicate
    #   corrupted data or mis-labeled club (wedge tracked as driver).
    # note: we saw one negative spin value in the data (-21 billion) which is
    # clearly a 32-bit integer overflow from trackman firmware.
    'spin_rate_min': 0,
    'spin_rate_max': 10000,
    
    # ball speed minimum: 80 mph represents approximately 55-60 mph club speed.
    # below this threshold, the shot is either:
    # - a severe mishit (chunk, top, shank)
    # - incorrect club tracking (putter or wedge)
    # - warm-up swing not intended as a real shot
    # we flag these rather than remove them entirely so users can optionally
    # analyze mishit patterns.
    'ball_speed_min': 80,
    
    # launch angle bounds for driver:
    # - negative launch is rare but possible with severe deloft (hitting down
    #   on a teed ball). below -5 degrees is almost certainly measurement error.
    # - maximum of 30 degrees catches pop-ups while allowing high-launch setups
    #   (some senior players legitimately launch 18-22 degrees for max carry).
    'launch_angle_min': -5,
    'launch_angle_max': 30,
}


# =============================================================================
# skill level segmentation thresholds
# =============================================================================
# ball speed is the most reliable indicator of player skill level because it
# directly correlates with club head speed and strike quality. these thresholds
# are based on trackman combine data and pga tour statistics.

SKILL_LEVELS = {
    # elite: 145+ mph ball speed corresponds to ~105+ mph club speed
    # this is pga tour average territory (tour avg: 167 mph ball speed)
    # approximately +2 to scratch handicap range
    'elite': {'ball_speed_min': 145, 'label': 'Elite (145+ mph)'},
    
    # scratch: 130-145 mph ball speed corresponds to ~95-105 mph club speed
    # this covers low single-digit handicaps through scratch players
    # typical "good club player" range
    'scratch': {'ball_speed_min': 130, 'ball_speed_max': 145, 'label': 'Scratch (130-145 mph)'},
    
    # mid: 110-130 mph ball speed corresponds to ~80-95 mph club speed
    # covers roughly 8-18 handicap range
    # largest segment of "regular" golfers
    'mid': {'ball_speed_min': 110, 'ball_speed_max': 130, 'label': 'Mid Handicap (110-130 mph)'},
    
    # high: below 110 mph ball speed, under ~80 mph club speed
    # includes beginners, seniors with slower swing speeds, and juniors
    # important segment for game-improvement equipment design
    'high': {'ball_speed_max': 110, 'label': 'High Handicap (<110 mph)'},
}


# =============================================================================
# column mapping: trackman export to standardized names
# =============================================================================
# trackman exports use verbose column names with units. we standardize to
# shorter, code-friendly names while preserving original meaning.

COLUMN_MAPPING = {
    'Club': 'club',
    'Ball': 'ball_type',
    'Club Speed': 'club_speed',          # mph
    'Attack Angle': 'attack_angle',      # degrees, negative = down
    'Club Path': 'club_path',            # degrees, positive = in-to-out
    'Swing Plane': 'swing_plane',        # degrees
    'Swing Direction': 'swing_direction', # degrees
    'Dyn. Loft': 'dynamic_loft',         # degrees at impact
    'Face Angle': 'face_angle',          # degrees, positive = open
    'Face To Path': 'face_to_path',      # degrees, determines curve
    'Ball Speed': 'ball_speed',          # mph
    'Smash Factor': 'smash_factor',      # ratio (ball_speed / club_speed)
    'Launch Angle': 'launch_angle',      # degrees
    'Launch Direction': 'launch_direction', # degrees, positive = right
    'Spin Rate': 'spin_rate',            # rpm
    'Spin Axis': 'spin_axis',            # degrees, positive = fade/slice
    'Max Height - Height': 'apex_height', # feet
    'Carry Flat - Length': 'carry',      # yards
    'Carry Flat - Side': 'offline',      # yards, positive = right
    'Carry Flat - Land. Angle': 'land_angle', # degrees
    'Total': 'total_distance',           # yards (carry + roll)
    'Side Total': 'side_total',          # yards at landing
    'Curve': 'curve',                    # feet of curvature
}


def load_data(filepath: str) -> pd.DataFrame:
    """
    load trackman csv export and return raw dataframe.
    
    trackman exports include a units row (row index 1) that we skip.
    all numeric columns are converted from string to float.
    
    args:
        filepath: path to trackman csv export
        
    returns:
        pandas dataframe with raw data, columns renamed to standardized names
    """
    # skip row 1 which contains units (e.g., "[mph]", "[deg]", "[yds]")
    # trackman exports have: row 0 = headers, row 1 = units, row 2+ = data
    df = pd.read_csv(filepath, skiprows=[1])
    
    # rename columns to standardized names for cleaner code
    df = df.rename(columns=COLUMN_MAPPING)
    
    # convert numeric columns from string to float
    # trackman sometimes exports numbers as strings with extra formatting
    numeric_columns = [
        'club_speed', 'ball_speed', 'attack_angle', 'club_path',
        'swing_plane', 'swing_direction', 'dynamic_loft', 'face_angle',
        'face_to_path', 'smash_factor', 'launch_angle', 'launch_direction',
        'spin_rate', 'spin_axis', 'apex_height', 'carry', 'offline',
        'land_angle', 'total_distance', 'side_total', 'curve'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def clean_data(df: pd.DataFrame, remove_invalid: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    clean trackman data by flagging and optionally removing invalid shots.
    
    this function applies physics-based filters to identify:
    1. measurement errors (impossible values)
    2. severe mishits (flagged but optionally kept for analysis)
    3. corrupted data (integer overflow, missing critical values)
    
    args:
        df: raw dataframe from load_data()
        remove_invalid: if true, return only valid shots. if false, add
                       'is_valid' column but keep all rows.
                       
    returns:
        tuple of (clean_df, removed_df) where:
        - clean_df: shots passing all quality filters
        - removed_df: shots that failed filters (with 'removal_reason' column)
    """
    df = df.copy()
    
    # track removal reasons for each shot
    df['removal_reasons'] = ''
    
    # ---------------------------------------------------------------------
    # filter 1: smash factor bounds (physics-based)
    # ---------------------------------------------------------------------
    # smash factor > 1.55 violates usga cor limits (measurement error)
    # smash factor < 0.8 indicates severe mis-contact or tracking error
    invalid_smash_high = df['smash_factor'] > THRESHOLDS['smash_factor_max']
    invalid_smash_low = df['smash_factor'] < THRESHOLDS['smash_factor_min']
    
    df.loc[invalid_smash_high, 'removal_reasons'] += 'smash_factor_too_high; '
    df.loc[invalid_smash_low, 'removal_reasons'] += 'smash_factor_too_low; '
    
    # ---------------------------------------------------------------------
    # filter 2: spin rate bounds
    # ---------------------------------------------------------------------
    # negative spin indicates integer overflow (trackman firmware bug)
    # spin > 10000 rpm on driver is severe pop-up or mis-labeled club
    invalid_spin_low = df['spin_rate'] < THRESHOLDS['spin_rate_min']
    invalid_spin_high = df['spin_rate'] > THRESHOLDS['spin_rate_max']
    
    df.loc[invalid_spin_low, 'removal_reasons'] += 'spin_rate_negative; '
    df.loc[invalid_spin_high, 'removal_reasons'] += 'spin_rate_extreme; '
    
    # ---------------------------------------------------------------------
    # filter 3: launch angle bounds
    # ---------------------------------------------------------------------
    # launch < -5 degrees on driver is almost certainly measurement error
    # launch > 30 degrees is a pop-up (flag but may want to keep for analysis)
    invalid_launch_low = df['launch_angle'] < THRESHOLDS['launch_angle_min']
    invalid_launch_high = df['launch_angle'] > THRESHOLDS['launch_angle_max']
    
    df.loc[invalid_launch_low, 'removal_reasons'] += 'launch_angle_negative; '
    df.loc[invalid_launch_high, 'removal_reasons'] += 'launch_angle_popup; '
    
    # ---------------------------------------------------------------------
    # filter 4: missing critical values
    # ---------------------------------------------------------------------
    # ball_speed is required for all analysis - if missing, shot is unusable
    # carry distance is our primary outcome variable
    missing_ball_speed = df['ball_speed'].isna()
    missing_carry = df['carry'].isna()
    
    df.loc[missing_ball_speed, 'removal_reasons'] += 'missing_ball_speed; '
    df.loc[missing_carry, 'removal_reasons'] += 'missing_carry; '
    
    # ---------------------------------------------------------------------
    # flag: low ball speed (mishit indicator)
    # ---------------------------------------------------------------------
    # we don't remove these by default - they may be useful for mishit analysis
    # but we flag them for easy filtering
    low_ball_speed = df['ball_speed'] < THRESHOLDS['ball_speed_min']
    df['is_mishit'] = low_ball_speed
    
    # determine overall validity
    df['is_valid'] = df['removal_reasons'] == ''
    
    # split into valid and invalid dataframes
    valid_df = df[df['is_valid']].copy()
    invalid_df = df[~df['is_valid']].copy()
    
    # clean up helper columns from output
    valid_df = valid_df.drop(columns=['removal_reasons', 'is_valid'])
    invalid_df = invalid_df.rename(columns={'removal_reasons': 'removal_reason'})
    
    # report cleaning summary
    print(f"data cleaning summary:")
    print(f"  total shots: {len(df):,}")
    print(f"  valid shots: {len(valid_df):,} ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"  removed shots: {len(invalid_df):,} ({len(invalid_df)/len(df)*100:.1f}%)")
    print(f"  shots flagged as mishits: {valid_df['is_mishit'].sum():,}")
    
    return valid_df, invalid_df


def segment_by_skill(df: pd.DataFrame) -> dict:
    """
    segment shots by player skill level based on ball speed.
    
    ball speed is the most reliable skill indicator because it directly
    reflects club head speed and strike quality combined. this segmentation
    allows analysis of how equipment performance varies across skill levels,
    which is critical for taylormade's product line strategy (qi10 max vs ls).
    
    args:
        df: cleaned dataframe with ball_speed column
        
    returns:
        dictionary with keys ['elite', 'scratch', 'mid', 'high', 'all']
        each value is a filtered dataframe for that skill segment
    """
    segments = {}
    
    # elite segment: ball speed >= 145 mph
    elite_mask = df['ball_speed'] >= SKILL_LEVELS['elite']['ball_speed_min']
    segments['elite'] = df[elite_mask].copy()
    
    # scratch segment: 130 <= ball speed < 145 mph
    scratch_mask = (
        (df['ball_speed'] >= SKILL_LEVELS['scratch']['ball_speed_min']) &
        (df['ball_speed'] < SKILL_LEVELS['scratch']['ball_speed_max'])
    )
    segments['scratch'] = df[scratch_mask].copy()
    
    # mid handicap segment: 110 <= ball speed < 130 mph
    mid_mask = (
        (df['ball_speed'] >= SKILL_LEVELS['mid']['ball_speed_min']) &
        (df['ball_speed'] < SKILL_LEVELS['mid']['ball_speed_max'])
    )
    segments['mid'] = df[mid_mask].copy()
    
    # high handicap segment: ball speed < 110 mph
    high_mask = df['ball_speed'] < SKILL_LEVELS['high']['ball_speed_max']
    segments['high'] = df[high_mask].copy()
    
    # also include all data for comparison
    segments['all'] = df.copy()
    
    # add skill level label to each segment
    segments['elite']['skill_level'] = 'elite'
    segments['scratch']['skill_level'] = 'scratch'
    segments['mid']['skill_level'] = 'mid'
    segments['high']['skill_level'] = 'high'
    segments['all']['skill_level'] = 'all'
    
    # report segmentation summary
    print(f"\nskill level segmentation:")
    for level, level_df in segments.items():
        if level != 'all':
            label = SKILL_LEVELS[level]['label']
            print(f"  {label}: {len(level_df):,} shots ({len(level_df)/len(df)*100:.1f}%)")
    
    return segments


def filter_by_ball_type(df: pd.DataFrame, ball_type: Optional[str] = None) -> pd.DataFrame:
    """
    filter shots by golf ball type.
    
    ball selection significantly impacts spin rate and launch conditions.
    for controlled equipment testing, isolating ball type reduces variables.
    
    args:
        df: dataframe with ball_type column
        ball_type: specific ball to filter (e.g., 'Premium'). if none, return all.
        
    returns:
        filtered dataframe
    """
    if ball_type is None:
        return df
    
    filtered = df[df['ball_type'] == ball_type].copy()
    print(f"filtered to ball type '{ball_type}': {len(filtered):,} shots")
    
    return filtered


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    generate summary statistics for a dataset.
    
    returns key metrics that taylormade performance research team would
    typically review when evaluating player or equipment data.
    
    args:
        df: cleaned dataframe
        
    returns:
        dictionary with summary statistics
    """
    summary = {
        'shot_count': len(df),
        'ball_speed': {
            'mean': df['ball_speed'].mean(),
            'std': df['ball_speed'].std(),
            'min': df['ball_speed'].min(),
            'max': df['ball_speed'].max(),
        },
        'carry': {
            'mean': df['carry'].mean(),
            'std': df['carry'].std(),
            'min': df['carry'].min(),
            'max': df['carry'].max(),
        },
        'offline': {
            'mean': df['offline'].mean(),
            'std': df['offline'].std(),  # this is dispersion
            'min': df['offline'].min(),
            'max': df['offline'].max(),
        },
        'spin_rate': {
            'mean': df['spin_rate'].mean(),
            'std': df['spin_rate'].std(),
            'min': df['spin_rate'].min(),
            'max': df['spin_rate'].max(),
        },
        'launch_angle': {
            'mean': df['launch_angle'].mean(),
            'std': df['launch_angle'].std(),
        },
        'smash_factor': {
            'mean': df['smash_factor'].mean(),
            'std': df['smash_factor'].std(),
        },
    }
    
    return summary


# =============================================================================
# main execution: demonstration of module usage
# =============================================================================
if __name__ == "__main__":
    # example usage demonstrating the full pipeline
    print("=" * 60)
    print("golf shot performance analyzer - data loader demo")
    print("=" * 60)
    
    # load raw data
    raw_df = load_data('/mnt/project/data.csv')
    print(f"\nloaded {len(raw_df):,} raw shots")
    
    # clean data
    clean_df, removed_df = clean_data(raw_df)
    
    # segment by skill level
    segments = segment_by_skill(clean_df)
    
    # filter to premium balls only (most common in dataset)
    premium_df = filter_by_ball_type(clean_df, 'Premium')
    
    # show summary for elite segment
    print("\n" + "=" * 60)
    print("elite segment summary (premium balls):")
    print("=" * 60)
    elite_premium = filter_by_ball_type(segments['elite'], 'Premium')
    summary = get_data_summary(elite_premium)
    print(f"  shots: {summary['shot_count']:,}")
    print(f"  ball speed: {summary['ball_speed']['mean']:.1f} ± {summary['ball_speed']['std']:.1f} mph")
    print(f"  carry: {summary['carry']['mean']:.1f} ± {summary['carry']['std']:.1f} yds")
    print(f"  dispersion: {summary['offline']['std']:.1f} yds (1σ)")
    print(f"  spin rate: {summary['spin_rate']['mean']:.0f} ± {summary['spin_rate']['std']:.0f} rpm")
