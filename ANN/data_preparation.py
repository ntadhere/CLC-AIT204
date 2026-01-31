"""
NBA Optimal Team Selection - Data Preparation Script
====================================================

Purpose:
    This script processes raw NBA player statistics to create a curated dataset
    for training a neural network. It performs the following tasks:
    1. Loads and filters NBA player data from 2018-2023
    2. Aggregates player statistics across multiple seasons
    3. Selects top 100 players based on composite scoring
    4. Categorizes players into roles (Playmaker, Scorer, etc.)
    5. Generates synthetic training data (team combinations)
    6. Evaluates team quality based on 7 optimal criteria
    7. Normalizes features and splits into train/test sets

Author: [Your Name]
Date: January 31, 2026
Course: AIT-204 - Topic 2 Assignment
"""

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd                          # Data manipulation and analysis
import numpy as np                           # Numerical computing
from sklearn.preprocessing import StandardScaler  # Feature normalization
from sklearn.model_selection import train_test_split  # Data splitting
import pickle                                # Save/load Python objects

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

print("="*80)
print("LOADING NBA PLAYERS DATASET")
print("="*80)

# Load the CSV file containing NBA player statistics
# Dataset source: Kaggle - "NBA Players" by Justinas Cirtautas
# Contains player statistics from 1996-97 through 2022-23 seasons
data = pd.read_csv('./data/all_seasons.csv')

print(f"Total records loaded: {data.shape[0]}")
print(f"Total columns: {data.shape[1]}")
print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# STEP 2: SELECT 5-YEAR TIME WINDOW
# ============================================================================

print("\n" + "="*80)
print("STEP 2: SELECTING 5-YEAR WINDOW (2018-19 to 2022-23)")
print("="*80)

# Select recent seasons to ensure:
# - Modern playing styles and rules
# - Consistent statistical tracking
# - Players at or near current performance levels
seasons_5year = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23']

# Filter dataframe to only include records from these seasons
data_5year = data[data['season'].isin(seasons_5year)].copy()

print(f"Records in 5-year window: {data_5year.shape[0]}")
print(f"Percentage of total data: {(data_5year.shape[0] / data.shape[0]) * 100:.1f}%")

# ============================================================================
# STEP 3: FILTER BY GAMES PLAYED
# ============================================================================

print("\n" + "="*80)
print("STEP 3: FILTERING PLAYERS WITH MINIMUM GAMES PLAYED (GP >= 20)")
print("="*80)

# Keep only players who played at least 20 games in a season
# Rationale:
# - Ensures statistical reliability (larger sample size)
# - Filters out injured players or end-of-bench players
# - 20 games = ~25% of an NBA season (82 games)
MIN_GAMES_PLAYED = 20

data_filtered = data_5year[data_5year['gp'] >= MIN_GAMES_PLAYED].copy()

print(f"Records before filtering: {data_5year.shape[0]}")
print(f"Records after filtering: {data_filtered.shape[0]}")
print(f"Records removed: {data_5year.shape[0] - data_filtered.shape[0]}")

# ============================================================================
# STEP 4: AGGREGATE PLAYER STATISTICS ACROSS SEASONS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: AGGREGATING PLAYER STATISTICS ACROSS SEASONS")
print("="*80)

# Since players appear multiple times (one row per season), we need to
# aggregate their statistics to get a single row per player

# Define how to aggregate each column:
aggregation_dict = {
    'age': 'mean',                # Average age across seasons
    'player_height': 'first',     # Height doesn't change (take first value)
    'player_weight': 'first',     # Weight is relatively stable
    'gp': 'sum',                  # Sum total games played across all seasons
    'pts': 'mean',                # Average points per game
    'reb': 'mean',                # Average rebounds per game
    'ast': 'mean',                # Average assists per game
    'net_rating': 'mean',         # Average net rating (team performance when player is on court)
    'oreb_pct': 'mean',          # Average offensive rebound percentage
    'dreb_pct': 'mean',          # Average defensive rebound percentage
    'usg_pct': 'mean',           # Average usage percentage (how much player uses possessions)
    'ts_pct': 'mean',            # Average true shooting percentage (efficiency metric)
    'ast_pct': 'mean',           # Average assist percentage
    'team_abbreviation': 'last', # Most recent team
    'season': 'last'             # Most recent season played
}

# Group by player name and aggregate
# .reset_index() converts player_name from index back to regular column
player_stats = data_filtered.groupby('player_name').agg(aggregation_dict).reset_index()

print(f"Unique players after aggregation: {player_stats.shape[0]}")
print(f"Average games per player: {player_stats['gp'].mean():.1f}")

# ============================================================================
# STEP 5: SELECT TOP 100 PLAYERS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SELECTING TOP 100 PLAYERS BASED ON COMPOSITE SCORE")
print("="*80)

# Create a composite score to rank players
# This weighted formula emphasizes:
# - Scoring ability (30% weight)
# - Rebounding (20% weight)
# - Playmaking (20% weight)
# - Overall impact (15% weight via net_rating)
# - Efficiency (15% weight via true shooting %)
#
# Note: ts_pct is multiplied by 20 to scale it to similar range as other stats
# (ts_pct ranges from 0.4-0.7, so 20× gives 8-14)

player_stats['composite_score'] = (
    player_stats['pts'] * 0.30 +      # Points per game (highest weight)
    player_stats['reb'] * 0.20 +      # Rebounds per game
    player_stats['ast'] * 0.20 +      # Assists per game
    player_stats['net_rating'] * 0.15 +  # Net rating
    player_stats['ts_pct'] * 20 * 0.15   # True shooting % (scaled)
)

# Select the top 100 players by composite score
# .nlargest() returns the N rows with largest values for specified column
top_100_players = player_stats.nlargest(100, 'composite_score').copy()

print(f"Selected {top_100_players.shape[0]} players")
print(f"\nComposite score statistics:")
print(f"  Mean: {top_100_players['composite_score'].mean():.2f}")
print(f"  Std:  {top_100_players['composite_score'].std():.2f}")
print(f"  Min:  {top_100_players['composite_score'].min():.2f}")
print(f"  Max:  {top_100_players['composite_score'].max():.2f}")

# Display top 10 players
print("\nTop 10 players by composite score:")
print(top_100_players[['player_name', 'pts', 'reb', 'ast', 'composite_score']].head(10).to_string(index=False))

# ============================================================================
# STEP 6: CATEGORIZE PLAYERS BY ROLE
# ============================================================================

print("\n" + "="*80)
print("STEP 6: CATEGORIZING PLAYERS BY ROLE/POSITION")
print("="*80)

def categorize_player_role(row):
    """
    Categorize a player into one of 5 roles based on their statistics.
    
    Roles:
    - Playmaker: High assists (>60th percentile)
    - Scorer: High points (>60th percentile) and high usage (>25%)
    - Rebounder: High rebounds (>60th percentile) or high defensive rebounding
    - Defender: High defensive rebounding but low offensive output
    - All-Around: Balanced stats that don't fit other categories
    
    Parameters:
        row (pandas.Series): A row from the dataframe with player statistics
    
    Returns:
        str: Role category ('Playmaker', 'Scorer', 'Rebounder', 'Defender', or 'All-Around')
    """
    
    # Calculate percentile thresholds (60th percentile = top 40% of players)
    pts_high = row['pts'] > top_100_players['pts'].quantile(0.6)
    ast_high = row['ast'] > top_100_players['ast'].quantile(0.6)
    reb_high = row['reb'] > top_100_players['reb'].quantile(0.6)
    dreb_high = row['dreb_pct'] > top_100_players['dreb_pct'].quantile(0.6)
    
    # Role assignment logic (checked in priority order)
    if ast_high:
        # Primary ball-handlers and facilitators
        return 'Playmaker'
    elif pts_high and row['usg_pct'] > 0.25:
        # High-volume scorers who use a lot of possessions
        return 'Scorer'
    elif reb_high or dreb_high:
        # Players who dominate the boards
        return 'Rebounder'
    elif dreb_high and not pts_high:
        # Strong defensive rebounders without offensive production
        return 'Defender'
    else:
        # Players with balanced contributions across categories
        return 'All-Around'

# Apply the categorization function to each player
top_100_players['role'] = top_100_players.apply(categorize_player_role, axis=1)

# Display role distribution
print("\nPlayer distribution by role:")
role_counts = top_100_players['role'].value_counts()
for role, count in role_counts.items():
    print(f"  {role:12s}: {count:3d} players ({count/100*100:.0f}%)")

# ============================================================================
# STEP 6: DEFINE FEATURES FOR NEURAL NETWORK
# ============================================================================

print("\n" + "="*80)
print("STEP 6: CREATING FEATURES FOR NEURAL NETWORK")
print("="*80)

# Define the features that will be used to train the neural network
# These 10 features represent key aspects of player performance
feature_columns = [
    'pts',           # Scoring ability - Points per game
    'reb',           # Rebounding ability - Rebounds per game
    'ast',           # Playmaking ability - Assists per game
    'net_rating',    # Overall impact - Net rating per 100 possessions
    'ts_pct',        # Shooting efficiency - True shooting percentage
    'usg_pct',       # Ball usage - Usage percentage
    'ast_pct',       # Assist rate - Assist percentage
    'dreb_pct',      # Defensive rebounding - Defensive rebound percentage
    'oreb_pct',      # Offensive rebounding - Offensive rebound percentage
    'age',           # Experience/prime status - Player age
]

print(f"Selected {len(feature_columns)} features for the model:")
for i, feature in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {feature}")

# Extract these features from the player dataframe
# This creates a new dataframe with only the columns we need
X = top_100_players[feature_columns].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"  Rows (players): {X.shape[0]}")
print(f"  Columns (features): {X.shape[1]}")

# ============================================================================
# STEP 7: DEFINE OPTIMAL TEAM CRITERIA
# ============================================================================

print("\n" + "="*80)
print("STEP 7: DEFINING OPTIMAL TEAM CRITERIA")
print("="*80)

# These 7 criteria define what makes a basketball team "optimal"
# Each criterion is based on basketball strategy and analytics research
print("""
OPTIMAL TEAM DEFINITION:
An optimal basketball team of 5 players should have:

1. BALANCED SCORING (1-2 primary scorers, 2-3 complementary scorers)
   - At least 1 player with pts > 20
   - Team average pts between 12-18
   - Ensures offense without over-reliance on one player

2. PLAYMAKING (1-2 playmakers)
   - At least 1 player with ast > 5
   - Team average ast > 3
   - Ensures good ball movement and shot creation

3. REBOUNDING (1-2 strong rebounders)
   - At least 1 player with reb > 7
   - Team average reb > 5
   - Ensures possession control

4. EFFICIENCY (high true shooting percentage)
   - Team average ts_pct > 0.55
   - Ensures quality shots, not just volume

5. DEFENSIVE PRESENCE (good defensive rebounding)
   - Team average dreb_pct > 0.15
   - Ensures ability to end opponent possessions

6. ROLE DIVERSITY (balanced team composition)
   - Mix of at least 3 different roles (Scorers, Playmakers, Rebounders, etc.)
   - Avoids having all players of the same type

7. NET RATING (positive team impact)
   - Team average net_rating > 0
   - Ensures players contribute to winning
""")

def evaluate_team_quality(team_indices, player_data):
    """
    Evaluate if a team of 5 players is optimal based on defined criteria.
    
    Parameters:
        team_indices (list or array): Indices of 5 players in player_data
        player_data (pandas.DataFrame): DataFrame containing all player statistics
    
    Returns:
        float: Quality score from 0.0 to 1.0 (sum of criteria met / 7)
    
    Criteria Breakdown:
        Each criterion contributes 0.10 to 0.15 to the total score
        Perfect team (all 7 criteria met) scores 1.0
        Poor team (no criteria met) scores 0.0
    """
    
    # Extract the team's player data
    team = player_data.iloc[team_indices]
    
    # Initialize score
    score = 0.0
    
    # Criterion 1: Balanced scoring (0.15 points possible)
    has_primary_scorer = (team['pts'] > 20).sum() >= 1  # At least one 20+ PPG scorer
    avg_pts = team['pts'].mean()
    if has_primary_scorer and 12 <= avg_pts <= 18:
        score += 0.15
    
    # Criterion 2: Playmaking (0.15 points possible)
    has_playmaker = (team['ast'] > 5).sum() >= 1  # At least one player with 5+ APG
    avg_ast = team['ast'].mean()
    if has_playmaker and avg_ast > 3:
        score += 0.15
    
    # Criterion 3: Rebounding (0.15 points possible)
    has_rebounder = (team['reb'] > 7).sum() >= 1  # At least one player with 7+ RPG
    avg_reb = team['reb'].mean()
    if has_rebounder and avg_reb > 5:
        score += 0.15
    
    # Criterion 4: Shooting efficiency (0.15 points possible)
    avg_ts = team['ts_pct'].mean()
    if avg_ts > 0.55:  # Above-average efficiency
        score += 0.15
    
    # Criterion 5: Defensive presence (0.15 points possible)
    avg_dreb = team['dreb_pct'].mean()
    if avg_dreb > 0.15:  # Solid defensive rebounding
        score += 0.15
    
    # Criterion 6: Role diversity (0.15 points possible)
    role_counts = team['role'].value_counts()
    if len(role_counts) >= 3:  # At least 3 different roles represented
        score += 0.15
    
    # Criterion 7: Positive net rating (0.10 points possible)
    avg_net_rating = team['net_rating'].mean()
    if avg_net_rating > 0:  # Team has positive impact
        score += 0.10
    
    return score

# ============================================================================
# STEP 8: GENERATE TRAINING DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 8: GENERATING TRAINING DATA (TEAM COMBINATIONS)")
print("="*80)

# Set random seed for reproducibility
# This ensures the same random teams are generated each time
np.random.seed(42)

# Number of random team combinations to generate
# More samples = better model training, but takes longer
N_SAMPLES = 10000

print(f"Generating {N_SAMPLES} random team combinations...")
print("This may take a minute...")

# Initialize lists to store results
team_samples = []      # Will store team feature averages
quality_scores = []    # Will store quality scores

# Generate random teams
for i in range(N_SAMPLES):
    # Randomly select 5 players without replacement
    # np.random.choice(100, 5, replace=False) picks 5 unique numbers from 0-99
    team_indices = np.random.choice(100, size=5, replace=False)
    
    # Calculate team features (average of the 5 players' stats)
    team_features = top_100_players.iloc[team_indices][feature_columns].mean().values
    
    # Evaluate team quality using our criteria
    quality = evaluate_team_quality(team_indices, top_100_players)
    
    # Store results
    team_samples.append(team_features)
    quality_scores.append(quality)
    
    # Print progress every 1000 samples
    if (i + 1) % 1000 == 0:
        print(f"  Generated {i+1}/{N_SAMPLES} teams...")

# Convert lists to numpy arrays for easier manipulation
X_teams = np.array(team_samples)      # Shape: (10000, 10) - 10 features per team
y_quality = np.array(quality_scores)  # Shape: (10000,) - 1 quality score per team

print(f"\nTraining data created successfully!")
print(f"  Input features shape: {X_teams.shape}")
print(f"  Output labels shape: {y_quality.shape}")
print(f"\nQuality score statistics:")
print(f"  Mean:   {y_quality.mean():.3f}")
print(f"  Std:    {y_quality.std():.3f}")
print(f"  Min:    {y_quality.min():.3f}")
print(f"  Max:    {y_quality.max():.3f}")

# Analyze quality score distribution
print(f"\nQuality score distribution:")
print(f"  Excellent (>0.7):  {(y_quality > 0.7).sum():5d} teams ({(y_quality > 0.7).sum()/N_SAMPLES*100:.1f}%)")
print(f"  Good (0.5-0.7):    {((y_quality > 0.5) & (y_quality <= 0.7)).sum():5d} teams ({((y_quality > 0.5) & (y_quality <= 0.7)).sum()/N_SAMPLES*100:.1f}%)")
print(f"  Average (0.3-0.5): {((y_quality > 0.3) & (y_quality <= 0.5)).sum():5d} teams ({((y_quality > 0.3) & (y_quality <= 0.5)).sum()/N_SAMPLES*100:.1f}%)")
print(f"  Poor (<0.3):       {(y_quality <= 0.3).sum():5d} teams ({(y_quality <= 0.3).sum()/N_SAMPLES*100:.1f}%)")

# ============================================================================
# STEP 9: FEATURE NORMALIZATION
# ============================================================================

print("\n" + "="*80)
print("STEP 9: NORMALIZING FEATURES")
print("="*80)

# Features have different scales (e.g., pts: 0-30, ts_pct: 0.4-0.7)
# Normalization ensures all features contribute equally to the neural network

# StandardScaler transforms each feature to have:
# - Mean = 0
# - Standard deviation = 1
# Formula: z = (x - mean) / std_dev

scaler = StandardScaler()
X_teams_scaled = scaler.fit_transform(X_teams)

print("Features normalized using StandardScaler")
print(f"  Original data shape: {X_teams.shape}")
print(f"  Scaled data shape:   {X_teams_scaled.shape}")
print(f"\nExample: First team's features before and after scaling:")
print(f"  Before (first 5 features): {X_teams[0, :5]}")
print(f"  After  (first 5 features): {X_teams_scaled[0, :5]}")

# ============================================================================
# STEP 10: TRAIN/TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("STEP 10: SPLITTING DATA INTO TRAIN/TEST SETS")
print("="*80)

# Split data into training (80%) and testing (20%) sets
# Training set: Used to train the neural network
# Testing set: Used to evaluate performance on unseen data
# random_state=42: Ensures same split every time (reproducibility)

X_train, X_test, y_train, y_test = train_test_split(
    X_teams_scaled,   # Input features (normalized)
    y_quality,        # Output labels (quality scores)
    test_size=0.2,    # 20% for testing
    random_state=42   # Reproducibility
)

print(f"Training set:   {X_train.shape[0]:5d} samples ({X_train.shape[0]/N_SAMPLES*100:.0f}%)")
print(f"Testing set:    {X_test.shape[0]:5d} samples ({X_test.shape[0]/N_SAMPLES*100:.0f}%)")
print(f"\nTraining quality score range: {y_train.min():.3f} to {y_train.max():.3f}")
print(f"Testing quality score range:  {y_test.min():.3f} to {y_test.max():.3f}")

# ============================================================================
# STEP 11: SAVE PROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 11: SAVING PROCESSED DATA TO DISK")
print("="*80)

# Save player pool to CSV (human-readable format)
top_100_players.to_csv('top_100_players.csv', index=False)
print("✓ Saved: top_100_players.csv")

# Save the scaler (needed for future predictions)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

# Save training and testing data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("✓ Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")

# Save feature column names (already defined earlier in Step 6)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✓ Saved: feature_columns.pkl")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE!")
print("="*80)

print(f"\nSummary of processed data:")
print(f"  ✓ Selected {len(top_100_players)} players from 5-year window (2018-2023)")
print(f"  ✓ Generated {N_SAMPLES} team combinations")
print(f"  ✓ Created {len(feature_columns)} features per team")
print(f"  ✓ Split into {X_train.shape[0]} training and {X_test.shape[0]} testing samples")
print(f"  ✓ Normalized features using StandardScaler")
print(f"  ✓ Saved all data files for model training")

print(f"\nNext step: Run model.py to train the neural network!")

# END OF SCRIPT