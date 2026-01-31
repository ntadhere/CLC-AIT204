"""
NBA Optimal Team Selection - Data Preparation
This script prepares the data for training a neural network to select an optimal 5-player basketball team.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load Dataset
print("Loading NBA Players Dataset...")
data = pd.read_csv('./data/all_seasons.csv')
print(f"Total records: {data.shape[0]}")
print(f"Total columns: {data.shape[1]}")

# Step 1: Select a 5-year window (2018-2023 - most recent complete data)
print("\n" + "="*80)
print("STEP 1: Selecting 5-year window (2018-19 to 2022-23)")
print("="*80)

seasons_5year = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
data_5year = data[data['season'].isin(seasons_5year)].copy()
print(f"Records in 5-year window: {data_5year.shape[0]}")

# Step 2: Filter players with minimum games played (quality filter)
# Keep players who played at least 20 games to ensure they have meaningful stats
print("\n" + "="*80)
print("STEP 2: Filtering players with minimum games played (GP >= 20)")
print("="*80)

data_filtered = data_5year[data_5year['gp'] >= 20].copy()
print(f"Records after filtering: {data_filtered.shape[0]}")

# Step 3: Aggregate player stats across the 5 years (average performance)
print("\n" + "="*80)
print("STEP 3: Aggregating player statistics across seasons")
print("="*80)

# Group by player and take the mean of their stats across seasons
aggregation_dict = {
    'age': 'mean',
    'player_height': 'first',  # Height doesn't change
    'player_weight': 'first',  # Weight relatively stable
    'gp': 'sum',  # Total games played
    'pts': 'mean',
    'reb': 'mean',
    'ast': 'mean',
    'net_rating': 'mean',
    'oreb_pct': 'mean',
    'dreb_pct': 'mean',
    'usg_pct': 'mean',
    'ts_pct': 'mean',
    'ast_pct': 'mean',
    'team_abbreviation': 'last',  # Most recent team
    'season': 'last'  # Most recent season
}

player_stats = data_filtered.groupby('player_name').agg(aggregation_dict).reset_index()
print(f"Unique players: {player_stats.shape[0]}")

# Step 4: Select top 100 players based on games played and performance
print("\n" + "="*80)
print("STEP 4: Selecting pool of 100 players")
print("="*80)

# Create a composite score to rank players
player_stats['composite_score'] = (
    player_stats['pts'] * 0.3 +
    player_stats['reb'] * 0.2 +
    player_stats['ast'] * 0.2 +
    player_stats['net_rating'] * 0.15 +
    player_stats['ts_pct'] * 20 * 0.15  # Scale ts_pct to similar range
)

# Select top 100 players
top_100_players = player_stats.nlargest(100, 'composite_score').copy()
print(f"Selected {top_100_players.shape[0]} players")
print("\nTop 10 players by composite score:")
print(top_100_players[['player_name', 'pts', 'reb', 'ast', 'composite_score']].head(10))

# Step 5: Define player positions/roles based on their statistics
print("\n" + "="*80)
print("STEP 5: Categorizing players by role/position")
print("="*80)

def categorize_player_role(row):
    """
    Categorize players into roles based on their statistics:
    - Scorer: High points, high usage
    - Playmaker: High assists
    - Rebounder: High rebounds
    - Defender: High defensive rebounds, low offensive usage
    - All-Around: Balanced stats
    """
    pts_high = row['pts'] > top_100_players['pts'].quantile(0.6)
    ast_high = row['ast'] > top_100_players['ast'].quantile(0.6)
    reb_high = row['reb'] > top_100_players['reb'].quantile(0.6)
    dreb_high = row['dreb_pct'] > top_100_players['dreb_pct'].quantile(0.6)
    
    if ast_high:
        return 'Playmaker'
    elif pts_high and row['usg_pct'] > 0.25:
        return 'Scorer'
    elif reb_high or dreb_high:
        return 'Rebounder'
    elif dreb_high and not pts_high:
        return 'Defender'
    else:
        return 'All-Around'

top_100_players['role'] = top_100_players.apply(categorize_player_role, axis=1)
print("\nPlayer distribution by role:")
print(top_100_players['role'].value_counts())

# Step 6: Create features for the neural network
print("\n" + "="*80)
print("STEP 6: Creating features for neural network")
print("="*80)

# Features that represent player characteristics
feature_columns = [
    'pts',           # Scoring ability
    'reb',           # Rebounding ability
    'ast',           # Playmaking ability
    'net_rating',    # Overall impact
    'ts_pct',        # Shooting efficiency
    'usg_pct',       # Ball usage
    'ast_pct',       # Assist rate
    'dreb_pct',      # Defensive rebounding
    'oreb_pct',      # Offensive rebounding
    'age',           # Experience/prime
]

X = top_100_players[feature_columns].copy()

# Step 7: Define what makes an "optimal team"
print("\n" + "="*80)
print("STEP 7: Defining optimal team criteria")
print("="*80)

print("""
OPTIMAL TEAM DEFINITION:
An optimal basketball team of 5 players should have:

1. BALANCED SCORING (1-2 primary scorers, 2-3 complementary scorers)
   - At least 1 player with pts > 20
   - Team average pts between 12-18

2. PLAYMAKING (1-2 playmakers)
   - At least 1 player with ast > 5
   - Team average ast > 3

3. REBOUNDING (1-2 strong rebounders)
   - At least 1 player with reb > 7
   - Team average reb > 5

4. EFFICIENCY (high true shooting percentage)
   - Team average ts_pct > 0.55

5. DEFENSIVE PRESENCE (good defensive rebounding)
   - Team average dreb_pct > 0.15

6. ROLE DIVERSITY (balanced team composition)
   - Mix of Scorers, Playmakers, Rebounders, Defenders
   - Avoid having all players of the same role

7. NET RATING (positive team impact)
   - Team average net_rating > 0
""")

# Step 8: Create training labels
print("\n" + "="*80)
print("STEP 8: Creating training labels for team quality")
print("="*80)

def evaluate_team_quality(team_indices, player_data):
    """
    Evaluate if a team of 5 players is optimal based on defined criteria.
    Returns a quality score from 0 to 1.
    """
    team = player_data.iloc[team_indices]
    
    score = 0.0
    
    # Criterion 1: Balanced scoring
    has_primary_scorer = (team['pts'] > 20).sum() >= 1
    avg_pts = team['pts'].mean()
    if has_primary_scorer and 12 <= avg_pts <= 18:
        score += 0.15
    
    # Criterion 2: Playmaking
    has_playmaker = (team['ast'] > 5).sum() >= 1
    avg_ast = team['ast'].mean()
    if has_playmaker and avg_ast > 3:
        score += 0.15
    
    # Criterion 3: Rebounding
    has_rebounder = (team['reb'] > 7).sum() >= 1
    avg_reb = team['reb'].mean()
    if has_rebounder and avg_reb > 5:
        score += 0.15
    
    # Criterion 4: Efficiency
    avg_ts = team['ts_pct'].mean()
    if avg_ts > 0.55:
        score += 0.15
    
    # Criterion 5: Defensive presence
    avg_dreb = team['dreb_pct'].mean()
    if avg_dreb > 0.15:
        score += 0.15
    
    # Criterion 6: Role diversity
    role_counts = team['role'].value_counts()
    if len(role_counts) >= 3:  # At least 3 different roles
        score += 0.15
    
    # Criterion 7: Net rating
    avg_net_rating = team['net_rating'].mean()
    if avg_net_rating > 0:
        score += 0.10
    
    return score

# Generate training data by creating random teams and evaluating them
print("Generating training data (10,000 random team combinations)...")
np.random.seed(42)
n_samples = 10000

team_samples = []
quality_scores = []

for _ in range(n_samples):
    # Randomly select 5 players
    team_indices = np.random.choice(100, size=5, replace=False)
    
    # Get their combined features (mean of the 5 players)
    team_features = X.iloc[team_indices].mean().values
    
    # Evaluate team quality
    quality = evaluate_team_quality(team_indices, top_100_players)
    
    team_samples.append(team_features)
    quality_scores.append(quality)

X_teams = np.array(team_samples)
y_quality = np.array(quality_scores)

print(f"Training samples created: {X_teams.shape}")
print(f"Quality score range: {y_quality.min():.3f} - {y_quality.max():.3f}")
print(f"Mean quality score: {y_quality.mean():.3f}")
print(f"Quality score distribution:")
print(f"  Excellent (>0.7): {(y_quality > 0.7).sum()}")
print(f"  Good (0.5-0.7): {((y_quality > 0.5) & (y_quality <= 0.7)).sum()}")
print(f"  Average (0.3-0.5): {((y_quality > 0.3) & (y_quality <= 0.5)).sum()}")
print(f"  Poor (<0.3): {(y_quality <= 0.3).sum()}")

# Step 9: Normalize features
print("\n" + "="*80)
print("STEP 9: Normalizing features")
print("="*80)

scaler = StandardScaler()
X_teams_scaled = scaler.fit_transform(X_teams)

print("Features normalized using StandardScaler")
print(f"Scaled data shape: {X_teams_scaled.shape}")

# Step 10: Split into training and testing sets
print("\n" + "="*80)
print("STEP 10: Splitting data into train/test sets")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_teams_scaled, y_quality, 
    test_size=0.2, 
    random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Step 11: Save processed data
print("\n" + "="*80)
print("STEP 11: Saving processed data")
print("="*80)

# Save the player pool
top_100_players.to_csv('top_100_players.csv', index=False)
print("✓ Saved: top_100_players.csv")

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: scaler.pkl")

# Save training data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("✓ Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")

# Save feature columns
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("✓ Saved: feature_columns.pkl")

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE!")
print("="*80)
print(f"\nReady for model training with:")
print(f"  - {len(top_100_players)} players in the pool")
print(f"  - {len(feature_columns)} features per player")
print(f"  - {X_train.shape[0]} training samples")
print(f"  - {X_test.shape[0]} testing samples")
