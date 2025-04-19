import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MLBModelAnalysis:
    """
    This class performs the analytical modeling to identify hot batters,
    effective pitchers, and predict interesting upcoming matchups.
    """

    def __init__(self, analyzer):
        """Initialize with an MLBAnalyzer instance that has collected data"""
        self.analyzer = analyzer
        self.hot_batters = None
        self.effective_pitchers = None
        self.matchup_predictions = None

    def analyze_hot_batters(self, min_games=5, recency_weight=True):
        """
        Identify hot batters based on recent performance

        Args:
            min_games (int): Minimum number of games to consider
            recency_weight (bool): Whether to weight recent games more heavily

        Returns:
            DataFrame: Batter stats with "hotness" scores
        """
        if self.analyzer.batter_data is None or self.analyzer.batter_data.empty:
            print("No batter data available. Run collect_all_team_data first.")
            return None

        # Ensure columns we need are present
        required_cols = ['player_id', 'player_name', 'team_name', 'date', 'atBats', 'hits', 'doubles',
                         'triples', 'homeRuns', 'rbi', 'baseOnBalls', 'strikeOuts']

        # Find which columns are available
        available_cols = [col for col in required_cols if col in self.analyzer.batter_data.columns]
        missing_cols = [col for col in required_cols if col not in available_cols]

        if missing_cols:
            print(f"Warning: Missing columns in data: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                if col not in ['player_id', 'player_name', 'team_name', 'date']:  # These are essential
                    self.analyzer.batter_data[col] = 0

        # Convert date to datetime
        self.analyzer.batter_data['date'] = pd.to_datetime(self.analyzer.batter_data['date'])

        # Sort by player and date
        sorted_data = self.analyzer.batter_data.sort_values(['player_id', 'date'], ascending=[True, False])

        # Group by player
        player_groups = sorted_data.groupby(['player_id', 'player_name', 'team_name'])

        # Calculate player stats
        player_stats = []

        for (player_id, player_name, team_name), group in player_groups:
            # Only consider players with minimum number of games
            if len(group) < min_games:
                continue

            # Get the most recent games
            recent_games = group.head(10).copy()

            # Apply recency weights if enabled
            if recency_weight:
                # Create weights that decay with older games
                weights = np.linspace(1.0, 0.5, len(recent_games))

                # Apply weights to relevant stats
                for stat in ['hits', 'doubles', 'triples', 'homeRuns', 'rbi', 'baseOnBalls', 'strikeOuts']:
                    if stat in recent_games.columns:
                        recent_games[f'weighted_{stat}'] = recent_games[stat] * weights

            # Calculate key metrics
            totals = {}
            totals['player_id'] = player_id
            totals['player_name'] = player_name
            totals['team_name'] = team_name
            totals['games'] = len(recent_games)

            # Calculate batting average and other metrics
            if 'atBats' in recent_games.columns and 'hits' in recent_games.columns:
                totals['at_bats'] = recent_games['atBats'].sum()
                totals['hits'] = recent_games['hits'].sum()

                # Avoid division by zero
                if totals['at_bats'] > 0:
                    totals['batting_avg'] = totals['hits'] / totals['at_bats']
                else:
                    totals['batting_avg'] = 0.0

            # Calculate weighted metrics if used
            if recency_weight:
                if 'weighted_hits' in recent_games.columns and 'atBats' in recent_games.columns:
                    weighted_hits = recent_games['weighted_hits'].sum()
                    weighted_at_bats = (recent_games['atBats'] * weights).sum()

                    if weighted_at_bats > 0:
                        totals['weighted_avg'] = weighted_hits / weighted_at_bats
                    else:
                        totals['weighted_avg'] = 0.0

            # Calculate power metrics
            if all(col in recent_games.columns for col in ['doubles', 'triples', 'homeRuns', 'atBats']):
                totals['doubles'] = recent_games['doubles'].sum()
                totals['triples'] = recent_games['triples'].sum()
                totals['home_runs'] = recent_games['homeRuns'].sum()

                # Calculate slugging percentage
                if totals['at_bats'] > 0:
                    totals['slugging'] = (totals['hits'] + totals['doubles'] + 2 * totals['triples'] + 3 * totals[
                        'home_runs']) / totals['at_bats']
                else:
                    totals['slugging'] = 0.0

            # Calculate OBP if we have walks
            if 'baseOnBalls' in recent_games.columns and 'atBats' in recent_games.columns:
                walks = recent_games['baseOnBalls'].sum()
                hits = totals.get('hits', 0)
                at_bats = totals['at_bats']

                # Formula: (Hits + Walks) / (At Bats + Walks)
                if at_bats + walks > 0:
                    totals['obp'] = (hits + walks) / (at_bats + walks)
                else:
                    totals['obp'] = 0.0

                # Calculate OPS
                if 'slugging' in totals:
                    totals['ops'] = totals['obp'] + totals['slugging']

            # Calculate strikeout rate
            if 'strikeOuts' in recent_games.columns and 'atBats' in recent_games.columns:
                strikeouts = recent_games['strikeOuts'].sum()
                if totals['at_bats'] > 0:
                    totals['k_rate'] = strikeouts / totals['at_bats']
                else:
                    totals['k_rate'] = 0.0

            # Calculate a composite "hotness" score
            # This formula can be refined based on your preferences
            hotness_components = []

            if 'batting_avg' in totals:
                hotness_components.append(totals['batting_avg'] * 1.0)  # Weight batting average normally

            if 'weighted_avg' in totals:
                hotness_components.append(totals['weighted_avg'] * 1.5)  # Weight recent performance higher

            if 'slugging' in totals:
                hotness_components.append(totals['slugging'] * 0.8)  # Power is important

            if 'ops' in totals:
                hotness_components.append(totals['ops'] * 0.7)  # OPS is a good overall metric

            if 'k_rate' in totals:
                # Lower strikeout rate is better, so invert it
                hotness_components.append((1 - totals['k_rate']) * 0.5)

            if hotness_components:
                totals['hotness_score'] = sum(hotness_components) / len(hotness_components)
            else:
                totals['hotness_score'] = 0.0

            player_stats.append(totals)

        # Convert to DataFrame
        stats_df = pd.DataFrame(player_stats)

        # Sort by hotness score
        if 'hotness_score' in stats_df.columns:
            stats_df = stats_df.sort_values('hotness_score', ascending=False)

        self.hot_batters = stats_df
        return stats_df

    def analyze_pitchers(self, min_games=3):
        """
        Analyze pitcher effectiveness based on recent performance

        Args:
            min_games (int): Minimum number of games to consider

        Returns:
            DataFrame: Pitcher stats with effectiveness scores
        """
        if self.analyzer.pitcher_data is None or self.analyzer.pitcher_data.empty:
            print("No pitcher data available. Run collect_all_team_data first.")
            return None

        # Convert date to datetime
        self.analyzer.pitcher_data['date'] = pd.to_datetime(self.analyzer.pitcher_data['date'])

        # Sort by player and date
        sorted_data = self.analyzer.pitcher_data.sort_values(['player_id', 'date'], ascending=[True, False])

        # Group by player
        player_groups = sorted_data.groupby(['player_id', 'player_name', 'team_name'])

        # Calculate pitcher stats
        pitcher_stats = []

        for (player_id, player_name, team_name), group in player_groups:
            # Only consider pitchers with minimum number of games
            if len(group) < min_games:
                continue

            # Get the most recent games
            recent_games = group.head(10).copy()

            # Calculate key metrics
            totals = {}
            totals['player_id'] = player_id
            totals['player_name'] = player_name
            totals['team_name'] = team_name
            totals['games'] = len(recent_games)

            # Process innings pitched
            if 'inningsPitched' in recent_games.columns:
                # Convert innings pitched to numeric (handle fractional innings)
                innings = recent_games['inningsPitched'].apply(
                    lambda x: float(str(x).replace('.1', '.33').replace('.2', '.67'))
                    if not pd.isna(x) else 0.0
                ).sum()

                totals['innings_pitched'] = innings

            # Calculate ERA
            if 'inningsPitched' in recent_games.columns and 'earnedRuns' in recent_games.columns:
                earned_runs = recent_games['earnedRuns'].sum()

                if totals.get('innings_pitched', 0) > 0:
                    totals['era'] = (earned_runs * 9) / totals['innings_pitched']
                else:
                    totals['era'] = float('inf')

            # Calculate WHIP
            if all(col in recent_games.columns for col in ['inningsPitched', 'baseOnBalls', 'hits']):
                walks = recent_games['baseOnBalls'].sum()
                hits = recent_games['hits'].sum()

                if totals.get('innings_pitched', 0) > 0:
                    totals['whip'] = (walks + hits) / totals['innings_pitched']
                else:
                    totals['whip'] = float('inf')

            # Calculate K/9
            if 'inningsPitched' in recent_games.columns and 'strikeOuts' in recent_games.columns:
                strikeouts = recent_games['strikeOuts'].sum()

                if totals.get('innings_pitched', 0) > 0:
                    totals['k_per_9'] = (strikeouts * 9) / totals['innings_pitched']
                else:
                    totals['k_per_9'] = 0.0

            # Calculate BB/9
            if 'inningsPitched' in recent_games.columns and 'baseOnBalls' in recent_games.columns:
                walks = recent_games['baseOnBalls'].sum()

                if totals.get('innings_pitched', 0) > 0:
                    totals['bb_per_9'] = (walks * 9) / totals['innings_pitched']
                else:
                    totals['bb_per_9'] = float('inf')

            # Calculate a composite "effectiveness" score
            # Lower is better for ERA and WHIP, higher is better for K/9
            effectiveness_components = []

            if 'era' in totals and totals['era'] != float('inf'):
                # Normalize ERA - lower is better, so invert
                # Typical ERA range is 0-10, with 3-4 being good
                era_score = max(0, min(1, 1 - (totals['era'] / 10)))
                effectiveness_components.append(era_score * 1.5)  # Weight ERA highly

            if 'whip' in totals and totals['whip'] != float('inf'):
                # Normalize WHIP - lower is better, so invert
                # Typical WHIP range is 0-3, with 1-1.3 being good
                whip_score = max(0, min(1, 1 - (totals['whip'] / 3)))
                effectiveness_components.append(whip_score * 1.3)

            if 'k_per_9' in totals:
                # Normalize K/9 - higher is better
                # Typical K/9 range is 0-15, with 9+ being good
                k_score = min(1, totals['k_per_9'] / 15)
                effectiveness_components.append(k_score * 1.0)

            if 'bb_per_9' in totals and totals['bb_per_9'] != float('inf'):
                # Normalize BB/9 - lower is better, so invert
                # Typical BB/9 range is 0-6, with 2-3 being average
                bb_score = max(0, min(1, 1 - (totals['bb_per_9'] / 6)))
                effectiveness_components.append(bb_score * 0.8)

            if effectiveness_components:
                # Get weighted average of all components
                total_weight = 1.5 + 1.3 + 1.0 + 0.8  # Sum of all weights
                totals['effectiveness_score'] = sum(effectiveness_components) / (
                            len(effectiveness_components) * total_weight / len(effectiveness_components))
            else:
                totals['effectiveness_score'] = 0.0

            pitcher_stats.append(totals)

        # Convert to DataFrame
        stats_df = pd.DataFrame(pitcher_stats)

        # Sort by effectiveness score (higher is better)
        if 'effectiveness_score' in stats_df.columns:
            stats_df = stats_df.sort_values('effectiveness_score', ascending=False)

        self.effective_pitchers = stats_df
        return stats_df

    def predict_batter_pitcher_matchups(self, days_ahead=7, top_n=20):
        """
        Predict outcomes of upcoming batter-pitcher matchups

        Args:
            days_ahead (int): Number of days ahead to check schedules
            top_n (int): Number of top batters and pitchers to consider

        Returns:
            DataFrame: Predictions for upcoming matchups
        """
        if self.hot_batters is None or self.effective_pitchers is None:
            print("Run analyze_hot_batters and analyze_pitchers first.")
            return None

        # Get top batters and pitchers
        top_batters = self.hot_batters.head(top_n) if len(self.hot_batters) > 0 else self.hot_batters
        top_pitchers = self.effective_pitchers.head(top_n) if len(
            self.effective_pitchers) > 0 else self.effective_pitchers

        # Get team IDs for all teams with top batters or pitchers
        team_ids = set()
        team_names_to_ids = {}

        # FIX: The teams dictionary is keyed by team ID in the MLBAnalyzer
        # So we need to use the team_id as the key, not access it from the team_data
        for team_id, team_data in self.analyzer.teams.items():
            # Use the ID from the dictionary key, which is already an integer or string
            team_names_to_ids[team_data['name']] = team_id

        # Add teams with top batters or pitchers
        for df in [top_batters, top_pitchers]:
            if 'team_name' in df.columns:
                for team_name in df['team_name'].unique():
                    if team_name in team_names_to_ids:
                        # Convert to int if it's a string
                        team_id = team_names_to_ids[team_name]
                        if isinstance(team_id, str):
                            try:
                                team_id = int(team_id)
                            except ValueError:
                                # Skip if we can't convert to int
                                continue
                        team_ids.add(team_id)

        # Get upcoming games for these teams
        upcoming_games = []
        for team_id in team_ids:
            team_games = self.analyzer.get_upcoming_games(team_id, days=days_ahead)
            upcoming_games.extend(team_games)

        # Create matchup predictions
        matchup_predictions = []

        for game in upcoming_games:
            # Skip games already processed (each game appears once per team)
            if any(m['game_id'] == game['game_id'] for m in matchup_predictions):
                continue

            home_team_name = game['home_name']
            away_team_name = game['away_name']

            # Get pitchers for the game if available (often not available for future games)
            home_pitcher_id = None
            away_pitcher_id = None
            home_pitcher_name = "Unknown"
            away_pitcher_name = "Unknown"

            # If probable pitchers are known, get their IDs
            # Note: This information might not be available far in advance

            # Find batters from these teams in our top_batters list
            home_batters = top_batters[
                top_batters['team_name'] == home_team_name] if 'team_name' in top_batters.columns else pd.DataFrame()
            away_batters = top_batters[
                top_batters['team_name'] == away_team_name] if 'team_name' in top_batters.columns else pd.DataFrame()

            # Find pitchers from these teams in our top_pitchers list
            home_pitchers = top_pitchers[
                top_pitchers['team_name'] == home_team_name] if 'team_name' in top_pitchers.columns else pd.DataFrame()
            away_pitchers = top_pitchers[
                top_pitchers['team_name'] == away_team_name] if 'team_name' in top_pitchers.columns else pd.DataFrame()

            # Create a matchup prediction
            matchup = {
                'game_id': game['game_id'],
                'date': game['game_date'],
                'home_team': home_team_name,
                'away_team': away_team_name,
                'venue': game.get('venue_name', "Unknown"),
                'home_pitcher_id': home_pitcher_id,
                'home_pitcher_name': home_pitcher_name,
                'away_pitcher_id': away_pitcher_id,
                'away_pitcher_name': away_pitcher_name,
                'home_hot_batters': len(home_batters),
                'away_hot_batters': len(away_batters),
                'home_effective_pitchers': len(home_pitchers),
                'away_effective_pitchers': len(away_pitchers)
            }

            # Calculate an overall "interest" score for the matchup based on hot batters and pitchers
            interest_score = 0

            # More hot batters and effective pitchers make a game more interesting
            interest_score += matchup['home_hot_batters'] * 0.5
            interest_score += matchup['away_hot_batters'] * 0.5
            interest_score += matchup['home_effective_pitchers'] * 0.5
            interest_score += matchup['away_effective_pitchers'] * 0.5

            # If we know the starting pitchers, add their effectiveness scores
            # (This assumes we've gotten this info from somewhere)

            matchup['interest_score'] = interest_score
            matchup_predictions.append(matchup)

        # Convert to DataFrame
        if matchup_predictions:
            predictions_df = pd.DataFrame(matchup_predictions)
            # Sort by interest score
            predictions_df = predictions_df.sort_values('interest_score', ascending=False)
            self.matchup_predictions = predictions_df
            return predictions_df
        else:
            return pd.DataFrame()  # Return empty DataFrame if no predictions