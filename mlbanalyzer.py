import statsapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time


class MLBAnalyzer:
    """
    This class handles the collection and organization of MLB player data,
    primarily focused on retrieving game logs and stats for analysis.
    """

    def __init__(self):
        """Initialize the MLBAnalyzer with empty data structures"""
        self.teams = self.get_all_teams()
        self.team_rosters = {}
        self.player_stats = {}
        self.batter_data = None
        self.pitcher_data = None

    def get_all_teams(self):
        """Get a dictionary of all MLB teams"""
        teams_data = statsapi.get("teams", {"sportId": 1})
        teams = {}

        if "teams" in teams_data:
            for team in teams_data["teams"]:
                teams[team["id"]] = {
                    "name": team["name"],
                    "abbreviation": team.get("abbreviation", ""),
                    "teamName": team.get("teamName", ""),
                    "division": team.get("division", {}).get("name", ""),
                    "league": team.get("league", {}).get("name", "")
                }

        return teams

    def get_team_roster(self, team_id, roster_type="active"):
        """Get the roster for a specific team"""
        roster_data = statsapi.get("team_roster", {
            "teamId": team_id,
            "rosterType": roster_type
        })

        players = []
        if "roster" in roster_data:
            for player in roster_data["roster"]:
                players.append({
                    "id": player["person"]["id"],
                    "name": player["person"]["fullName"],
                    "position": player["position"]["abbreviation"],
                    "jersey": player.get("jerseyNumber", "")
                })

        return players

    def get_player_game_logs(self, player_id, num_games=20):
        """Get recent game logs for a player"""
        season = datetime.now().year

        params = {
            "personIds": player_id,
            "hydrate": f"stats(group=[hitting,pitching],type=[gameLog],season={season})"
        }

        game_data = statsapi.get("people", params)

        result = {
            "hitting": [],
            "pitching": []
        }

        if "people" in game_data and len(game_data["people"]) > 0:
            person = game_data["people"][0]

            if "stats" in person:
                for stat_group in person["stats"]:
                    if "group" in stat_group and "splits" in stat_group:
                        group_type = stat_group["group"]["displayName"].lower()

                        if group_type in result and len(stat_group["splits"]) > 0:
                            # Get the most recent games
                            recent_games = stat_group["splits"][:num_games]

                            for game in recent_games:
                                game_info = {
                                    "date": game.get("date", ""),
                                    "opponent": game.get("opponent", {}).get("name", "Unknown"),
                                    "opponent_id": game.get("opponent", {}).get("id", 0),
                                    "is_home": game.get("isHome", False),
                                    "game_id": game.get("game", {}).get("gamePk", "")
                                }

                                if "stat" in game:
                                    game_info.update(game["stat"])

                                result[group_type].append(game_info)

        return result

    def get_upcoming_games(self, team_id, days=14):
        """Get upcoming games for a team"""
        today = datetime.now()
        start_date = today.strftime("%m/%d/%Y")
        end_date = (today + timedelta(days=days)).strftime("%m/%d/%Y")

        schedule = statsapi.schedule(
            team=team_id,
            start_date=start_date,
            end_date=end_date
        )

        return schedule

    def collect_team_data(self, team_id, games_back=20):
        """Collect game logs for all players on a team"""
        # Get team roster if not already cached
        if team_id not in self.team_rosters:
            self.team_rosters[team_id] = self.get_team_roster(team_id)

        # Collect data for each player
        for player in self.team_rosters[team_id]:
            player_id = player["id"]

            if player_id not in self.player_stats:
                print(f"  Getting data for {player['name']}...")
                self.player_stats[player_id] = {
                    "name": player["name"],
                    "position": player["position"],
                    "team_id": team_id,
                    "team_name": self.teams[team_id]["name"],
                    "logs": self.get_player_game_logs(player_id, games_back)
                }

        # Save data to CSV files
        self.save_team_data_to_csv(team_id)

    def save_team_data_to_csv(self, team_id):
        """Save team data to CSV files"""
        team_name = self.teams[team_id]["name"].replace(" ", "_")
        hitting_data = []
        pitching_data = []

        # Create data directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")

        # Collect all hitting and pitching data for the team
        for player_id, player_data in self.player_stats.items():
            if player_data["team_id"] == team_id:
                # Process hitting data
                for game in player_data["logs"]["hitting"]:
                    game_copy = game.copy()
                    game_copy["player_id"] = player_id
                    game_copy["player_name"] = player_data["name"]
                    game_copy["position"] = player_data["position"]
                    hitting_data.append(game_copy)

                # Process pitching data
                for game in player_data["logs"]["pitching"]:
                    game_copy = game.copy()
                    game_copy["player_id"] = player_id
                    game_copy["player_name"] = player_data["name"]
                    game_copy["position"] = player_data["position"]
                    pitching_data.append(game_copy)

        # Create DataFrames
        hitting_df = pd.DataFrame(hitting_data)
        pitching_df = pd.DataFrame(pitching_data)

        # Save to CSV
        if not hitting_df.empty:
            hitting_df.to_csv(f"data/{team_name}_hitting.csv", index=False)
        if not pitching_df.empty:
            pitching_df.to_csv(f"data/{team_name}_pitching.csv", index=False)

    def collect_all_team_data(self, min_at_bats=10, min_innings_pitched=5):
        """Collect data for all MLB teams"""
        for team_id in self.teams:
            print(f"Collecting data for {self.teams[team_id]['name']}...")
            self.collect_team_data(team_id)
            # Pause briefly to avoid rate limiting
            time.sleep(1)

        # Create consolidated DataFrames for all batters and pitchers
        self.consolidate_player_data(min_at_bats, min_innings_pitched)

    def consolidate_player_data(self, min_at_bats=10, min_innings_pitched=5):
        """Consolidate player data into DataFrames for analysis"""
        all_batting_data = []
        all_pitching_data = []

        for player_id, player_data in self.player_stats.items():
            # Process batting data
            for game in player_data["logs"]["hitting"]:
                game_copy = game.copy()
                game_copy["player_id"] = player_id
                game_copy["player_name"] = player_data["name"]
                game_copy["position"] = player_data["position"]
                game_copy["team_id"] = player_data["team_id"]
                game_copy["team_name"] = player_data["team_name"]
                all_batting_data.append(game_copy)

            # Process pitching data
            for game in player_data["logs"]["pitching"]:
                game_copy = game.copy()
                game_copy["player_id"] = player_id
                game_copy["player_name"] = player_data["name"]
                game_copy["position"] = player_data["position"]
                game_copy["team_id"] = player_data["team_id"]
                game_copy["team_name"] = player_data["team_name"]
                all_pitching_data.append(game_copy)

        # Create DataFrames
        self.batter_data = pd.DataFrame(all_batting_data)
        self.pitcher_data = pd.DataFrame(all_pitching_data)

        # Save consolidated data
        if not self.batter_data.empty:
            self.batter_data.to_csv("data/all_batters.csv", index=False)
        if not self.pitcher_data.empty:
            self.pitcher_data.to_csv("data/all_pitchers.csv", index=False)

        # Apply filters
        if not self.batter_data.empty:
            # Convert at_bats to numeric
            if 'atBats' in self.batter_data.columns:
                self.batter_data['atBats'] = pd.to_numeric(self.batter_data['atBats'], errors='coerce')
                # Group by player and filter based on total at bats
                batter_totals = self.batter_data.groupby(['player_id', 'player_name', 'team_name'])[
                    'atBats'].sum().reset_index()
                qualified_batters = batter_totals[batter_totals['atBats'] >= min_at_bats]
                self.batter_data = self.batter_data[self.batter_data['player_id'].isin(qualified_batters['player_id'])]

        if not self.pitcher_data.empty:
            # Filter pitchers with minimum innings pitched
            if 'inningsPitched' in self.pitcher_data.columns:
                # Convert innings pitched to numeric
                self.pitcher_data['inningsPitched'] = self.pitcher_data['inningsPitched'].apply(
                    lambda x: float(str(x).replace('.1', '.33').replace('.2', '.67')) if isinstance(x, str) else float(
                        x))

                # Group by pitcher and filter
                pitcher_totals = self.pitcher_data.groupby(['player_id', 'player_name', 'team_name'])[
                    'inningsPitched'].sum().reset_index()
                qualified_pitchers = pitcher_totals[pitcher_totals['inningsPitched'] >= min_innings_pitched]
                self.pitcher_data = self.pitcher_data[
                    self.pitcher_data['player_id'].isin(qualified_pitchers['player_id'])]

    def get_player_stats_summary(self, player_id, stat_type='hitting'):
        """Get a summary of a player's recent stats"""
        if player_id not in self.player_stats:
            return None

        player_data = self.player_stats[player_id]
        logs = player_data['logs'][stat_type]

        if not logs:
            return None

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(logs)

        # Calculate summary statistics
        summary = {
            'player_id': player_id,
            'player_name': player_data['name'],
            'position': player_data['position'],
            'team_name': player_data['team_name'],
            'games': len(df)
        }

        if stat_type == 'hitting':
            # Calculate batting stats
            if 'atBats' in df.columns:
                summary['at_bats'] = df['atBats'].sum()

            if 'hits' in df.columns:
                summary['hits'] = df['hits'].sum()

                if 'at_bats' in summary and summary['at_bats'] > 0:
                    summary['batting_avg'] = summary['hits'] / summary['at_bats']

            if all(col in df.columns for col in ['doubles', 'triples', 'homeRuns']):
                summary['doubles'] = df['doubles'].sum()
                summary['triples'] = df['triples'].sum()
                summary['home_runs'] = df['homeRuns'].sum()

                # Calculate slugging
                if 'at_bats' in summary and summary['at_bats'] > 0 and 'hits' in summary:
                    summary['slugging'] = (summary['hits'] + summary['doubles'] +
                                           2 * summary['triples'] + 3 * summary['home_runs']) / summary['at_bats']

            if 'baseOnBalls' in df.columns:
                summary['walks'] = df['baseOnBalls'].sum()

                # Calculate OBP
                if 'at_bats' in summary and 'hits' in summary:
                    summary['obp'] = (summary['hits'] + summary['walks']) / (summary['at_bats'] + summary['walks'])

                    # Calculate OPS
                    if 'slugging' in summary:
                        summary['ops'] = summary['obp'] + summary['slugging']

            if 'strikeOuts' in df.columns:
                summary['strikeouts'] = df['strikeOuts'].sum()

                if 'at_bats' in summary and summary['at_bats'] > 0:
                    summary['k_rate'] = summary['strikeouts'] / summary['at_bats']

        elif stat_type == 'pitching':
            # Calculate pitching stats
            if 'inningsPitched' in df.columns:
                # Convert innings pitched to numeric
                df['innings'] = df['inningsPitched'].apply(
                    lambda x: float(str(x).replace('.1', '.33').replace('.2', '.67'))
                    if isinstance(x, str) else float(x))

                summary['innings_pitched'] = df['innings'].sum()

            if 'earnedRuns' in df.columns and 'innings_pitched' in summary:
                summary['earned_runs'] = df['earnedRuns'].sum()

                if summary['innings_pitched'] > 0:
                    summary['era'] = (summary['earned_runs'] * 9) / summary['innings_pitched']

            if all(col in df.columns for col in ['baseOnBalls', 'hits']) and 'innings_pitched' in summary:
                summary['walks'] = df['baseOnBalls'].sum()
                summary['hits_allowed'] = df['hits'].sum()

                if summary['innings_pitched'] > 0:
                    summary['whip'] = (summary['walks'] + summary['hits_allowed']) / summary['innings_pitched']

            if 'strikeOuts' in df.columns and 'innings_pitched' in summary:
                summary['strikeouts'] = df['strikeOuts'].sum()

                if summary['innings_pitched'] > 0:
                    summary['k_per_9'] = (summary['strikeouts'] * 9) / summary['innings_pitched']

            if 'baseOnBalls' in df.columns and 'innings_pitched' in summary:
                if summary['innings_pitched'] > 0:
                    summary['bb_per_9'] = (summary['walks'] * 9) / summary['innings_pitched']

        return summary

    def get_team_stats_summary(self, team_id, stat_type='hitting'):
        """Get a summary of a team's recent stats"""
        if team_id not in self.teams:
            return None

        team_name = self.teams[team_id]['name']

        # Get all players for this team
        team_player_ids = [
            player_id for player_id, data in self.player_stats.items()
            if data['team_id'] == team_id
        ]

        if not team_player_ids:
            return None

        # Get stats for each player
        player_summaries = []

        for player_id in team_player_ids:
            summary = self.get_player_stats_summary(player_id, stat_type)
            if summary:
                player_summaries.append(summary)

        if not player_summaries:
            return None

        # Combine stats
        team_summary = {
            'team_id': team_id,
            'team_name': team_name,
            'players': len(player_summaries)
        }

        if stat_type == 'hitting':
            # Aggregate batting stats
            team_summary['at_bats'] = sum(s.get('at_bats', 0) for s in player_summaries)
            team_summary['hits'] = sum(s.get('hits', 0) for s in player_summaries)

            if team_summary['at_bats'] > 0:
                team_summary['batting_avg'] = team_summary['hits'] / team_summary['at_bats']

            team_summary['home_runs'] = sum(s.get('home_runs', 0) for s in player_summaries)
            team_summary['walks'] = sum(s.get('walks', 0) for s in player_summaries)
            team_summary['strikeouts'] = sum(s.get('strikeouts', 0) for s in player_summaries)

            if team_summary['at_bats'] > 0:
                team_summary['k_rate'] = team_summary['strikeouts'] / team_summary['at_bats']

        elif stat_type == 'pitching':
            # Aggregate pitching stats
            team_summary['innings_pitched'] = sum(s.get('innings_pitched', 0) for s in player_summaries)
            team_summary['earned_runs'] = sum(s.get('earned_runs', 0) for s in player_summaries)
            team_summary['hits_allowed'] = sum(s.get('hits_allowed', 0) for s in player_summaries)
            team_summary['walks'] = sum(s.get('walks', 0) for s in player_summaries)
            team_summary['strikeouts'] = sum(s.get('strikeouts', 0) for s in player_summaries)

            if team_summary['innings_pitched'] > 0:
                team_summary['era'] = (team_summary['earned_runs'] * 9) / team_summary['innings_pitched']
                team_summary['whip'] = (team_summary['hits_allowed'] + team_summary['walks']) / team_summary[
                    'innings_pitched']
                team_summary['k_per_9'] = (team_summary['strikeouts'] * 9) / team_summary['innings_pitched']
                team_summary['bb_per_9'] = (team_summary['walks'] * 9) / team_summary['innings_pitched']

        return team_summary