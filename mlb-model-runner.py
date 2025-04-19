import statsapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# Import our model components
from mlbanalyzer import MLBAnalyzer
from mlbmodelanalysis import MLBModelAnalysis
from mlbvisualizer import MLBVisualizer


def main():
    """Main function to run the MLB hot batter and pitching matchup analysis"""
    print("Starting MLB Analysis...")

    # Create output directories
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")

    # Initialize our analyzer
    analyzer = MLBAnalyzer()

    # Collect data for a subset of teams to start
    # Collecting all teams can take time due to API rate limits
    teams_to_analyze = [
        147,  # Yankees
        119,  # Dodgers
        111,  # Red Sox
        108,  # Angels
        143,  # Phillies
        142,  # Twins
        138,  # Cardinals
        133,  # Athletics
        146,  # Marlins
        114,  # Indians
    ]

    print(f"Collecting data for {len(teams_to_analyze)} teams...")
    for team_id in teams_to_analyze:
        team_name = analyzer.teams.get(team_id, {}).get('name', f"Team {team_id}")
        print(f"  Processing {team_name}...")
        analyzer.collect_team_data(team_id)
        # Pause briefly to avoid rate limiting
        time.sleep(1)

    # Consolidate all player data
    print("Consolidating player data...")
    analyzer.consolidate_player_data()

    # Initialize our model analysis
    model = MLBModelAnalysis(analyzer)

    # Analyze hot batters
    print("Analyzing hot batters...")
    hot_batters = model.analyze_hot_batters(min_games=5)

    if hot_batters is not None:
        print(f"Found {len(hot_batters)} qualified batters")
        print("Top 5 hottest batters:")
        if 'hotness_score' in hot_batters.columns:
            for _, row in hot_batters.head(5).iterrows():
                print(f"  {row['player_name']} ({row['team_name']}): {row['hotness_score']:.3f}")

    # Analyze effective pitchers
    print("Analyzing effective pitchers...")
    effective_pitchers = model.analyze_pitchers(min_games=3)

    if effective_pitchers is not None:
        print(f"Found {len(effective_pitchers)} qualified pitchers")
        print("Top 5 most effective pitchers:")
        if 'effectiveness_score' in effective_pitchers.columns:
            for _, row in effective_pitchers.head(5).iterrows():
                print(f"  {row['player_name']} ({row['team_name']}): {row['effectiveness_score']:.3f}")

    # Predict upcoming matchups
    print("Predicting upcoming matchups...")
    matchup_predictions = model.predict_batter_pitcher_matchups(days_ahead=7)

    if matchup_predictions is not None:
        print(f"Found {len(matchup_predictions)} upcoming matchups")
        print("Top 5 most interesting matchups:")
        if 'interest_score' in matchup_predictions.columns:
            for _, row in matchup_predictions.head(5).iterrows():
                print(f"  {row['date']}: {row['away_team']} @ {row['home_team']} - Score: {row['interest_score']:.3f}")

    # Initialize our visualizer
    viz = MLBVisualizer(model)

    # Create visualizations
    print("Creating visualizations...")

    # Hot batters visualization
    hot_batters_fig = viz.visualize_hot_batters(n=15, save_path="visualizations/hot_batters.html")
    if hot_batters_fig:
        print("Created hot batters visualization")

    # Effective pitchers visualization
    effective_pitchers_fig = viz.visualize_effective_pitchers(n=15, save_path="visualizations/effective_pitchers.html")
    if effective_pitchers_fig:
        print("Created effective pitchers visualization")

    # Matchup predictions visualization
    matchups_fig = viz.visualize_matchup_predictions(n=10, save_path="visualizations/upcoming_matchups.html")
    if matchups_fig:
        print("Created matchup predictions visualization")

    # Create individual player visualizations for a few top players
    if hot_batters is not None and len(hot_batters) > 0:
        top_batter_id = hot_batters.iloc[0]['player_id']
        batter_trend_fig = viz.visualize_batter_trend(
            top_batter_id, metric='batting_avg',
            save_path=f"visualizations/batter_{top_batter_id}_trend.html"
        )
        if batter_trend_fig:
            print(f"Created trend visualization for top batter (ID: {top_batter_id})")

    if effective_pitchers is not None and len(effective_pitchers) > 0:
        top_pitcher_id = effective_pitchers.iloc[0]['player_id']
        pitcher_trend_fig = viz.visualize_pitcher_trend(
            top_pitcher_id, metric='era',
            save_path=f"visualizations/pitcher_{top_pitcher_id}_trend.html"
        )
        if pitcher_trend_fig:
            print(f"Created trend visualization for top pitcher (ID: {top_pitcher_id})")

    # Create team performance visualizations for Yankees and Dodgers
    yankees_batting_fig = viz.visualize_team_performance(
        147, metric='batting_avg',
        save_path="visualizations/yankees_batting_trend.html"
    )
    if yankees_batting_fig:
        print("Created Yankees batting trend visualization")

    dodgers_era_fig = viz.visualize_team_performance(
        119, metric='era',
        save_path="visualizations/dodgers_era_trend.html"
    )
    if dodgers_era_fig:
        print("Created Dodgers ERA trend visualization")

    print("\nAnalysis complete! Results saved to data/ and visualizations/ directories.")
    print("Open the HTML files in a browser to view the interactive visualizations.")


if __name__ == "__main__":
    main()
