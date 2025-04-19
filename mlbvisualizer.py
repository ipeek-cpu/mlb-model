import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os


# Visualization class to complement the MLBAnalyzer and MLBModelAnalysis
class MLBVisualizer:
    def __init__(self, model_analysis):
        """Initialize with an MLBModelAnalysis instance"""
        self.model = model_analysis

    def visualize_hot_batters(self, n=15, save_path=None):
        """
        Create a visualization of the hottest batters

        Args:
            n (int): Number of top batters to display
            save_path (str): Path to save the figure (if None, will show instead)

        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        if self.model.hot_batters is None or self.model.hot_batters.empty:
            print("No hot batters data available. Run analyze_hot_batters first.")
            return None

        # Get top n batters
        top_batters = self.model.hot_batters.head(n).copy()

        # Choose which metrics to visualize
        metrics = []
        if 'batting_avg' in top_batters.columns:
            metrics.append('batting_avg')
        if 'weighted_avg' in top_batters.columns:
            metrics.append('weighted_avg')
        if 'ops' in top_batters.columns:
            metrics.append('ops')
        if 'slugging' in top_batters.columns:
            metrics.append('slugging')

        if not metrics:
            print("No visualization metrics available in the data.")
            return None

        # Create a subplot for each metric
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=[metric.replace('_', ' ').title() for metric in metrics],
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        # Add bars for each metric
        for i, metric in enumerate(metrics):
            # Sort data for this specific metric
            sorted_data = top_batters.sort_values(metric, ascending=False)

            # Create a horizontal bar for this metric
            fig.add_trace(
                go.Bar(
                    y=sorted_data['player_name'],
                    x=sorted_data[metric],
                    orientation='h',
                    name=metric.replace('_', ' ').title(),
                    marker_color='rgba(0, 123, 255, 0.6)' if i % 2 == 0 else 'rgba(220, 53, 69, 0.6)'
                ),
                row=i + 1, col=1
            )

            # Add team name as text
            if 'team_name' in sorted_data.columns:
                teams = sorted_data['team_name']
                fig.add_trace(
                    go.Scatter(
                        y=sorted_data['player_name'],
                        x=[max(sorted_data[metric]) * 0.02] * len(sorted_data),  # Slight offset
                        mode='text',
                        text=teams,
                        textposition='middle right',
                        showlegend=False,
                        textfont=dict(size=10, color='rgba(0,0,0,0.6)')
                    ),
                    row=i + 1, col=1
                )

        # Update layout
        fig.update_layout(
            title={
                'text': f'Top {n} Hottest MLB Batters - Last 10 Games',
                'x': 0.5,
                'font': dict(size=20)
            },
            height=200 + (len(metrics) * 300),  # Adjust height based on number of metrics
            width=900,
            barmode='group',
            bargap=0.15,
            showlegend=False
        )

        # Add a colored background based on the hotness score if available
        if 'hotness_score' in top_batters.columns:
            for i in range(len(metrics)):
                sorted_data = top_batters.sort_values(metrics[i], ascending=False)
                max_score = sorted_data['hotness_score'].max()
                min_score = sorted_data['hotness_score'].min()

                # Normalize scores between 0 and 1
                if max_score > min_score:
                    normalized_scores = (sorted_data['hotness_score'] - min_score) / (max_score - min_score)
                else:
                    normalized_scores = [0.5] * len(sorted_data)

                # Add colored backgrounds to y-axis labels
                for j, (name, score) in enumerate(zip(sorted_data['player_name'], normalized_scores)):
                    # Create color based on hotness (red = hot, blue = cold)
                    r = int(255 * score)
                    b = int(255 * (1 - score))
                    color = f'rgba({r}, 100, {b}, 0.1)'

                    fig.add_shape(
                        type="rect",
                        xref=f"x{i + 1}", yref=f"y{i + 1}",
                        x0=0, x1=1,
                        y0=j - 0.4, y1=j + 0.4,
                        fillcolor=color,
                        layer="below",
                        line_width=0,
                    )

        # Show or save the figure
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            fig.write_html(save_path)

        return fig

    def visualize_effective_pitchers(self, n=15, save_path=None):
        """
        Create a visualization of the most effective pitchers

        Args:
            n (int): Number of top pitchers to display
            save_path (str): Path to save the figure (if None, will show instead)

        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        if self.model.effective_pitchers is None or self.model.effective_pitchers.empty:
            print("No effective pitchers data available. Run analyze_pitchers first.")
            return None

        # Get top n pitchers
        top_pitchers = self.model.effective_pitchers.head(n).copy()

        # Choose which metrics to visualize
        metrics = []
        if 'era' in top_pitchers.columns:
            metrics.append(('era', True))  # True means lower is better
        if 'whip' in top_pitchers.columns:
            metrics.append(('whip', True))  # True means lower is better
        if 'k_per_9' in top_pitchers.columns:
            metrics.append(('k_per_9', False))  # False means higher is better
        if 'bb_per_9' in top_pitchers.columns:
            metrics.append(('bb_per_9', True))  # True means lower is better

        if not metrics:
            print("No visualization metrics available in the data.")
            return None

        # Create a subplot for each metric
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=[metric[0].replace('_', ' ').upper() for metric in metrics],
            shared_xaxes=True,
            vertical_spacing=0.1
        )

        # Add bars for each metric
        for i, (metric, lower_is_better) in enumerate(metrics):
            # Sort data for this specific metric (sometimes lower is better)
            sorted_data = top_pitchers.sort_values(metric, ascending=lower_is_better)

            # Handle infinity values in ERA or other metrics
            if sorted_data[metric].isin([float('inf')]).any():
                sorted_data.loc[sorted_data[metric] == float('inf'), metric] = sorted_data[metric].replace(float('inf'),
                                                                                                           np.nan).max() * 1.5

            # Limit to top n entries
            sorted_data = sorted_data.head(n)

            # Choose color based on metric (red for ERA/WHIP, blue for strikeouts)
            color = 'rgba(220, 53, 69, 0.6)' if lower_is_better else 'rgba(0, 123, 255, 0.6)'

            # Create a horizontal bar for this metric
            fig.add_trace(
                go.Bar(
                    y=sorted_data['player_name'],
                    x=sorted_data[metric],
                    orientation='h',
                    name=metric.replace('_', ' ').upper(),
                    marker_color=color
                ),
                row=i + 1, col=1
            )

            # Add team name as text
            if 'team_name' in sorted_data.columns:
                teams = sorted_data['team_name']
                fig.add_trace(
                    go.Scatter(
                        y=sorted_data['player_name'],
                        x=[max(sorted_data[metric]) * 0.02] * len(sorted_data),  # Slight offset
                        mode='text',
                        text=teams,
                        textposition='middle right',
                        showlegend=False,
                        textfont=dict(size=10, color='rgba(0,0,0,0.6)')
                    ),
                    row=i + 1, col=1
                )

        # Update layout
        fig.update_layout(
            title={
                'text': f'Top {n} Most Effective MLB Pitchers - Last 10 Games',
                'x': 0.5,
                'font': dict(size=20)
            },
            height=200 + (len(metrics) * 300),  # Adjust height based on number of metrics
            width=900,
            barmode='group',
            bargap=0.15,
            showlegend=False
        )

        # Add a colored background based on the effectiveness score if available
        if 'effectiveness_score' in top_pitchers.columns:
            for i, (metric, lower_is_better) in enumerate(metrics):
                sorted_data = top_pitchers.sort_values(metric, ascending=lower_is_better).head(n)
                max_score = sorted_data['effectiveness_score'].max()
                min_score = sorted_data['effectiveness_score'].min()

                # Normalize scores between 0 and 1
                if max_score > min_score:
                    normalized_scores = (sorted_data['effectiveness_score'] - min_score) / (max_score - min_score)
                else:
                    normalized_scores = [0.5] * len(sorted_data)

                # Add colored backgrounds to y-axis labels
                for j, (name, score) in enumerate(zip(sorted_data['player_name'], normalized_scores)):
                    # Create color based on effectiveness (green = effective, red = less effective)
                    g = int(255 * score)
                    r = int(255 * (1 - score))
                    color = f'rgba({r}, {g}, 100, 0.1)'

                    fig.add_shape(
                        type="rect",
                        xref=f"x{i + 1}", yref=f"y{i + 1}",
                        x0=0, x1=1,
                        y0=j - 0.4, y1=j + 0.4,
                        fillcolor=color,
                        layer="below",
                        line_width=0,
                    )

        # Show or save the figure
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            fig.write_html(save_path)

        return fig

    def visualize_matchup_predictions(self, n=10, save_path=None):
        """
        Create a visualization of the most interesting upcoming matchups

        Args:
            n (int): Number of top matchups to display
            save_path (str): Path to save the figure (if None, will show instead)

        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        if self.model.matchup_predictions is None or self.model.matchup_predictions.empty:
            print("No matchup predictions available. Run predict_batter_pitcher_matchups first.")
            return None

        # Get top n matchups
        top_matchups = self.model.matchup_predictions.head(n).copy()

        # Create the figure
        fig = go.Figure()

        # Create a custom text for each matchup
        matchup_texts = []
        for _, row in top_matchups.iterrows():
            text = f"{row['date']}: {row['away_team']} @ {row['home_team']}<br>"
            text += f"Hot Batters: {row['away_hot_batters']} (Away) vs {row['home_hot_batters']} (Home)<br>"
            text += f"Effective Pitchers: {row['away_effective_pitchers']} (Away) vs {row['home_effective_pitchers']} (Home)"
            matchup_texts.append(text)

        # Create matchup labels
        matchup_labels = [f"{row['away_team']} @ {row['home_team']}" for _, row in top_matchups.iterrows()]

        # Add horizontal bar for interest score
        fig.add_trace(
            go.Bar(
                y=matchup_labels,
                x=top_matchups['interest_score'],
                orientation='h',
                marker_color='rgba(0, 123, 255, 0.6)',
                hovertext=matchup_texts,
                hoverinfo='text',
                name='Interest Score'
            )
        )

        # Add text for the game date
        fig.add_trace(
            go.Scatter(
                y=matchup_labels,
                x=[0] * len(top_matchups),
                mode='text',
                text=top_matchups['date'],
                textposition='middle left',
                showlegend=False,
                textfont=dict(size=12, color='rgba(0,0,0,0.8)')
            )
        )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Top Upcoming MLB Matchups by Interest Level',
                'x': 0.5,
                'font': dict(size=20)
            },
            height=400 + (n * 40),  # Adjust height based on number of matchups
            width=900,
            xaxis_title='Interest Score',
            margin=dict(l=150),
            showlegend=False
        )

        # Show or save the figure
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            fig.write_html(save_path)

        return fig

    def visualize_batter_trend(self, player_id, metric='batting_avg', last_n_games=10, save_path=None):
        """
        Create a visualization of a specific batter's performance trend

        Args:
            player_id (int): MLB player ID
            metric (str): Metric to visualize (batting_avg, ops, etc.)
            last_n_games (int): Number of recent games to show
            save_path (str): Path to save the figure (if None, will show instead)

        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        # Get player data from the analyzer
        player_data = self.model.analyzer.player_stats.get(player_id)

        if player_data is None:
            print(f"No data found for player ID {player_id}")
            return None

        # Get the hitting logs
        hitting_logs = player_data['logs']['hitting']

        if not hitting_logs:
            print(f"No hitting logs found for player {player_data['name']}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(hitting_logs)

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date')

        # Limit to the last N games
        df = df.tail(last_n_games)

        # Calculate the metric if it's not already present
        if metric == 'batting_avg' and 'batting_avg' not in df.columns:
            df['batting_avg'] = df['hits'] / df['atBats']

        if metric == 'ops' and 'ops' not in df.columns:
            if all(col in df.columns for col in ['hits', 'doubles', 'triples', 'homeRuns', 'atBats', 'baseOnBalls']):
                # Calculate slugging
                df['slugging'] = (df['hits'] + df['doubles'] + 2 * df['triples'] + 3 * df['homeRuns']) / df['atBats']
                # Calculate OBP
                df['obp'] = (df['hits'] + df['baseOnBalls']) / (df['atBats'] + df['baseOnBalls'])
                # Calculate OPS
                df['ops'] = df['obp'] + df['slugging']

        # Ensure the metric is present
        if metric not in df.columns:
            print(f"Metric {metric} not available in the data.")
            return None

        # Create the figure
        fig = go.Figure()

        # Add the trend line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(width=3, color='rgb(0, 123, 255)'),
                marker=dict(size=10)
            )
        )

        # Add opponent annotation
        if 'opponent' in df.columns:
            for i, row in df.iterrows():
                fig.add_annotation(
                    x=row['date'],
                    y=row[metric],
                    text=row['opponent'],
                    showarrow=False,
                    yshift=15,
                    font=dict(size=10)
                )

        # Calculate a trend line using rolling average
        if len(df) >= 3:
            df['rolling_avg'] = df[metric].rolling(window=3, min_periods=1).mean()

            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['rolling_avg'],
                    mode='lines',
                    name='3-Game Average',
                    line=dict(width=2, color='rgba(255, 0, 0, 0.7)', dash='dash')
                )
            )

        # Update layout
        fig.update_layout(
            title={
                'text': f"{player_data['name']} - {metric.replace('_', ' ').title()} Trend (Last {last_n_games} Games)",
                'x': 0.5,
                'font': dict(size=20)
            },
            xaxis_title='Game Date',
            yaxis_title=metric.replace('_', ' ').title(),
            height=500,
            width=900,
            hovermode='x unified'
        )

        # Show or save the figure
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            fig.write_html(save_path)

        return fig

    def visualize_pitcher_trend(self, player_id, metric='era', last_n_games=10, save_path=None):
        """
        Create a visualization of a specific pitcher's performance trend

        Args:
            player_id (int): MLB player ID
            metric (str): Metric to visualize (era, whip, k_per_9, etc.)
            last_n_games (int): Number of recent games to show
            save_path (str): Path to save the figure (if None, will show instead)

        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        # Get player data from the analyzer
        player_data = self.model.analyzer.player_stats.get(player_id)

        if player_data is None:
            print(f"No data found for player ID {player_id}")
            return None

        # Get the pitching logs
        pitching_logs = player_data['logs']['pitching']

        if not pitching_logs:
            print(f"No pitching logs found for player {player_data['name']}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(pitching_logs)

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date')

        # Limit to the last N games
        df = df.tail(last_n_games)

        # Process innings pitched
        if 'inningsPitched' in df.columns:
            # Convert innings pitched to numeric (handle fractional innings)
            df['innings_pitched'] = df['inningsPitched'].apply(
                lambda x: float(str(x).replace('.1', '.33').replace('.2', '.67'))
                if not pd.isna(x) else 0.0
            )

        # Calculate metrics if not already present
        if metric == 'era' and 'era' not in df.columns:
            if all(col in df.columns for col in ['innings_pitched', 'earnedRuns']):
                df['era'] = (df['earnedRuns'] * 9) / df['innings_pitched']

        if metric == 'whip' and 'whip' not in df.columns:
            if all(col in df.columns for col in ['innings_pitched', 'baseOnBalls', 'hits']):
                df['whip'] = (df['baseOnBalls'] + df['hits']) / df['innings_pitched']

        if metric == 'k_per_9' and 'k_per_9' not in df.columns:
            if all(col in df.columns for col in ['innings_pitched', 'strikeOuts']):
                df['k_per_9'] = (df['strikeOuts'] * 9) / df['innings_pitched']

        # Replace infinity values with NaN
        if metric in df.columns:
            df[metric] = df[metric].replace([float('inf'), -float('inf')], np.nan)

        # Ensure the metric is present
        if metric not in df.columns:
            print(f"Metric {metric} not available in the data.")
            return None

        # Create the figure
        fig = go.Figure()

        # Add the trend line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').upper(),
                line=dict(width=3, color='rgb(220, 53, 69)'),
                marker=dict(size=10)
            )
        )

        # Add opponent annotation
        if 'opponent' in df.columns:
            for i, row in df.iterrows():
                fig.add_annotation(
                    x=row['date'],
                    y=row[metric],
                    text=row['opponent'],
                    showarrow=False,
                    yshift=15,
                    font=dict(size=10)
                )

        # Calculate a trend line using rolling average
        if len(df) >= 3:
            df['rolling_avg'] = df[metric].rolling(window=3, min_periods=1).mean()

            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['rolling_avg'],
                    mode='lines',
                    name='3-Game Average',
                    line=dict(width=2, color='rgba(0, 0, 255, 0.7)', dash='dash')
                )
            )

        # Update layout
        fig.update_layout(
            title={
                'text': f"{player_data['name']} - {metric.replace('_', ' ').upper()} Trend (Last {last_n_games} Games)",
                'x': 0.5,
                'font': dict(size=20)
            },
            xaxis_title='Game Date',
            yaxis_title=metric.replace('_', ' ').upper(),
            height=500,
            width=900,
            hovermode='x unified'
        )

        # For ERA and WHIP, lower is better, so invert the y-axis
        if metric in ['era', 'whip']:
            fig.update_layout(yaxis=dict(autorange="reversed"))

        # Show or save the figure
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            fig.write_html(save_path)

        return fig

    def visualize_team_performance(self, team_id, metric='batting_avg', last_n_games=10, save_path=None):
        """
        Create a visualization of a team's overall batting or pitching performance

        Args:
            team_id (int): MLB team ID
            metric (str): Metric to visualize (batting_avg, era, etc.)
            last_n_games (int): Number of recent games to show
            save_path (str): Path to save the figure (if None, will show instead)

        Returns:
            plotly.graph_objects.Figure: The created figure
        """
        # Get team name
        team_name = self.model.analyzer.teams.get(team_id, {}).get('name', f"Team {team_id}")

        # Get all players for this team
        team_player_ids = [player_id for player_id, data in self.model.analyzer.player_stats.items()
                           if data['team_id'] == team_id]

        if not team_player_ids:
            print(f"No players found for team {team_name}")
            return None

        # Determine if we're looking at batting or pitching metric
        batting_metrics = ['batting_avg', 'ops', 'slugging', 'obp']
        pitching_metrics = ['era', 'whip', 'k_per_9', 'bb_per_9']

        is_batting = metric in batting_metrics
        is_pitching = metric in pitching_metrics

        if not (is_batting or is_pitching):
            print(f"Unsupported metric: {metric}")
            return None

        # Collect all game logs for the team
        game_logs = []

        for player_id in team_player_ids:
            player_data = self.model.analyzer.player_stats.get(player_id)
            if player_data:
                if is_batting and player_data['logs']['hitting']:
                    for game in player_data['logs']['hitting']:
                        game_copy = game.copy()
                        game_copy['player_id'] = player_id
                        game_copy['player_name'] = player_data['name']
                        game_logs.append(game_copy)
                elif is_pitching and player_data['logs']['pitching']:
                    for game in player_data['logs']['pitching']:
                        game_copy = game.copy()
                        game_copy['player_id'] = player_id
                        game_copy['player_name'] = player_data['name']
                        game_logs.append(game_copy)

        if not game_logs:
            print(f"No {('batting' if is_batting else 'pitching')} game logs found for team {team_name}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(game_logs)

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date')

        # Get unique game dates
        unique_dates = df['date'].unique()

        # Get the last N game dates
        recent_dates = sorted(unique_dates)[-last_n_games:]

        # Filter to recent games
        df = df[df['date'].isin(recent_dates)]

        # Group by date and calculate the team average for the metric
        team_avg_by_date = []

        for date in recent_dates:
            date_games = df[df['date'] == date]

            if is_batting:
                # For batting average
                if metric == 'batting_avg':
                    total_hits = date_games['hits'].sum()
                    total_at_bats = date_games['atBats'].sum()

                    if total_at_bats > 0:
                        team_avg = total_hits / total_at_bats
                    else:
                        team_avg = np.nan

                elif metric in ['ops', 'slugging', 'obp']:
                    # These are more complex, we could calculate them if needed
                    # For now, take the average of individual player metrics
                    if metric in date_games.columns:
                        team_avg = date_games[metric].mean()
                    else:
                        team_avg = np.nan

            elif is_pitching:
                # For ERA
                if metric == 'era':
                    if 'inningsPitched' in date_games.columns and 'earnedRuns' in date_games.columns:
                        # Convert innings pitched to numeric
                        innings = date_games['inningsPitched'].apply(
                            lambda x: float(str(x).replace('.1', '.33').replace('.2', '.67'))
                            if not pd.isna(x) else 0.0
                        ).sum()

                        earned_runs = date_games['earnedRuns'].sum()

                        if innings > 0:
                            team_avg = (earned_runs * 9) / innings
                        else:
                            team_avg = np.nan
                    else:
                        team_avg = np.nan

                # For WHIP
                elif metric == 'whip':
                    if all(col in date_games.columns for col in ['inningsPitched', 'baseOnBalls', 'hits']):
                        # Convert innings pitched to numeric
                        innings = date_games['inningsPitched'].apply(
                            lambda x: float(str(x).replace('.1', '.33').replace('.2', '.67'))
                            if not pd.isna(x) else 0.0
                        ).sum()

                        walks = date_games['baseOnBalls'].sum()
                        hits = date_games['hits'].sum()

                        if innings > 0:
                            team_avg = (walks + hits) / innings
                        else:
                            team_avg = np.nan
                    else:
                        team_avg = np.nan

                # For K/9 or BB/9
                elif metric in ['k_per_9', 'bb_per_9']:
                    stat_key = 'strikeOuts' if metric == 'k_per_9' else 'baseOnBalls'

                    if all(col in date_games.columns for col in ['inningsPitched', stat_key]):
                        # Convert innings pitched to numeric
                        innings = date_games['inningsPitched'].apply(
                            lambda x: float(str(x).replace('.1', '.33').replace('.2', '.67'))
                            if not pd.isna(x) else 0.0
                        ).sum()

                        stat_total = date_games[stat_key].sum()

                        if innings > 0:
                            team_avg = (stat_total * 9) / innings
                        else:
                            team_avg = np.nan
                    else:
                        team_avg = np.nan

            # Get opponent name for annotation
            opponent = None
            if 'opponent' in date_games.columns:
                opponent_counts = date_games['opponent'].value_counts()
                if not opponent_counts.empty:
                    opponent = opponent_counts.index[0]

            # Store in our results
            team_avg_by_date.append({
                'date': date,
                'value': team_avg,
                'opponent': opponent
            })

        # Convert to DataFrame
        team_avg_df = pd.DataFrame(team_avg_by_date)

        # Remove any NaN values
        team_avg_df = team_avg_df.dropna(subset=['value'])

        if team_avg_df.empty:
            print(f"No valid {metric} data found for team {team_name}")
            return None

        # Create the figure
        fig = go.Figure()

        # Add the trend line
        fig.add_trace(
            go.Scatter(
                x=team_avg_df['date'],
                y=team_avg_df['value'],
                mode='lines+markers',
                name=metric.replace('_', ' ').title() if is_batting else metric.replace('_', ' ').upper(),
                line=dict(width=3, color='rgb(0, 123, 255)' if is_batting else 'rgb(220, 53, 69)'),
                marker=dict(size=10)
            )
        )

        # Add opponent annotation
        if 'opponent' in team_avg_df.columns:
            for i, row in team_avg_df.iterrows():
                if not pd.isna(row['opponent']):
                    fig.add_annotation(
                        x=row['date'],
                        y=row['value'],
                        text=row['opponent'],
                        showarrow=False,
                        yshift=15,
                        font=dict(size=10)
                    )

        # Calculate a trend line using rolling average
        if len(team_avg_df) >= 3:
            team_avg_df['rolling_avg'] = team_avg_df['value'].rolling(window=3, min_periods=1).mean()

            fig.add_trace(
                go.Scatter(
                    x=team_avg_df['date'],
                    y=team_avg_df['rolling_avg'],
                    mode='lines',
                    name='3-Game Average',
                    line=dict(width=2, color='rgba(255, 0, 0, 0.7)' if is_batting else 'rgba(0, 0, 255, 0.7)',
                              dash='dash')
                )
            )

        # Get metric name for display
        if is_batting:
            metric_display = metric.replace('_', ' ').title()
        else:
            metric_display = metric.replace('_', ' ').upper()

        # Update layout
        fig.update_layout(
            title={
                'text': f"{team_name} - Team {metric_display} (Last {last_n_games} Games)",
                'x': 0.5,
                'font': dict(size=20)
            },
            xaxis_title='Game Date',
            yaxis_title=metric_display,
            height=500,
            width=900,
            hovermode='x unified'
        )

        # For pitching metrics like ERA and WHIP, lower is better, so invert the y-axis
        if is_pitching and metric in ['era', 'whip', 'bb_per_9']:
            fig.update_layout(yaxis=dict(autorange="reversed"))

        # Show or save the figure
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            fig.write_html(save_path)

        return fig