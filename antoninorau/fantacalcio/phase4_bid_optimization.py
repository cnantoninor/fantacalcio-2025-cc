"""
Fantacalcio Bid Optimization Module
Phase 4: Optimization & Bid Suggestion Engine

This module implements the final optimization engine that suggests optimal bids
to maximize team fantasy score while respecting budget and roster constraints.
"""

import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, LpBinary, lpSum, PULP_CBC_CMD, LpStatus
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RosterConstraints:
    """Define roster composition constraints"""
    goalkeepers: int = 3
    defenders: int = 7
    midfielders: int = 7
    forwards: int = 5
    total_budget: int = 500
    
    @property
    def total_players(self) -> int:
        return self.goalkeepers + self.defenders + self.midfielders + self.forwards


class BidOptimizer:
    """
    Optimization engine for Fantacalcio bid suggestions using integer programming
    """
    
    def __init__(self, roster_constraints: RosterConstraints):
        """
        Initialize the bid optimizer
        
        Args:
            roster_constraints: Roster composition and budget constraints
        """
        self.constraints = roster_constraints
        self.problem = None
        self.player_vars = {}
        self.bid_vars = {}
        self.solution = None
        
    def suggest_bids(self, player_df: pd.DataFrame, opponent_model, 
                    risk_tolerance: float = 0.7) -> pd.DataFrame:
        """
        Main function to suggest optimal bids for all players
        
        Args:
            player_df: DataFrame with player predictions and market prices
            opponent_model: OpponentModeler instance from Phase 3
            risk_tolerance: Risk tolerance (0-1, higher = more aggressive bidding)
            
        Returns:
            DataFrame with recommended bids and team composition
        """
        logger.info("Starting bid optimization process")
        
        # Prepare data for optimization
        players_prepared = self._prepare_optimization_data(player_df, opponent_model, risk_tolerance)
        
        # Create the optimization problem
        self._create_optimization_problem(players_prepared)
        
        # Solve the optimization
        solution_df = self._solve_optimization(players_prepared)
        
        # Post-process results
        final_recommendations = self._post_process_solution(solution_df, players_prepared)
        
        return final_recommendations
    
    def _prepare_optimization_data(self, player_df: pd.DataFrame, opponent_model, 
                                 risk_tolerance: float) -> pd.DataFrame:
        """Prepare player data for the optimization problem"""
        
        # Get market predictions if not already present
        if 'estimated_market_price' not in player_df.columns:
            player_df = opponent_model.estimate_market_price(player_df)
        
        # Calculate win probabilities if not present
        if 'win_probability' not in player_df.columns:
            player_df = opponent_model.simulate_auction_outcomes(player_df)
        
        players_opt = player_df.copy()
        
        # Calculate bid ranges based on risk tolerance
        players_opt['min_bid'] = players_opt['estimated_market_price'] * 0.8
        players_opt['max_bid'] = players_opt['estimated_market_price'] * (1 + risk_tolerance)
        
        # Calculate expected value considering win probability
        players_opt['expected_value'] = (
            players_opt['predicted_fantasy_score'] * 
            players_opt['win_probability']
        )
        
        # Calculate value per million spent
        players_opt['value_efficiency'] = (
            players_opt['expected_value'] / 
            players_opt['estimated_market_price'].replace(0, 1)
        )
        
        # Adjust for position scarcity (fewer players in position = higher value)
        position_counts = players_opt['position'].value_counts()
        players_opt['scarcity_multiplier'] = players_opt['position'].map(
            lambda pos: 1 + (1 / position_counts[pos]) if pos in position_counts else 1
        )
        
        players_opt['adjusted_value'] = (
            players_opt['expected_value'] * players_opt['scarcity_multiplier']
        )
        
        # Filter to top players by position to reduce problem size
        top_players = []
        for position in ['GK', 'DF', 'MF', 'FW']:
            pos_players = players_opt[players_opt['position'] == position].copy()
            # Take top players by value efficiency
            top_n = min(len(pos_players), self._get_position_limit(position) * 3)  # 3x limit for flexibility
            pos_top = pos_players.nlargest(top_n, 'value_efficiency')
            top_players.append(pos_top)
        
        players_final = pd.concat(top_players, ignore_index=True)
        
        logger.info(f"Prepared optimization data for {len(players_final)} players")
        return players_final
    
    def _get_position_limit(self, position: str) -> int:
        """Get roster limit for a position"""
        position_limits = {
            'GK': self.constraints.goalkeepers,
            'DF': self.constraints.defenders,
            'MF': self.constraints.midfielders,
            'FW': self.constraints.forwards
        }
        return position_limits.get(position, 0)
    
    def _create_optimization_problem(self, players_df: pd.DataFrame):
        """Create the integer programming optimization problem"""
        
        # Create the problem
        self.problem = LpProblem("Fantacalcio_Bid_Optimization", LpMaximize)
        
        # Create decision variables
        self.player_vars = {}  # Binary: whether to bid on player
        self.bid_vars = {}     # Continuous: how much to bid
        
        for idx, row in players_df.iterrows():
            player_id = f"player_{idx}"
            
            # Binary variable: select this player or not
            self.player_vars[player_id] = LpVariable(
                f"select_{player_id}", 
                cat='Binary'
            )
            
            # Continuous variable: bid amount
            self.bid_vars[player_id] = LpVariable(
                f"bid_{player_id}", 
                lowBound=0, 
                upBound=row['max_bid'],
                cat='Continuous'
            )
        
        # Objective function: maximize expected total fantasy score
        total_expected_score = lpSum([
            players_df.iloc[int(player_id.split('_')[1])]['adjusted_value'] * 
            self.player_vars[player_id] 
            for player_id in self.player_vars.keys()
        ])
        
        self.problem += total_expected_score
        
        # Add constraints
        self._add_constraints(players_df)
        
        logger.info(f"Created optimization problem with {len(self.player_vars)} players")
    
    def _add_constraints(self, players_df: pd.DataFrame):
        """Add constraints to the optimization problem"""
        
        # Budget constraint
        total_budget_used = lpSum([
            self.bid_vars[player_id] 
            for player_id in self.bid_vars.keys()
        ])
        self.problem += total_budget_used <= self.constraints.total_budget, "Budget_Constraint"
        
        # Position constraints
        for position in ['GK', 'DF', 'MF', 'FW']:
            position_players = players_df[players_df['position'] == position]
            if len(position_players) > 0:
                position_selection = lpSum([
                    self.player_vars[f"player_{idx}"]
                    for idx in position_players.index
                ])
                limit = self._get_position_limit(position)
                self.problem += position_selection == limit, f"{position}_Constraint"
        
        # Bid logic constraints: can only bid if player is selected
        for idx, row in players_df.iterrows():
            player_id = f"player_{idx}"
            
            # If player not selected, bid must be 0
            self.problem += self.bid_vars[player_id] <= row['max_bid'] * self.player_vars[player_id], f"Bid_Logic_{player_id}"
            
            # If player selected, bid must be at least minimum
            self.problem += self.bid_vars[player_id] >= row['min_bid'] * self.player_vars[player_id], f"Min_Bid_{player_id}"
        
        # Win probability constraints (heuristic)
        for idx, row in players_df.iterrows():
            player_id = f"player_{idx}"
            
            # Adjust win probability based on our bid vs market price
            # Higher bid = higher win probability (simplified model)
            required_bid_for_high_prob = row['estimated_market_price'] * 1.1
            self.problem += (
                self.bid_vars[player_id] >= required_bid_for_high_prob * self.player_vars[player_id] * 0.8
            ), f"Win_Prob_{player_id}"
        
        logger.info("Added optimization constraints")
    
    def _solve_optimization(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Solve the optimization problem and return results"""
        
        logger.info("Solving optimization problem...")
        
        # Solve the problem
        self.problem.solve(PULP_CBC_CMD(msg=0))
        
        # Check if solution was found
        if LpStatus[self.problem.status] != 'Optimal':
            logger.warning(f"Optimization status: {LpStatus[self.problem.status]}")
            return self._create_fallback_solution(players_df)
        
        # Extract solution
        solution_data = []
        total_budget_used = 0
        total_expected_score = 0
        
        for idx, row in players_df.iterrows():
            player_id = f"player_{idx}"
            
            selected = self.player_vars[player_id].varValue
            bid_amount = self.bid_vars[player_id].varValue
            
            if selected and selected > 0.5:  # Binary variable should be 1
                solution_data.append({
                    'player_name': row['player_name'],
                    'position': row['position'],
                    'predicted_fantasy_score': row['predicted_fantasy_score'],
                    'recommended_bid': max(1, round(bid_amount)),
                    'estimated_market_price': row['estimated_market_price'],
                    'win_probability': row['win_probability'],
                    'expected_value': row['expected_value'],
                    'value_efficiency': row['value_efficiency']
                })
                
                total_budget_used += bid_amount
                total_expected_score += row['adjusted_value']
        
        solution_df = pd.DataFrame(solution_data)
        
        if len(solution_df) > 0:
            logger.info(f"Optimization complete. Selected {len(solution_df)} players.")
            logger.info(f"Total budget used: {total_budget_used:.1f}/{self.constraints.total_budget}")
            logger.info(f"Expected total score: {total_expected_score:.1f}")
        else:
            logger.warning("No valid solution found, creating fallback")
            solution_df = self._create_fallback_solution(players_df)
        
        return solution_df
    
    def _create_fallback_solution(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Create a fallback solution using greedy algorithm"""
        
        logger.info("Creating fallback solution using greedy approach")
        
        # Sort players by value efficiency
        players_sorted = players_df.sort_values('value_efficiency', ascending=False)
        
        selected_players = []
        remaining_budget = self.constraints.total_budget
        position_counts = {'GK': 0, 'DF': 0, 'MF': 0, 'FW': 0}
        
        for _, player in players_sorted.iterrows():
            position = player['position']
            position_limit = self._get_position_limit(position)
            estimated_cost = player['estimated_market_price']
            
            # Check if we can afford this player and need this position
            if (position_counts[position] < position_limit and 
                estimated_cost <= remaining_budget):
                
                selected_players.append({
                    'player_name': player['player_name'],
                    'position': position,
                    'predicted_fantasy_score': player['predicted_fantasy_score'],
                    'recommended_bid': max(1, round(estimated_cost * 1.05)),  # Small premium
                    'estimated_market_price': estimated_cost,
                    'win_probability': player['win_probability'],
                    'expected_value': player['expected_value'],
                    'value_efficiency': player['value_efficiency']
                })
                
                remaining_budget -= estimated_cost
                position_counts[position] += 1
        
        return pd.DataFrame(selected_players)
    
    def _post_process_solution(self, solution_df: pd.DataFrame, 
                             players_df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the solution to add additional insights"""
        
        if len(solution_df) == 0:
            return solution_df
        
        # Add team composition summary
        position_summary = solution_df['position'].value_counts().to_dict()
        total_bid = solution_df['recommended_bid'].sum()
        total_expected_score = solution_df['predicted_fantasy_score'].sum()
        
        # Calculate risk metrics
        solution_df['bid_vs_market'] = (
            solution_df['recommended_bid'] / solution_df['estimated_market_price']
        )
        solution_df['risk_level'] = solution_df['bid_vs_market'].apply(
            lambda x: 'Low' if x < 1.05 else 'Medium' if x < 1.15 else 'High'
        )
        
        # Add alternative suggestions
        solution_df = self._add_alternative_suggestions(solution_df, players_df)
        
        # Sort by position and value
        position_order = {'GK': 1, 'DF': 2, 'MF': 3, 'FW': 4}
        solution_df['position_order'] = solution_df['position'].map(position_order)
        solution_df = solution_df.sort_values(['position_order', 'predicted_fantasy_score'], 
                                            ascending=[True, False])
        solution_df = solution_df.drop('position_order', axis=1)
        
        # Add summary statistics
        logger.info("\n=== BID OPTIMIZATION SUMMARY ===")
        logger.info(f"Total players selected: {len(solution_df)}")
        logger.info(f"Position breakdown: {position_summary}")
        logger.info(f"Total budget allocated: {total_bid:.1f}/{self.constraints.total_budget}")
        logger.info(f"Expected total fantasy score: {total_expected_score:.1f}")
        logger.info(f"Average win probability: {solution_df['win_probability'].mean():.2f}")
        
        return solution_df
    
    def _add_alternative_suggestions(self, solution_df: pd.DataFrame, 
                                   players_df: pd.DataFrame) -> pd.DataFrame:
        """Add alternative player suggestions for each position"""
        
        solution_df['alternatives'] = ""
        
        for position in ['GK', 'DF', 'MF', 'FW']:
            # Get selected players in this position
            selected_in_position = set(solution_df[solution_df['position'] == position]['player_name'])
            
            # Get top alternatives not selected
            pos_players = players_df[players_df['position'] == position]
            alternatives = pos_players[~pos_players['player_name'].isin(selected_in_position)]
            alternatives = alternatives.nlargest(3, 'value_efficiency')
            
            # Add alternatives to solution
            for idx, row in solution_df[solution_df['position'] == position].iterrows():
                alt_names = alternatives['player_name'].tolist()[:2]  # Top 2 alternatives
                solution_df.loc[idx, 'alternatives'] = ", ".join(alt_names)
        
        return solution_df
    
    def analyze_sensitivity(self, player_df: pd.DataFrame, opponent_model, 
                          risk_levels: List[float] = [0.5, 0.7, 0.9]) -> pd.DataFrame:
        """
        Perform sensitivity analysis across different risk tolerance levels
        
        Args:
            player_df: Player predictions DataFrame
            opponent_model: OpponentModeler instance
            risk_levels: List of risk tolerance levels to analyze
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        sensitivity_results = []
        
        for risk_level in risk_levels:
            logger.info(f"Running optimization for risk level: {risk_level}")
            
            solution = self.suggest_bids(player_df, opponent_model, risk_tolerance=risk_level)
            
            if len(solution) > 0:
                sensitivity_results.append({
                    'risk_level': risk_level,
                    'total_budget_used': solution['recommended_bid'].sum(),
                    'expected_total_score': solution['predicted_fantasy_score'].sum(),
                    'avg_win_probability': solution['win_probability'].mean(),
                    'high_risk_players': len(solution[solution['risk_level'] == 'High']),
                    'top_player': solution.iloc[0]['player_name'],
                    'most_expensive': solution.loc[solution['recommended_bid'].idxmax(), 'player_name']
                })
        
        return pd.DataFrame(sensitivity_results)


def suggest_bids(player_df: pd.DataFrame, opponent_model, 
                budget: int = 500, roster_constraints: Dict[str, int] = None,
                risk_tolerance: float = 0.7) -> pd.DataFrame:
    """
    Main function to suggest optimal bids for Fantacalcio auction
    
    Args:
        player_df: DataFrame with player predictions and market prices
        opponent_model: OpponentModeler instance from Phase 3
        budget: Total budget available
        roster_constraints: Dictionary with position requirements
        risk_tolerance: Risk tolerance level (0-1)
        
    Returns:
        DataFrame with recommended bids and team composition
    """
    
    # Set default roster constraints if not provided
    if roster_constraints is None:
        constraints = RosterConstraints(
            goalkeepers=3, defenders=7, midfielders=7, forwards=5, total_budget=budget
        )
    else:
        constraints = RosterConstraints(
            goalkeepers=roster_constraints.get('GK', 3),
            defenders=roster_constraints.get('DF', 7),
            midfielders=roster_constraints.get('MF', 7),
            forwards=roster_constraints.get('FW', 5),
            total_budget=budget
        )
    
    # Initialize optimizer
    optimizer = BidOptimizer(constraints)
    
    # Generate recommendations
    recommendations = optimizer.suggest_bids(player_df, opponent_model, risk_tolerance)
    
    return recommendations


def main():
    """Example usage of the bid optimization module"""
    try:
        # Import required modules
        from antoninorau.fantacalcio.phase3_opponent_modeling import OpponentModeler, create_sample_historical_data
        
        # Create sample data
        historical_data = create_sample_historical_data()
        opponent_model = OpponentModeler(historical_data)
        
        # Create sample current players
        current_players = pd.DataFrame({
            'player_name': ['Vlahovic', 'Lautaro', 'Osimhen', 'Theo', 'Bastoni', 
                          'Barella', 'Milinkovic', 'Zielinski', 'Maignan', 'Szczesny'],
            'position': ['FW', 'FW', 'FW', 'DF', 'DF', 'MF', 'MF', 'MF', 'GK', 'GK'],
            'predicted_fantasy_score': [88, 82, 90, 76, 70, 78, 75, 70, 58, 55],
            'market_value': [75000000, 85000000, 90000000, 50000000, 45000000, 
                           60000000, 55000000, 40000000, 35000000, 30000000]
        })
        
        # Add more players to meet roster requirements
        additional_players = pd.DataFrame({
            'player_name': ['Di Lorenzo', 'Skriniar', 'Dumfries', 'Kvaratskhelia', 
                          'Chiesa', 'Leao', 'Handanovic', 'Tonali', 'Verratti', 
                          'Immobile', 'Abraham', 'Dybala', 'Acerbi', 'Calabria', 'Spinazzola'],
            'position': ['DF', 'DF', 'DF', 'FW', 'FW', 'FW', 'GK', 'MF', 'MF', 
                        'FW', 'FW', 'FW', 'DF', 'DF', 'DF'],
            'predicted_fantasy_score': [65, 68, 60, 85, 80, 83, 50, 72, 74, 
                                      78, 75, 81, 62, 58, 56],
            'market_value': [35000000, 40000000, 30000000, 70000000, 60000000, 
                           80000000, 25000000, 50000000, 45000000, 
                           40000000, 35000000, 55000000, 25000000, 20000000, 18000000]
        })
        
        all_players = pd.concat([current_players, additional_players], ignore_index=True)
        
        print(f"Running bid optimization for {len(all_players)} players")
        
        # Generate bid recommendations
        recommendations = suggest_bids(
            all_players, 
            opponent_model, 
            budget=500, 
            risk_tolerance=0.7
        )
        
        print("\n=== FINAL BID RECOMMENDATIONS ===")
        print(recommendations[['player_name', 'position', 'predicted_fantasy_score', 
                             'recommended_bid', 'win_probability', 'risk_level']])
        
        print(f"\nTotal budget used: {recommendations['recommended_bid'].sum()}/500")
        print(f"Expected total score: {recommendations['predicted_fantasy_score'].sum():.1f}")
        
        # Save recommendations
        recommendations.to_csv('bid_recommendations.csv', index=False)
        print("\nRecommendations saved to bid_recommendations.csv")
        
        # Perform sensitivity analysis
        optimizer = BidOptimizer(RosterConstraints())
        sensitivity = optimizer.analyze_sensitivity(all_players, opponent_model, [0.5, 0.7, 0.9])
        print("\n=== SENSITIVITY ANALYSIS ===")
        print(sensitivity)
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all previous modules are available")


if __name__ == "__main__":
    main()