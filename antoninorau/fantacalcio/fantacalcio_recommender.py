"""
Fantacalcio AI-Powered Recommendation Engine
Main Integration Module

This module integrates all four phases to provide complete bid recommendations
for Italian Fantacalcio sealed-bid auctions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

# Import our custom modules
from .phase1_data_collection import FantacalcioDataCollector
from .phase2_performance_prediction import predict_player_performance
from .phase3_opponent_modeling import OpponentModeler, create_sample_historical_data
from .phase4_bid_optimization import suggest_bids, BidOptimizer, RosterConstraints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FantacalcioRecommender:
    """
    Complete AI-powered recommendation engine for Fantacalcio auctions
    """
    
    def __init__(self, historical_auction_data: Optional[pd.DataFrame] = None):
        """
        Initialize the recommendation engine
        
        Args:
            historical_auction_data: DataFrame with past auction results
                                   If None, sample data will be used
        """
        self.data_collector = FantacalcioDataCollector()
        self.historical_data = historical_auction_data
        self.opponent_model = None
        self.player_predictions = None
        self.recommendations = None
        
        # Initialize opponent model
        if historical_auction_data is not None:
            self.opponent_model = OpponentModeler(historical_auction_data)
        else:
            logger.warning("No historical data provided. Using sample data for opponent modeling.")
            sample_data = create_sample_historical_data()
            self.opponent_model = OpponentModeler(sample_data)
    
    def collect_current_season_data(self, seasons: list = None) -> pd.DataFrame:
        """
        Collect current season player data
        
        Args:
            seasons: List of seasons to collect data for
            
        Returns:
            DataFrame with player statistics
        """
        if seasons is None:
            seasons = ["2024-25", "2023-24", "2022-23"]
        
        logger.info("Phase 1: Collecting player data...")
        
        try:
            # Try to collect real data
            player_data = self.data_collector.create_master_dataframe(seasons)
            
            if len(player_data) == 0:
                logger.warning("No data collected from web sources. Using sample data.")
                player_data = self._create_sample_player_data()
                
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            logger.info("Falling back to sample data")
            player_data = self._create_sample_player_data()
        
        logger.info(f"Data collection complete. {len(player_data)} player records collected.")
        return player_data
    
    def _create_sample_player_data(self) -> pd.DataFrame:
        """Create comprehensive sample data for demonstration"""
        
        np.random.seed(42)
        
        # Define player pools by position
        players_data = {
            'GK': [
                ('Maignan', 25, 'Milan'), ('Szczesny', 33, 'Juventus'), 
                ('Handanovic', 38, 'Inter'), ('Meret', 26, 'Napoli'),
                ('Perin', 30, 'Genoa'), ('Provedel', 29, 'Lazio')
            ],
            'DF': [
                ('Theo Hernandez', 26, 'Milan'), ('Bastoni', 24, 'Inter'),
                ('Di Lorenzo', 29, 'Napoli'), ('Skriniar', 28, 'PSG'),
                ('Dumfries', 27, 'Inter'), ('Acerbi', 35, 'Inter'),
                ('Calabria', 26, 'Milan'), ('Spinazzola', 30, 'Roma'),
                ('Darmian', 33, 'Inter'), ('Tomori', 25, 'Milan'),
                ('Kolasinac', 30, 'Atalanta'), ('Hateboer', 29, 'Atalanta'),
                ('Cambiaso', 23, 'Juventus'), ('Dimarco', 26, 'Inter'),
                ('Udogie', 21, 'Udinese')
            ],
            'MF': [
                ('Barella', 26, 'Inter'), ('Milinkovic-Savic', 28, 'Lazio'),
                ('Zielinski', 29, 'Napoli'), ('Tonali', 23, 'Milan'),
                ('Verratti', 30, 'PSG'), ('Locatelli', 25, 'Juventus'),
                ('Bennacer', 25, 'Milan'), ('Brozovic', 30, 'Al-Nassr'),
                ('Pellegrini', 27, 'Roma'), ('Chiesa', 26, 'Juventus'),
                ('Kvaratskhelia', 22, 'Napoli'), ('Leao', 24, 'Milan'),
                ('Politano', 29, 'Napoli'), ('Insigne', 32, 'Toronto')
            ],
            'FW': [
                ('Vlahovic', 23, 'Juventus'), ('Lautaro', 26, 'Inter'),
                ('Osimhen', 25, 'Napoli'), ('Immobile', 33, 'Lazio'),
                ('Abraham', 26, 'Roma'), ('Dybala', 30, 'Roma'),
                ('Giroud', 37, 'Milan'), ('Morata', 31, 'Atletico'),
                ('Scamacca', 24, 'West Ham'), ('Belotti', 29, 'Roma'),
                ('Kean', 23, 'Juventus'), ('Raspadori', 23, 'Napoli')
            ]
        }
        
        all_players = []
        seasons = ['2024-25', '2023-24', '2022-23']
        
        for position, players in players_data.items():
            for season in seasons:
                for player_name, age, team in players:
                    # Generate realistic fantasy stats based on position
                    base_scores = {'GK': 55, 'DF': 65, 'MF': 70, 'FW': 75}
                    base_score = base_scores[position]
                    
                    # Add age-based variation
                    if age < 25:
                        age_bonus = 5
                    elif age > 30:
                        age_bonus = -3
                    else:
                        age_bonus = 0
                    
                    fantavoto = max(5.5, min(8.5, np.random.normal(6.0 + age_bonus/10, 0.3)))
                    games_played = np.random.randint(25, 38)
                    
                    if position == 'FW':
                        goals = max(0, np.random.poisson(12))
                        assists = max(0, np.random.poisson(5))
                    elif position == 'MF':
                        goals = max(0, np.random.poisson(6))
                        assists = max(0, np.random.poisson(8))
                    elif position == 'DF':
                        goals = max(0, np.random.poisson(3))
                        assists = max(0, np.random.poisson(4))
                    else:  # GK
                        goals = 0
                        assists = max(0, np.random.poisson(1))
                    
                    market_value = np.random.uniform(15, 90) * 1000000
                    
                    all_players.append({
                        'player_name': player_name,
                        'season': season,
                        'age': age,
                        'team': team,
                        'position': position,
                        'games_played': games_played,
                        'minutes_played': games_played * np.random.randint(75, 90),
                        'goals': goals,
                        'assists': assists,
                        'fantavoto_avg': round(fantavoto, 2),
                        'market_value': market_value,
                        'yellow_cards': max(0, np.random.poisson(3)),
                        'red_cards': max(0, np.random.poisson(0.2))
                    })
        
        return pd.DataFrame(all_players)
    
    def generate_performance_predictions(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate performance predictions for all players
        
        Args:
            player_data: Historical player data
            
        Returns:
            DataFrame with performance predictions
        """
        logger.info("Phase 2: Generating performance predictions...")
        
        try:
            predictions = predict_player_performance(player_data, model_type='xgboost')
            logger.info(f"Performance predictions generated for {len(predictions)} players")
            self.player_predictions = predictions
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            # Fallback: simple prediction based on last season
            current_season = player_data[player_data['season'] == '2024-25']
            if len(current_season) == 0:
                current_season = player_data.groupby('player_name').last().reset_index()
            
            current_season['predicted_fantasy_score'] = (
                current_season['fantavoto_avg'] * 30 + 
                current_season['goals'] * 3 + 
                current_season['assists'] * 1
            )
            
            self.player_predictions = current_season
            return current_season
    
    def generate_final_recommendations(self, budget: int = 500, 
                                     roster_constraints: Dict[str, int] = None,
                                     risk_tolerance: float = 0.7) -> pd.DataFrame:
        """
        Generate final bid recommendations
        
        Args:
            budget: Total budget available
            roster_constraints: Position requirements
            risk_tolerance: Risk tolerance level (0-1)
            
        Returns:
            DataFrame with final recommendations
        """
        if self.player_predictions is None:
            raise ValueError("Must generate predictions first")
        
        logger.info("Phase 4: Generating final bid recommendations...")
        
        # Set default constraints if not provided
        if roster_constraints is None:
            roster_constraints = {'GK': 3, 'DF': 7, 'MF': 7, 'FW': 5}
        
        # Generate recommendations
        self.recommendations = suggest_bids(
            self.player_predictions,
            self.opponent_model,
            budget=budget,
            roster_constraints=roster_constraints,
            risk_tolerance=risk_tolerance
        )
        
        logger.info(f"Final recommendations generated for {len(self.recommendations)} players")
        return self.recommendations
    
    def run_complete_analysis(self, budget: int = 500, 
                            risk_tolerance: float = 0.7,
                            save_results: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run complete analysis pipeline
        
        Args:
            budget: Total budget available
            risk_tolerance: Risk tolerance level
            save_results: Whether to save results to CSV files
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting complete Fantacalcio analysis...")
        
        # Phase 1: Data Collection
        player_data = self.collect_current_season_data()
        
        # Phase 2: Performance Predictions
        predictions = self.generate_performance_predictions(player_data)
        
        # Phase 3: Market Analysis (already done in __init__)
        market_analysis = self.opponent_model.estimate_market_price(predictions)
        
        # Phase 4: Bid Optimization
        recommendations = self.generate_final_recommendations(
            budget=budget, 
            risk_tolerance=risk_tolerance
        )
        
        # Compile results
        results = {
            'player_data': player_data,
            'predictions': predictions,
            'market_analysis': market_analysis,
            'recommendations': recommendations,
            'opponent_summary': self.opponent_model.get_opponent_summary()
        }
        
        # Save results if requested
        if save_results:
            self._save_results(results)
        
        # Print summary
        self._print_analysis_summary(results)
        
        return results
    
    def _save_results(self, results: Dict[str, pd.DataFrame]):
        """Save analysis results to CSV files"""
        
        logger.info("Saving analysis results...")
        
        for name, df in results.items():
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                filename = f"fantacalcio_{name}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved {name} to {filename}")
    
    def _print_analysis_summary(self, results: Dict[str, pd.DataFrame]):
        """Print comprehensive analysis summary"""
        
        recommendations = results['recommendations']
        
        if len(recommendations) == 0:
            logger.error("No recommendations generated!")
            return
        
        print("\n" + "="*60)
        print("FANTACALCIO AI RECOMMENDATION SUMMARY")
        print("="*60)
        
        # Team composition
        print("\n📋 RECOMMENDED TEAM COMPOSITION:")
        position_summary = recommendations.groupby('position').agg({
            'player_name': 'count',
            'recommended_bid': 'sum',
            'predicted_fantasy_score': 'sum'
        }).round(1)
        position_summary.columns = ['Count', 'Total_Bid', 'Expected_Score']
        print(position_summary)
        
        # Budget summary
        total_budget = recommendations['recommended_bid'].sum()
        print(f"\n💰 BUDGET ANALYSIS:")
        print(f"Total Budget Used: {total_budget:.1f}/500 ({total_budget/500*100:.1f}%)")
        print(f"Budget Remaining: {500-total_budget:.1f}")
        
        # Top recommendations by position
        print(f"\n⭐ TOP RECOMMENDATIONS BY POSITION:")
        for position in ['GK', 'DF', 'MF', 'FW']:
            pos_players = recommendations[recommendations['position'] == position]
            if len(pos_players) > 0:
                top_player = pos_players.iloc[0]
                print(f"{position}: {top_player['player_name']} "
                      f"(Bid: {top_player['recommended_bid']:.0f}, "
                      f"Score: {top_player['predicted_fantasy_score']:.1f})")
        
        # Risk analysis
        risk_summary = recommendations['risk_level'].value_counts()
        print(f"\n⚠️  RISK ANALYSIS:")
        for risk, count in risk_summary.items():
            print(f"{risk} Risk Players: {count}")
        
        print(f"\nAverage Win Probability: {recommendations['win_probability'].mean():.1%}")
        print(f"Expected Total Fantasy Score: {recommendations['predicted_fantasy_score'].sum():.1f}")
        
        print("\n" + "="*60)
        print("🚀 Good luck with your Fantacalcio auction!")
        print("="*60)


def main():
    """Main execution function"""
    
    print("🏈 Fantacalcio AI-Powered Recommendation Engine")
    print("=" * 50)
    
    # Initialize the recommender
    recommender = FantacalcioRecommender()
    
    # Run complete analysis
    results = recommender.run_complete_analysis(
        budget=500,
        risk_tolerance=0.7,
        save_results=True
    )
    
    # Display final recommendations table
    if len(results['recommendations']) > 0:
        print("\n📊 DETAILED BID RECOMMENDATIONS:")
        display_cols = ['player_name', 'position', 'predicted_fantasy_score', 
                       'recommended_bid', 'estimated_market_price', 'win_probability']
        print(results['recommendations'][display_cols].to_string(index=False))


if __name__ == "__main__":
    main()