"""
Fantacalcio Opponent Modeling Module
Phase 3: Opponent Behavior & Bid Prediction Model

This module models opponent behavior and predicts auction prices by analyzing
historical bidding patterns and market dynamics.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpponentModeler:
    """
    Models opponent behavior and predicts market clearing prices for fantasy auction
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        Initialize the opponent modeler with historical auction data
        
        Args:
            historical_data: DataFrame with columns ['player_name', 'winning_bid', 
                           'winning_manager', 'season', 'position', 'predicted_fantasy_score']
        """
        self.historical_data = historical_data.copy()
        self.opponent_profiles = {}
        self.market_model = None
        self.scaler = StandardScaler()
        self.position_multipliers = {}
        
        # Initialize the modeler
        self._analyze_historical_data()
        self._create_opponent_profiles()
        self._train_market_model()
        
        logger.info(f"OpponentModeler initialized with {len(historical_data)} historical auctions")
    
    def _analyze_historical_data(self):
        """Analyze historical data to extract market patterns"""
        
        # Calculate value-to-price ratios
        self.historical_data['value_per_bid'] = (
            self.historical_data['predicted_fantasy_score'] / 
            self.historical_data['winning_bid'].replace(0, 1)
        )
        
        # Position-based analysis
        position_stats = self.historical_data.groupby('position').agg({
            'winning_bid': ['mean', 'std', 'median'],
            'predicted_fantasy_score': ['mean', 'std'],
            'value_per_bid': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        position_stats.columns = ['_'.join(col).strip() for col in position_stats.columns]
        self.position_stats = position_stats
        
        # Calculate position multipliers (how much each position typically costs relative to predicted value)
        for position in self.historical_data['position'].unique():
            pos_data = self.historical_data[self.historical_data['position'] == position]
            if len(pos_data) > 0:
                self.position_multipliers[position] = pos_data['winning_bid'].sum() / pos_data['predicted_fantasy_score'].sum()
        
        logger.info(f"Analyzed historical data across {len(self.position_multipliers)} positions")
    
    def _create_opponent_profiles(self):
        """Create behavioral profiles for each opponent based on historical bidding"""
        
        for manager in self.historical_data['winning_manager'].unique():
            if pd.isna(manager):
                continue
                
            manager_data = self.historical_data[self.historical_data['winning_manager'] == manager]
            
            if len(manager_data) < 3:  # Need minimum data for profiling
                continue
            
            # Calculate manager-specific metrics
            profile = {
                'total_bids': len(manager_data),
                'avg_bid': manager_data['winning_bid'].mean(),
                'total_spent': manager_data['winning_bid'].sum(),
                'avg_value_per_bid': manager_data['value_per_bid'].mean(),
                'bid_variance': manager_data['winning_bid'].var(),
                'positions_preference': {},
                'overbid_tendency': 0.0,
                'budget_aggressiveness': 0.0
            }
            
            # Position preferences and tendencies
            for position in ['GK', 'DF', 'MF', 'FW']:
                pos_data = manager_data[manager_data['position'] == position]
                if len(pos_data) > 0:
                    global_pos_avg = self.historical_data[
                        self.historical_data['position'] == position
                    ]['winning_bid'].mean()
                    
                    manager_pos_avg = pos_data['winning_bid'].mean()
                    preference_ratio = manager_pos_avg / global_pos_avg if global_pos_avg > 0 else 1.0
                    
                    profile['positions_preference'][position] = {
                        'count': len(pos_data),
                        'avg_bid': manager_pos_avg,
                        'preference_ratio': preference_ratio,
                        'avg_value_ratio': pos_data['value_per_bid'].mean()
                    }
            
            # Calculate overbidding tendency (how often they bid above market average)
            market_avg_by_position = self.historical_data.groupby('position')['winning_bid'].mean()
            overbids = 0
            for _, row in manager_data.iterrows():
                market_avg = market_avg_by_position.get(row['position'], 0)
                if row['winning_bid'] > market_avg:
                    overbids += 1
            profile['overbid_tendency'] = overbids / len(manager_data) if len(manager_data) > 0 else 0.5
            
            # Budget aggressiveness (early vs late season spending)
            if 'auction_round' in manager_data.columns:
                early_spending = manager_data[manager_data['auction_round'] <= 0.3]['winning_bid'].sum()
                total_spending = manager_data['winning_bid'].sum()
                profile['budget_aggressiveness'] = early_spending / total_spending if total_spending > 0 else 0.3
            else:
                profile['budget_aggressiveness'] = 0.4  # Default moderate aggressiveness
            
            self.opponent_profiles[manager] = profile
        
        # Cluster opponents into behavioral groups
        self._cluster_opponents()
        
        logger.info(f"Created profiles for {len(self.opponent_profiles)} opponents")
    
    def _cluster_opponents(self):
        """Cluster opponents based on bidding behavior"""
        
        if len(self.opponent_profiles) < 3:
            return
        
        # Extract features for clustering
        features = []
        manager_names = []
        
        for manager, profile in self.opponent_profiles.items():
            feature_vector = [
                profile['avg_bid'],
                profile['avg_value_per_bid'],
                profile['overbid_tendency'],
                profile['budget_aggressiveness'],
                len(profile['positions_preference'])
            ]
            features.append(feature_vector)
            manager_names.append(manager)
        
        features_array = np.array(features)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_array)
        
        # Perform clustering (3 clusters: Conservative, Moderate, Aggressive)
        n_clusters = min(3, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_normalized)
        
        # Assign cluster labels
        cluster_names = ['Conservative', 'Moderate', 'Aggressive']
        for i, manager in enumerate(manager_names):
            self.opponent_profiles[manager]['cluster'] = cluster_names[clusters[i] % len(cluster_names)]
    
    def _train_market_model(self):
        """Train a model to predict market clearing prices"""
        
        # Prepare features for market price prediction
        features = []
        targets = []
        
        for _, row in self.historical_data.iterrows():
            if pd.isna(row['predicted_fantasy_score']) or pd.isna(row['winning_bid']):
                continue
            
            feature_vector = [
                row['predicted_fantasy_score'],
                self.position_multipliers.get(row['position'], 1.0),
                len(self.historical_data[self.historical_data['position'] == row['position']]) / len(self.historical_data),  # Position popularity
            ]
            
            # Add seasonal trends if available
            if 'season' in row and pd.notna(row['season']):
                try:
                    season_numeric = float(str(row['season'])[:4])
                    feature_vector.append(season_numeric)
                except:
                    feature_vector.append(2024)  # Default
            else:
                feature_vector.append(2024)
            
            features.append(feature_vector)
            targets.append(row['winning_bid'])
        
        if len(features) > 5:
            X = np.array(features)
            y = np.array(targets)
            
            # Train Random Forest model for better non-linear relationships
            self.market_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.market_model.fit(X, y)
            
            # Calculate model performance
            predictions = self.market_model.predict(X)
            r2_score = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
            logger.info(f"Market model trained with R² score: {r2_score:.3f}")
        else:
            logger.warning("Insufficient data to train market model")
    
    def estimate_market_price(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate market clearing price for each player
        
        Args:
            player_df: DataFrame with player predictions from Phase 2
            
        Returns:
            DataFrame with estimated market prices
        """
        player_df = player_df.copy()
        
        # Initialize market price column
        player_df['estimated_market_price'] = 0.0
        player_df['price_confidence'] = 0.0
        
        for idx, row in player_df.iterrows():
            try:
                if self.market_model is not None:
                    # Use trained model
                    features = np.array([[
                        row['predicted_fantasy_score'],
                        self.position_multipliers.get(row['position'], 1.0),
                        len(player_df[player_df['position'] == row['position']]) / len(player_df),
                        2025  # Current season
                    ]])
                    
                    predicted_price = self.market_model.predict(features)[0]
                    
                    # Add position-based adjustment
                    position_avg = self.position_stats.loc[
                        row['position'], 'winning_bid_mean'
                    ] if row['position'] in self.position_stats.index else 50
                    
                    # Weighted average of model prediction and position average
                    estimated_price = 0.7 * predicted_price + 0.3 * position_avg
                    
                else:
                    # Fallback: use position multiplier and fantasy score
                    multiplier = self.position_multipliers.get(row['position'], 15.0)
                    estimated_price = row['predicted_fantasy_score'] * multiplier
                
                # Apply minimum and maximum bounds
                min_price = 1  # Minimum bid
                max_price = 150  # Reasonable maximum for most players
                estimated_price = np.clip(estimated_price, min_price, max_price)
                
                player_df.loc[idx, 'estimated_market_price'] = estimated_price
                player_df.loc[idx, 'price_confidence'] = 0.8 if self.market_model is not None else 0.5
                
            except Exception as e:
                logger.warning(f"Error estimating price for {row['player_name']}: {e}")
                # Fallback to simple calculation
                fallback_price = max(1, row.get('market_value', 0) / 1000000)  # Convert market value to bid range
                player_df.loc[idx, 'estimated_market_price'] = fallback_price
                player_df.loc[idx, 'price_confidence'] = 0.3
        
        return player_df
    
    def predict_opponent_bid(self, player_id: str, opponent_id: str, 
                           base_market_price: float, position: str) -> float:
        """
        Predict specific opponent's bid for a player
        
        Args:
            player_id: Player identifier
            opponent_id: Opponent manager identifier  
            base_market_price: Estimated market price for the player
            position: Player position
            
        Returns:
            Predicted bid amount
        """
        if opponent_id not in self.opponent_profiles:
            # Return market price with some random variation for unknown opponents
            return base_market_price * np.random.uniform(0.8, 1.2)
        
        profile = self.opponent_profiles[opponent_id]
        
        # Start with base market price
        predicted_bid = base_market_price
        
        # Apply position preference
        if position in profile['positions_preference']:
            pos_pref = profile['positions_preference'][position]
            predicted_bid *= pos_pref['preference_ratio']
        
        # Apply overbidding tendency
        if profile['overbid_tendency'] > 0.6:  # Aggressive bidder
            predicted_bid *= 1.1
        elif profile['overbid_tendency'] < 0.4:  # Conservative bidder
            predicted_bid *= 0.9
        
        # Apply budget aggressiveness (early in auction)
        if profile['budget_aggressiveness'] > 0.5:
            predicted_bid *= 1.05
        
        # Add some randomness to reflect uncertainty
        noise_factor = np.random.uniform(0.95, 1.05)
        predicted_bid *= noise_factor
        
        # Ensure minimum bid
        predicted_bid = max(1, predicted_bid)
        
        return predicted_bid
    
    def simulate_auction_outcomes(self, player_df: pd.DataFrame, 
                                num_simulations: int = 1000) -> pd.DataFrame:
        """
        Simulate auction outcomes to estimate win probabilities
        
        Args:
            player_df: DataFrame with players and estimated market prices
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            DataFrame with win probability estimates
        """
        results = player_df.copy()
        results['win_probability'] = 0.0
        results['avg_winning_bid'] = 0.0
        results['bid_std'] = 0.0
        
        opponent_list = list(self.opponent_profiles.keys())
        if len(opponent_list) < 2:
            # If insufficient opponent data, use simplified model
            results['win_probability'] = 0.5
            results['avg_winning_bid'] = results['estimated_market_price']
            results['bid_std'] = results['estimated_market_price'] * 0.2
            return results
        
        for idx, row in results.iterrows():
            player_name = row['player_name']
            position = row['position']
            market_price = row['estimated_market_price']
            
            winning_bids = []
            
            # Run simulations
            for _ in range(min(num_simulations, 100)):  # Limit for performance
                # Generate bids from different opponents
                bids = []
                
                # Randomly select subset of opponents (simulate not all opponents bidding)
                active_opponents = np.random.choice(
                    opponent_list, 
                    size=min(len(opponent_list), np.random.randint(2, 8)),
                    replace=False
                )
                
                for opponent in active_opponents:
                    bid = self.predict_opponent_bid(player_name, opponent, market_price, position)
                    bids.append(bid)
                
                if bids:
                    winning_bids.append(max(bids))
            
            if winning_bids:
                results.loc[idx, 'avg_winning_bid'] = np.mean(winning_bids)
                results.loc[idx, 'bid_std'] = np.std(winning_bids)
                
                # Calculate win probability at market price
                win_count = sum(1 for bid in winning_bids if market_price >= bid)
                results.loc[idx, 'win_probability'] = win_count / len(winning_bids)
        
        return results
    
    def get_opponent_summary(self) -> pd.DataFrame:
        """
        Get a summary of opponent profiles
        
        Returns:
            DataFrame with opponent characteristics
        """
        if not self.opponent_profiles:
            return pd.DataFrame()
        
        summary_data = []
        for manager, profile in self.opponent_profiles.items():
            summary_data.append({
                'manager': manager,
                'total_bids': profile['total_bids'],
                'avg_bid': profile['avg_bid'],
                'total_spent': profile['total_spent'],
                'overbid_tendency': profile['overbid_tendency'],
                'budget_aggressiveness': profile['budget_aggressiveness'],
                'cluster': profile.get('cluster', 'Unknown'),
                'favorite_position': max(profile['positions_preference'].items(), 
                                       key=lambda x: x[1]['count'])[0] if profile['positions_preference'] else 'None'
            })
        
        return pd.DataFrame(summary_data).sort_values('total_spent', ascending=False)


def create_sample_historical_data() -> pd.DataFrame:
    """
    Create sample historical auction data for demonstration
    """
    np.random.seed(42)
    
    players = [
        ('Vlahovic', 'FW', 85), ('Lautaro', 'FW', 80), ('Osimhen', 'FW', 90),
        ('Theo Hernandez', 'DF', 75), ('Bastoni', 'DF', 70), ('Di Lorenzo', 'DF', 65),
        ('Barella', 'MF', 80), ('Milinkovic-Savic', 'MF', 75), ('Zielinski', 'MF', 70),
        ('Maignan', 'GK', 60), ('Szczesny', 'GK', 55), ('Handanovic', 'GK', 50)
    ]
    
    managers = ['Manager_A', 'Manager_B', 'Manager_C', 'Manager_D', 'Manager_E', 'Manager_F']
    seasons = ['2023-24', '2022-23', '2021-22']
    
    historical_data = []
    
    for season in seasons:
        for player_name, position, base_score in players:
            # Random variation in predicted score
            predicted_score = base_score + np.random.normal(0, 5)
            
            # Generate winning bid based on score and position
            if position == 'FW':
                base_bid = predicted_score * 0.8
            elif position == 'MF':
                base_bid = predicted_score * 0.7
            elif position == 'DF':
                base_bid = predicted_score * 0.6
            else:  # GK
                base_bid = predicted_score * 0.5
            
            winning_bid = max(1, base_bid + np.random.normal(0, 10))
            winning_manager = np.random.choice(managers)
            
            historical_data.append({
                'player_name': player_name,
                'winning_bid': winning_bid,
                'winning_manager': winning_manager,
                'season': season,
                'position': position,
                'predicted_fantasy_score': predicted_score
            })
    
    return pd.DataFrame(historical_data)


def main():
    """Example usage of the opponent modeling module"""
    # Create sample data
    historical_data = create_sample_historical_data()
    print(f"Created sample historical data with {len(historical_data)} records")
    
    # Initialize opponent modeler
    modeler = OpponentModeler(historical_data)
    
    # Create sample current players for prediction
    current_players = pd.DataFrame({
        'player_name': ['Vlahovic', 'Lautaro', 'Theo Hernandez', 'Barella', 'Maignan'],
        'position': ['FW', 'FW', 'DF', 'MF', 'GK'],
        'predicted_fantasy_score': [88, 82, 76, 78, 58],
        'market_value': [75000000, 85000000, 50000000, 60000000, 35000000]
    })
    
    # Estimate market prices
    market_prices = modeler.estimate_market_price(current_players)
    print("\nEstimated Market Prices:")
    print(market_prices[['player_name', 'position', 'predicted_fantasy_score', 'estimated_market_price']])
    
    # Simulate auction outcomes
    auction_simulation = modeler.simulate_auction_outcomes(market_prices)
    print("\nAuction Simulation Results:")
    print(auction_simulation[['player_name', 'estimated_market_price', 'win_probability', 'avg_winning_bid']])
    
    # Get opponent summary
    opponent_summary = modeler.get_opponent_summary()
    print("\nOpponent Summary:")
    print(opponent_summary)
    
    # Test individual opponent bid prediction
    if len(modeler.opponent_profiles) > 0:
        test_opponent = list(modeler.opponent_profiles.keys())[0]
        test_bid = modeler.predict_opponent_bid('Vlahovic', test_opponent, 70, 'FW')
        print(f"\nPredicted bid from {test_opponent} for Vlahovic: {test_bid:.1f}")


if __name__ == "__main__":
    main()