"""
Fantacalcio AI-Powered Recommendation Engine

An intelligent bid optimization system for Italian Fantacalcio sealed-bid auctions
using machine learning, opponent modeling, and mathematical optimization.
"""

from .fantacalcio_recommender import FantacalcioRecommender
from .phase1_data_collection import FantacalcioDataCollector, PlayerData
from .phase2_performance_prediction import FantacalcioPredictor, predict_player_performance
from .phase3_opponent_modeling import OpponentModeler, create_sample_historical_data
from .phase4_bid_optimization import BidOptimizer, RosterConstraints, suggest_bids

__version__ = "1.0.0"
__author__ = "Antonino Rau"

__all__ = [
    "FantacalcioRecommender",
    "FantacalcioDataCollector", 
    "PlayerData",
    "FantacalcioPredictor",
    "predict_player_performance",
    "OpponentModeler",
    "create_sample_historical_data",
    "BidOptimizer",
    "RosterConstraints", 
    "suggest_bids"
]