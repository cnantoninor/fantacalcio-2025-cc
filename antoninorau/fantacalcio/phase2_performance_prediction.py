"""
Fantacalcio Performance Prediction Module
Phase 2: Player Performance Prediction Model

This module implements machine learning models to predict player fantasy scores
for the upcoming season based on historical performance data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, List, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FantacalcioPredictor:
    """
    Machine learning model for predicting Fantacalcio player performance
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the predictor
        
        Args:
            model_type: Type of model to use ('xgboost', 'lightgbm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'total_fantasy_score'
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset
        
        Args:
            df: Raw player data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Sort by player and season for time-series features
        df_features = df_features.sort_values(['player_name', 'season'])
        
        # Calculate target variable: total fantasy score
        # Fantasy score approximation: (fantavoto_avg * games_played) + goal/assist bonuses
        df_features['total_fantasy_score'] = (
            df_features['fantavoto_avg'] * df_features['games_played'] +
            df_features['goals'] * 3 +  # 3 points per goal
            df_features['assists'] * 1 +  # 1 point per assist
            df_features['yellow_cards'] * -0.5 +  # -0.5 per yellow
            df_features['red_cards'] * -3  # -3 per red card
        )
        
        # Age-based features
        df_features['age_squared'] = df_features['age'] ** 2
        df_features['is_prime_age'] = ((df_features['age'] >= 24) & (df_features['age'] <= 28)).astype(int)
        df_features['is_young'] = (df_features['age'] < 23).astype(int)
        df_features['is_veteran'] = (df_features['age'] > 30).astype(int)
        
        # Performance ratios and rates
        df_features['goals_per_90'] = (df_features['goals'] / (df_features['minutes_played'] / 90)).fillna(0)
        df_features['assists_per_90'] = (df_features['assists'] / (df_features['minutes_played'] / 90)).fillna(0)
        df_features['involvement_rate'] = (df_features['goals'] + df_features['assists']) / df_features['games_played'].replace(0, 1)
        df_features['consistency_score'] = df_features['fantavoto_avg'] / df_features['fantavoto_avg'].std() if df_features['fantavoto_avg'].std() > 0 else 0
        
        # Market value features
        df_features['log_market_value'] = np.log1p(df_features['market_value'])
        df_features['market_value_per_age'] = df_features['market_value'] / df_features['age'].replace(0, 1)
        
        # Team strength proxy (based on average market value)
        team_strength = df_features.groupby(['team', 'season'])['market_value'].mean().reset_index()
        team_strength.columns = ['team', 'season', 'team_avg_value']
        df_features = df_features.merge(team_strength, on=['team', 'season'], how='left')
        df_features['team_avg_value'].fillna(df_features['team_avg_value'].median(), inplace=True)
        
        # Historical performance features (lagged variables)
        df_features = self._create_lagged_features(df_features)
        
        # Position-specific features
        df_features = self._create_position_features(df_features)
        
        # Season trend features
        df_features['season_numeric'] = df_features['season'].str[:4].astype(float)
        
        logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
        return df_features
    
    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features from previous seasons"""
        
        # Sort by player and season
        df = df.sort_values(['player_name', 'season_numeric'])
        
        # Features to lag
        lag_features = ['total_fantasy_score', 'fantavoto_avg', 'goals', 'assists', 
                       'games_played', 'minutes_played', 'market_value']
        
        for feature in lag_features:
            # 1-season lag
            df[f'{feature}_lag1'] = df.groupby('player_name')[feature].shift(1)
            
            # 2-season lag
            df[f'{feature}_lag2'] = df.groupby('player_name')[feature].shift(2)
            
            # Rolling averages (2 and 3 seasons)
            df[f'{feature}_rolling2'] = df.groupby('player_name')[feature].rolling(2, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{feature}_rolling3'] = df.groupby('player_name')[feature].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        
        # Performance trend features
        df['fantasy_score_trend'] = df['total_fantasy_score'] - df['total_fantasy_score_lag1']
        df['fantavoto_trend'] = df['fantavoto_avg'] - df['fantavoto_avg_lag1']
        df['market_value_trend'] = df['market_value'] - df['market_value_lag1']
        
        # Career experience proxy
        df['seasons_played'] = df.groupby('player_name').cumcount() + 1
        
        return df
    
    def _create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features"""
        
        position_stats = df.groupby('position').agg({
            'total_fantasy_score': ['mean', 'std'],
            'fantavoto_avg': ['mean', 'std'],
            'goals': 'mean',
            'assists': 'mean'
        }).round(3)
        
        # Flatten column names
        position_stats.columns = [f"pos_{pos}_{'_'.join(col).strip('_')}" for col in position_stats.columns]
        position_stats = position_stats.reset_index()
        
        # Merge back to main dataframe
        df = df.merge(position_stats, on='position', how='left')
        
        # Position-specific ratios
        df['fantasy_vs_pos_avg'] = df['total_fantasy_score'] / df['pos_total_fantasy_score_mean']
        df['fantavoto_vs_pos_avg'] = df['fantavoto_avg'] / df['pos_fantavoto_avg_mean']
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training by splitting into features and targets
        
        Args:
            df: Feature-engineered DataFrame
            
        Returns:
            Tuple of (X, y) for training
        """
        # Remove rows where target is missing
        df_clean = df.dropna(subset=[self.target_column])
        
        # Remove current season data for prediction (keep for features only)
        current_season = df_clean['season_numeric'].max()
        df_train = df_clean[df_clean['season_numeric'] < current_season].copy()
        
        if len(df_train) == 0:
            logger.warning("No historical data available for training")
            df_train = df_clean.copy()
        
        # Define feature columns (exclude target and identifier columns)
        exclude_cols = ['player_name', 'season', 'total_fantasy_score', 'team']
        self.feature_columns = [col for col in df_train.columns if col not in exclude_cols]
        
        # Prepare features and target
        X = df_train[self.feature_columns].copy()
        y = df_train[self.target_column].copy()
        
        # Handle categorical variables
        categorical_features = ['position']
        for feature in categorical_features:
            if feature in X.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    X[feature] = self.label_encoders[feature].fit_transform(X[feature].astype(str))
                else:
                    X[feature] = self.label_encoders[feature].transform(X[feature].astype(str))
        
        # Fill missing values
        X = X.fillna(X.median())
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the machine learning model
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing model performance metrics
        """
        logger.info(f"Training {self.model_type} model")
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for certain models (not needed for tree-based models but kept for consistency)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        metrics['cv_mae'] = -cv_scores.mean()
        metrics['cv_mae_std'] = cv_scores.std()
        
        logger.info(f"Model training complete. Test R²: {metrics['test_r2']:.3f}, Test MAE: {metrics['test_mae']:.3f}")
        
        return metrics
    
    def predict_next_season(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fantasy scores for all players for the next season
        
        Args:
            df: Feature-engineered DataFrame including current season data
            
        Returns:
            DataFrame with player predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Generating predictions for next season")
        
        # Get the most recent season data for each player
        current_season = df['season_numeric'].max()
        df_current = df[df['season_numeric'] == current_season].copy()
        
        # Prepare features
        X_pred = df_current[self.feature_columns].copy()
        
        # Handle categorical variables
        for feature, encoder in self.label_encoders.items():
            if feature in X_pred.columns:
                # Handle unseen categories
                X_pred[feature] = X_pred[feature].astype(str)
                mask = X_pred[feature].isin(encoder.classes_)
                X_pred.loc[mask, feature] = encoder.transform(X_pred.loc[mask, feature])
                X_pred.loc[~mask, feature] = -1  # Unknown category
        
        # Fill missing values
        X_pred = X_pred.fillna(X_pred.median())
        
        # Make predictions
        predictions = self.model.predict(X_pred)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'player_name': df_current['player_name'],
            'position': df_current['position'],
            'team': df_current['team'],
            'age': df_current['age'],
            'predicted_fantasy_score': predictions,
            'last_season_score': df_current['total_fantasy_score'],
            'fantavoto_avg': df_current['fantavoto_avg'],
            'market_value': df_current['market_value']
        })
        
        # Add confidence intervals (simple approach using model variance)
        if hasattr(self.model, 'predict'):
            # For tree-based models, use standard deviation of predictions as uncertainty measure
            results['prediction_std'] = predictions.std()
            results['prediction_lower'] = predictions - 1.96 * results['prediction_std']
            results['prediction_upper'] = predictions + 1.96 * results['prediction_std']
        
        # Sort by predicted score
        results = results.sort_values('predicted_fantasy_score', ascending=False)
        
        logger.info(f"Predictions generated for {len(results)} players")
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()


def predict_player_performance(df: pd.DataFrame, model_type: str = 'xgboost') -> pd.DataFrame:
    """
    Main function to predict player performance for the next season
    
    Args:
        df: Raw player data DataFrame from Phase 1
        model_type: Type of model to use ('xgboost', 'lightgbm')
        
    Returns:
        DataFrame with player predictions for next season
    """
    # Initialize predictor
    predictor = FantacalcioPredictor(model_type=model_type)
    
    # Feature engineering
    df_features = predictor.prepare_features(df)
    
    # Prepare training data
    X, y = predictor.prepare_training_data(df_features)
    
    # Train model
    metrics = predictor.train_model(X, y)
    
    # Generate predictions
    predictions = predictor.predict_next_season(df_features)
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    
    # Print summary
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    print(f"\nTop 10 Features:")
    print(feature_importance.head(10))
    
    print(f"\nTop 10 Predicted Players:")
    print(predictions[['player_name', 'position', 'predicted_fantasy_score', 'market_value']].head(10))
    
    return predictions


def main():
    """Example usage of the performance prediction module"""
    # Load data (assuming data_collection.py has been run)
    try:
        df = pd.read_csv('fantacalcio_player_data.csv')
        print(f"Loaded data for {len(df)} player records")
        
        # Generate predictions
        predictions = predict_player_performance(df, model_type='xgboost')
        
        # Save predictions
        predictions.to_csv('player_predictions.csv', index=False)
        print(f"\nPredictions saved to player_predictions.csv")
        
    except FileNotFoundError:
        print("Please run data_collection.py first to generate the dataset")


if __name__ == "__main__":
    main()