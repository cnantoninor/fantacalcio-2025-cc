# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, pulp; print('All dependencies installed successfully!')"
```

### Running the System
```bash
# Run complete analysis with default settings
python main.py

# Import and use programmatically
python -c "from antoninorau.fantacalcio import FantacalcioRecommender; rec = FantacalcioRecommender(); results = rec.run_complete_analysis()"
```

### Testing and Validation
```bash
# Test individual modules
python -m antoninorau.fantacalcio.data_collection
python -m antoninorau.fantacalcio.performance_prediction
python -m antoninorau.fantacalcio.opponent_modeling
python -m antoninorau.fantacalcio.bid_optimization

# Validate data quality
python -c "from antoninorau.fantacalcio import FantacalcioDataCollector; collector = FantacalcioDataCollector(); df = collector.create_master_dataframe(); print(f'Players: {df[\"player_name\"].nunique()}')"
```

## Architecture Overview

### Four-Phase Pipeline Architecture
The system follows a strict sequential pipeline with each phase depending on the previous:

1. **Phase 1: Data Collection** (`antoninorau.fantacalcio.data_collection`)
   - Web scraping from Fantacalcio.it, FBref, Transfermarkt
   - Structured data output via `FantacalcioDataCollector` class
   - Fallback to sample data when web scraping fails

2. **Phase 2: Performance Prediction** (`antoninorau.fantacalcio.performance_prediction`)
   - ML models (XGBoost/LightGBM) for fantasy score prediction
   - Feature engineering with lagged variables and age curves
   - `FantacalcioPredictor` class with configurable model types

3. **Phase 3: Opponent Modeling** (`antoninorau.fantacalcio.opponent_modeling`)
   - Historical auction analysis via `OpponentModeler` class
   - Market price estimation and win probability calculation
   - Manager behavioral profiling and clustering

4. **Phase 4: Bid Optimization** (`antoninorau.fantacalcio.bid_optimization`)
   - Integer Linear Programming using PuLP library
   - Budget and roster constraints optimization
   - `BidOptimizer` class with sensitivity analysis

### Key Integration Points

**Main Orchestrator**: `FantacalcioRecommender` class in `antoninorau.fantacalcio.fantacalcio_recommender`
- Manages the complete pipeline execution
- Handles error recovery and fallback strategies
- Exports results to CSV files when `save_results=True`

**Data Flow**:
```
Raw Data → ML Predictions → Market Analysis → Optimal Bids
```

**Critical Dependencies**:
- Each phase requires successful completion of previous phases
- `OpponentModeler` requires historical auction data (uses sample data as fallback)
- `BidOptimizer` uses both predictions and market analysis for optimization

### Roster Constraints System
```python
# Standard Fantacalcio roster (22 players total)
RosterConstraints(
    goalkeepers=3,    # GK
    defenders=7,      # DF  
    midfielders=7,    # MF
    forwards=5        # FW
)
```

### Risk Tolerance Configuration
- `0.3-0.5`: Conservative (higher win probability, lower bids)
- `0.7`: Moderate/Default (balanced approach)
- `0.8-0.9`: Aggressive (higher risk, potential bargains)

## Critical Implementation Details

### Web Scraping Resilience
- Graceful degradation when external sources fail
- Automatic fallback to sample/cached data
- Rate limiting and retry mechanisms built-in

### Optimization Engine
- Uses PuLP for mathematical programming
- Handles infeasible solutions with greedy fallback
- Supports sensitivity analysis across risk levels

### Model Training
- Cross-validation with time-series aware splits
- Feature importance analysis for interpretability
- Support for both XGBoost and LightGBM backends

### Data Validation
Essential checks when working with player data:
```python
# Always validate data structure
required_columns = ['player_name', 'season', 'position', 'fantavoto_avg']
assert all(col in df.columns for col in required_columns)

# Check position distribution
print(df['position'].value_counts())  # Should show GK, DF, MF, FW

# Validate budget constraints
assert df['recommended_bid'].sum() <= 500  # Total budget limit
```

### Error Handling Patterns
- Web scraping failures → Use sample data
- Optimization infeasible → Greedy algorithm fallback
- Missing features → Imputation with position averages
- Model training failures → Linear regression fallback

## Key Configuration Parameters

### Budget Settings
```python
TOTAL_BUDGET = 500      # Million (standard league budget)
MIN_BID = 1            # Million (minimum bid amount)  
MAX_BID = 150          # Million (reasonable maximum)
```

### Model Parameters
```python
# XGBoost default settings
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8
}

# Feature engineering windows
LAG_PERIODS = [1, 2, 3]  # Seasons for lagged features
AGE_CURVE_POSITIONS = ['GK', 'DF', 'MF', 'FW']  # Position-specific age curves
```

### Output Files Structure
When `save_results=True`:
- `fantacalcio_player_data.csv` → Raw collected data
- `fantacalcio_predictions.csv` → ML predictions
- `fantacalcio_market_analysis.csv` → Market intelligence
- `fantacalcio_recommendations.csv` → **Final bid suggestions**
- `fantacalcio_opponent_summary.csv` → Manager profiles