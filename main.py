#!/usr/bin/env python3
"""
Main entry point for the Fantacalcio AI-Powered Recommendation Engine
"""

from antoninorau.fantacalcio import FantacalcioRecommender

if __name__ == "__main__":
    # Initialize the system
    recommender = FantacalcioRecommender()
    
    # Run complete analysis
    print("🚀 Starting Fantacalcio AI Analysis...")
    results = recommender.run_complete_analysis(
        budget=500,
        risk_tolerance=0.7,
        save_results=True
    )
    
    print("✅ Analysis complete! Check the generated CSV files for detailed results.")