"""
Fantacalcio Data Collection Module
Phase 1: Data Collection & Structuring

This module handles web scraping and data structuring for player statistics,
market values, and performance data from Italian football sources.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PlayerData:
    """Data structure for individual player information"""

    player_name: str
    season: str
    age: int
    team: str
    position: str
    games_played: int
    goals: int
    assists: int
    fantavoto_avg: float
    market_value: float
    minutes_played: int
    yellow_cards: int = 0
    red_cards: int = 0


class FantacalcioDataCollector:
    """Main class for collecting Fantacalcio player data"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    def scrape_fantacalcio_stats(self, season: str = "2024-25") -> List[Dict]:
        """
        Scrape player statistics from Fantacalcio.it

        Args:
            season: Season string (e.g., "2024-25")

        Returns:
            List of dictionaries containing player statistics
        """
        players_data = []

        # Base URL for Fantacalcio.it (example structure)
        base_url = "https://www.fantacalcio.it/voti-fantacalcio-serie-a"

        try:
            logger.info(f"Scraping Fantacalcio stats for season {season}")
            response = self.session.get(base_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Find player tables (structure may vary)
            player_tables = soup.find_all("table", class_="player-stats")

            for table in player_tables:
                rows = table.find_all("tr")[1:]  # Skip header

                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) >= 8:
                        player_data = {
                            "player_name": cols[0].get_text(strip=True),
                            "team": cols[1].get_text(strip=True),
                            "position": self._standardize_position(
                                cols[2].get_text(strip=True)
                            ),
                            "games_played": self._safe_int(
                                cols[3].get_text(strip=True)
                            ),
                            "goals": self._safe_int(cols[4].get_text(strip=True)),
                            "assists": self._safe_int(cols[5].get_text(strip=True)),
                            "fantavoto_avg": self._safe_float(
                                cols[6].get_text(strip=True)
                            ),
                            "minutes_played": self._safe_int(
                                cols[7].get_text(strip=True)
                            ),
                            "season": season,
                        }
                        players_data.append(player_data)

            time.sleep(1)  # Be respectful to the server

        except Exception as e:
            logger.error(f"Error scraping Fantacalcio stats: {e}")

        return players_data

    def scrape_transfermarkt_values(self, players: List[str]) -> Dict[str, Dict]:
        """
        Scrape market values and basic info from Transfermarkt

        Args:
            players: List of player names to look up

        Returns:
            Dictionary mapping player names to their market data
        """
        market_data = {}
        base_url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"

        for player_name in players:
            try:
                logger.info(f"Fetching market data for {player_name}")

                params = {"query": player_name, "Verein_page": "serieA"}

                response = self.session.get(base_url, params=params)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")

                # Find player information
                player_row = soup.find("tr", class_="search-result-player")
                if player_row:
                    market_value_elem = player_row.find("td", class_="rechts hauptlink")
                    age_elem = player_row.find("td", class_="zentriert")

                    market_data[player_name] = {
                        "market_value": self._parse_market_value(
                            market_value_elem.get_text(strip=True)
                            if market_value_elem
                            else "0"
                        ),
                        "age": self._safe_int(
                            age_elem.get_text(strip=True) if age_elem else "0"
                        ),
                    }

                time.sleep(2)  # Longer delay for Transfermarkt

            except Exception as e:
                logger.error(f"Error fetching market data for {player_name}: {e}")
                market_data[player_name] = {"market_value": 0, "age": 0}

        return market_data

    def scrape_fbref_stats(
        self, league: str = "Serie-A", season: str = "2024-2025"
    ) -> List[Dict]:
        """
        Scrape detailed statistics from FBref

        Args:
            league: League name
            season: Season string

        Returns:
            List of dictionaries containing detailed player stats
        """
        players_data = []
        base_url = (
            f"https://fbref.com/en/comps/11/{season}/stats/{season}-{league}-Stats"
        )

        try:
            logger.info(f"Scraping FBref stats for {league} {season}")
            response = self.session.get(base_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Find stats table
            stats_table = soup.find("table", {"id": "stats_standard"})
            if stats_table:
                tbody = stats_table.find("tbody")
                rows = tbody.find_all("tr")

                for row in rows:
                    cols = row.find_all(["td", "th"])
                    if len(cols) >= 15:
                        player_data = {
                            "player_name": cols[1].get_text(strip=True),
                            "team": cols[4].get_text(strip=True),
                            "position": self._standardize_position(
                                cols[5].get_text(strip=True)
                            ),
                            "games_played": self._safe_int(
                                cols[6].get_text(strip=True)
                            ),
                            "minutes_played": self._safe_int(
                                cols[8].get_text(strip=True)
                            ),
                            "goals": self._safe_int(cols[9].get_text(strip=True)),
                            "assists": self._safe_int(cols[10].get_text(strip=True)),
                            "yellow_cards": self._safe_int(
                                cols[13].get_text(strip=True)
                            ),
                            "red_cards": self._safe_int(cols[14].get_text(strip=True)),
                            "season": season,
                        }
                        players_data.append(player_data)

            time.sleep(3)  # Respectful delay

        except Exception as e:
            logger.error(f"Error scraping FBref stats: {e}")

        return players_data

    def create_master_dataframe(
        self, seasons: List[str] = ["2024-25", "2023-24", "2022-23"]
    ) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with all player data

        Args:
            seasons: List of seasons to collect data for

        Returns:
            pandas DataFrame with structured player data
        """
        all_data = []

        for season in seasons:
            logger.info(f"Processing season {season}")

            # Collect Fantacalcio stats
            fantacalcio_data = self.scrape_fantacalcio_stats(season)

            # Collect FBref stats
            fbref_data = self.scrape_fbref_stats(season=season.replace("-", "-20"))

            # Get unique player names for market value lookup
            player_names = list(set([p["player_name"] for p in fantacalcio_data]))
            market_data = self.scrape_transfermarkt_values(player_names)

            # Merge data sources
            merged_data = self._merge_data_sources(
                fantacalcio_data, fbref_data, market_data
            )
            all_data.extend(merged_data)

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Define ideal column structure
        ideal_columns = [
            "player_name",
            "season",
            "age",
            "team",
            "position",
            "games_played",
            "minutes_played",
            "goals",
            "assists",
            "fantavoto_avg",
            "market_value",
            "yellow_cards",
            "red_cards",
        ]

        # Ensure all columns exist
        for col in ideal_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns
        df = df[ideal_columns]

        # Clean and validate data
        df = self._clean_dataframe(df)

        return df

    def _merge_data_sources(
        self,
        fantacalcio_data: List[Dict],
        fbref_data: List[Dict],
        market_data: Dict[str, Dict],
    ) -> List[Dict]:
        """Merge data from different sources"""
        merged_data = []

        # Create lookup dictionaries
        fbref_lookup = {p["player_name"]: p for p in fbref_data}

        for fc_player in fantacalcio_data:
            player_name = fc_player["player_name"]

            # Start with Fantacalcio data
            merged_player = fc_player.copy()

            # Add FBref data if available
            if player_name in fbref_lookup:
                fbref_player = fbref_lookup[player_name]
                merged_player.update(
                    {
                        "yellow_cards": fbref_player.get("yellow_cards", 0),
                        "red_cards": fbref_player.get("red_cards", 0),
                    }
                )

            # Add market data if available
            if player_name in market_data:
                merged_player.update(market_data[player_name])
            else:
                merged_player.update({"market_value": 0, "age": 0})

            merged_data.append(merged_player)

        return merged_data

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the DataFrame"""
        # Remove duplicates
        df = df.drop_duplicates(subset=["player_name", "season"])

        # Fill missing values
        numeric_columns = [
            "age",
            "games_played",
            "minutes_played",
            "goals",
            "assists",
            "fantavoto_avg",
            "market_value",
            "yellow_cards",
            "red_cards",
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Validate fantavoto_avg (should be between 0 and 10)
        df["fantavoto_avg"] = df["fantavoto_avg"].clip(0, 10)

        # Calculate derived metrics
        df["goals_per_game"] = df["goals"] / df["games_played"].replace(0, 1)
        df["assists_per_game"] = df["assists"] / df["games_played"].replace(0, 1)
        df["minutes_per_game"] = df["minutes_played"] / df["games_played"].replace(0, 1)

        return df

    def _standardize_position(self, position: str) -> str:
        """Standardize position names"""
        position = position.upper()
        if any(p in position for p in ["GK", "POR"]):
            return "GK"
        elif any(p in position for p in ["DF", "DIF", "DC", "DD", "DS"]):
            return "DF"
        elif any(p in position for p in ["MF", "CEN", "CC", "CD", "CS", "TRQ"]):
            return "MF"
        elif any(p in position for p in ["FW", "ATT", "PC", "W"]):
            return "FW"
        else:
            return "MF"  # Default to midfielder

    def _safe_int(self, value: str) -> int:
        """Safely convert string to integer"""
        try:
            return int(re.sub(r"[^\d]", "", str(value)))
        except Exception as e:
            logger.warning(f"Failed to convert to int: {value} - {e}")
            return 0

    def _safe_float(self, value: str) -> float:
        """Safely convert string to float"""
        try:
            return float(str(value).replace(",", "."))
        except Exception as e:
            logger.warning(f"Failed to convert to float: {value} - {e}")
            return 0.0

    def _parse_market_value(self, value_str: str) -> float:
        """Parse market value string (e.g., '€25.00m') to float"""
        try:
            # Remove currency symbols and convert
            value_str = re.sub(r"[€$£]", "", value_str)

            if "k" in value_str.lower():
                return float(value_str.lower().replace("k", "")) * 1000
            elif "m" in value_str.lower():
                return float(value_str.lower().replace("m", "")) * 1000000
            else:
                return float(value_str)
        except Exception as e:
            logger.warning(f"Failed to parse market value: {value_str} - {e}")
            return 0.0


def main():
    """Example usage of the data collection module"""
    collector = FantacalcioDataCollector()

    # Create master dataset
    df = collector.create_master_dataframe(seasons=["2024-25", "2023-24"])

    # Save to CSV
    df.to_csv("fantacalcio_player_data.csv", index=False)

    # Display summary
    print(f"Collected data for {len(df)} player-season records")
    print(f"Unique players: {df['player_name'].nunique()}")
    print(f"Positions: {df['position'].value_counts()}")
    print("\nDataFrame structure:")
    print(df.info())


if __name__ == "__main__":
    main()
