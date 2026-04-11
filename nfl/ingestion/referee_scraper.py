"""
Scrapes referee game assignments from nflpenalties.com.
"""
import logging
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

REFEREES = [
    "Tony-Corrente", "Brad-Allen", "Tra-Blake", "Clete-Blakeman",
    "Jerome-Boger", "Carl-Cheffers", "Land-Clark", "Adrian-Hill",
    "Shawn-Hochuli", "John-Hussey", "Alex-Kemp", "Clay-Martin",
    "Scott-Novak", "Brad-Rogers", "Shawn-Smith", "Ron-Torbert",
    "Bill-Vinovich", "Craig-Wrolstad",
]

REFEREE_IDS = {
    "Adrian-Hill": 1, "Alex-Kemp": 2, "Bill-Vinovich": 3,
    "Brad-Allen": 4, "Brad-Rogers": 5, "Carl-Cheffers": 6,
    "Clay-Martin": 7, "Clete-Blakeman": 8, "Craig-Wrolstad": 9,
    "Jerome-Boger": 10, "John-Hussey": 11, "Ron-Torbert": 12,
    "Scott-Novak": 13, "Shawn-Hochuli": 15, "Shawn-Smith": 16,
    "Tra-Blake": 17, "Land-Clark": 18, "Tony-Corrente": 19,
}

TEAM_NAME_MAP = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF",
    "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE",
    "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Green Bay": "GB",
    "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX",
    "Kansas City": "KC", "Miami": "MIA", "Minnesota": "MIN",
    "New England": "NE", "New Orleans": "NO", "N.Y. Giants": "NYG",
    "N.Y. Jets": "NYJ", "Las Vegas": "LV", "Philadelphia": "PHI",
    "Pittsburgh": "PIT", "LA Chargers": "LAC", "Seattle": "SEA",
    "San Francisco": "SF", "LA Rams": "LAR", "Tampa Bay": "TB",
    "Tennessee": "TEN", "Washington": "WAS",
}


def fetch_referee_assignments(years: list = None) -> pd.DataFrame:
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]

    rows = []
    for ref in REFEREES:
        for year in years:
            url = f"https://www.nflpenalties.com/referee/{ref}?year={year}"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.text, "html.parser")
                table = soup.find("table", {"class": "footable"})
                if not table:
                    continue
                for row in table.find_all("tr"):
                    cols = row.find_all("td")
                    if len(cols) < 4:
                        continue
                    date = cols[0].text.strip()
                    week = cols[1].text.strip()
                    team = cols[2].text.strip()
                    away = cols[3].text.strip()
                    if date == "Totals" or not date:
                        continue
                    rows.append({
                        "Day": date,
                        "Week": week,
                        "FullName": team,
                        "AwayTeam": away,
                        "Referee": ref,
                    })
            except Exception as e:
                log.warning(f"Failed to scrape {ref} {year}: {e}")
            time.sleep(2)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["HomeTeamName"] = df["FullName"].map(TEAM_NAME_MAP)
    df["Referee"] = df["Referee"].map(REFEREE_IDS)
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce")
    df.dropna(subset=["Day", "HomeTeamName"], inplace=True)
    df["month"] = df["Day"].dt.month
    df["year"] = df["Day"].dt.year
    df["dayofyear"] = df["Day"].dt.dayofyear
    log.info(f"Scraped {len(df)} referee assignment rows")
    return df[["HomeTeamName", "month", "year", "dayofyear", "Referee"]]
