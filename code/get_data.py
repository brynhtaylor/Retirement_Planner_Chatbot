import pandas as pd
import wrds
from pathlib import Path
from dotenv import load_dotenv
import os

# Paths
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load WRDS credentials
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

def connect_wrds():
    print("Connecting to WRDS...")
    
    wrds_username = os.getenv("WRDS_USERNAME")
    wrds_password = os.getenv("WRDS_PASSWORD")

    db = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    return db

def get_stock_returns(db):
    print("Downloading CRSP market index monthly returns...")

    query = """
        SELECT date, vwretd
        FROM crsp.msi
        WHERE date >= '2000-01-01'
    """
    sp500 = db.raw_sql(query)
    sp500.rename(columns={"vwretd": "sp500_return"}, inplace=True)
    return sp500

def get_bond_returns(db):
    print("Downloading LQD (Corporate Bond ETF) returns from CRSP...")
    query = """
        SELECT date, ret
        FROM crsp.msf
        WHERE permno = 89467
          AND date >= '2000-01-01'
    """
    bonds = db.raw_sql(query)
    bonds.rename(columns={"ret": "corp_bond_return"}, inplace=True)
    return bonds

def get_risk_free_rate(db):
    print("Downloading and converting 3-month T-Bill YTM from CRSP...")
    query = """
        SELECT mcaldt, tmbidytm
        FROM crsp.tfz_mth_rf
        WHERE mcaldt >= '2000-01-01'
          AND tmbidytm IS NOT NULL
    """
    rf = db.raw_sql(query)
    rf.rename(columns={"mcaldt": "date", "tmbidytm": "risk_free_rate"}, inplace=True)
    
    # Convert to monthly return
    rf["risk_free_return"] = rf["risk_free_rate"] / 100 / 12
    
    # Deduplicate: average across any same-month values
    rf = rf.groupby("date", as_index=False).mean(numeric_only=True)

    return rf[["date", "risk_free_return"]]

def preprocess_for_optimization(df):
    print("Preprocessing market data for portfolio optimization...")

    # Drop rows with missing values
    df_clean = df.dropna().copy()

    # Compute excess returns
    df_clean["stock_excess"] = df_clean["sp500_return"] - df_clean["risk_free_return"]
    df_clean["bond_excess"] = df_clean["corp_bond_return"] - df_clean["risk_free_return"]

    # Compute expected returns (monthly)
    mean_returns = df_clean[["stock_excess", "bond_excess"]].mean()

    # Compute covariance matrix (monthly)
    cov_matrix = df_clean[["stock_excess", "bond_excess"]].cov()

    # Convert to annualized (assuming monthly data)
    annualized_returns = mean_returns * 12
    annualized_cov = cov_matrix * 12

    print("Expected Annual Returns:")
    print(annualized_returns)
    print("\nAnnual Covariance Matrix:")
    print(annualized_cov)

    # Save for optimization model
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(processed_path / "cleaned_returns.csv", index=False)
    annualized_returns.to_csv(processed_path / "expected_returns.csv")
    annualized_cov.to_csv(processed_path / "cov_matrix.csv")

    return df_clean, annualized_returns, annualized_cov

def main():
    db = connect_wrds()
    sp500 = get_stock_returns(db)
    bonds = get_bond_returns(db)
    rf = get_risk_free_rate(db)

    # Merge on date
    merged = sp500.merge(bonds, on="date", how="outer")
    merged = merged.merge(rf, on="date", how="outer")
    merged.sort_values("date", inplace=True)

    merged.drop_duplicates(subset="date", keep="first", inplace=True)
    merged.dropna(inplace=True)
    merged = merged.sort_values("date").reset_index(drop=True)

    # Save to CSV
    merged.to_csv(RAW_DATA_DIR / "market_data.csv", index=False)
    print("Data saved to:", RAW_DATA_DIR / "market_data.csv")

    preprocess_for_optimization(merged)

if __name__ == "__main__":
    main()