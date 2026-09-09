import yfinance as yf
import pandas as pd
import os

def download_financial_data(ticker_symbol, start_date, end_date, save_path):
    """
    Downloads historical financial data from Yahoo Finance and saves it as a CSV.
    """
    print(f"Downloading data for {ticker_symbol}...")
    
    try:
        # Download data using yfinance
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: No data found for {ticker_symbol}.")
            return
            
        # Check if the directory exists, if not, create it
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the dataframe to a CSV file
        data.to_csv(save_path)
        print(f"Data successfully saved to: {save_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example: Download 10 years of S&P 500 index historical data (^GSPC)
    TICKER = "^GSPC"
    START_DATE = "2016-01-01"
    END_DATE = "2026-01-01"
    FILE_PATH = "data/raw/sp500_historical.csv"
    
    download_financial_data(TICKER, START_DATE, END_DATE, FILE_PATH)