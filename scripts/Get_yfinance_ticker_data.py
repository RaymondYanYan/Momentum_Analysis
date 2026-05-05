import yfinance as yf
import pandas as pd
import numpy as np

class PriceDataCollector:
    """Standalone, high-performance price and volume collector for EC2"""
    
    def __init__(self, tickers=['AAPL'], start='2024-01-01', end='2030-12-31'):
        self.tickers = [tickers] if isinstance(tickers, str) else tickers
        self.start = start
        self.end = end

    def collect_data(self) -> pd.DataFrame:
        print(f"--- Downloading data for: {self.tickers} ---")
        
        # Download data
        data = yf.download(
            tickers=self.tickers,
            start=self.start,
            end=self.end,
            interval='1d',
            auto_adjust=True,
            group_by='ticker', 
            progress=False
        )
        
        if data.empty:
            raise ValueError("YFinance returned no data. Check ticker symbols or dates.")

        # 1. Standardize Structure (Flattening MultiIndex)
        if len(self.tickers) == 1:
            df = data.reset_index()
            df['ticker'] = self.tickers[0]
        else:
            # stack(future_stack=True) is efficient for pandas 2.0+
            df = data.stack(level=0, future_stack=True).reset_index()
            df.rename(columns={'level_1': 'ticker'}, inplace=True)

        # 2. Clean Column Names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

        # 3. Vectorized Math (High Performance / Low Memory)
        print("--- Calculating Metrics ---")
        grouped = df.groupby('ticker', group_keys=False)

        # Log Returns
        df['log_return'] = grouped['close'].apply(lambda x: np.log(x).diff())

        # Volume Momentum (20-day)
        rolling_vol = grouped['volume'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        df['vol_momentum'] = df['volume'] / rolling_vol

        # Intra-day Range %
        df['day_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Distance from 52-Week High
        rolling_max = grouped['high'].transform(lambda x: x.rolling(window=252, min_periods=1).max())
        df['dist_from_peak'] = (df['high'] / rolling_max) - 1

        # 4. Final Cleanup
        df.dropna(subset=['close', 'volume'], inplace=True)
        
        print(f"--- Success: {len(df)} rows collected ---")
        return df



## Example on how to run
if __name__ == "__main__":
    try:
        collector = PriceDataCollector(
            tickers=['AAPL', 'TSLA', 'BTC-USD'], # input the ticker
            start='2024-01-01' # input the start date
        )
        final_df = collector.collect_data()
        
        print("\n--- SAMPLE DATA ---")
        print(final_df.tail())
        
    except Exception as e:
        print(f"Error occurred: {e}")