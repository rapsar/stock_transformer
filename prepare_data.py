import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler	
import ta
from typing import Union, List, Optional, Tuple
from pathlib import Path


def load_forex_data(csv_file: Union[str, Path]) -> pd.DataFrame:
    """
    Loads the CSV file into a Pandas DataFrame.
    Ensures it is sorted by date ascending.
    """
    df = pd.read_csv(csv_file)
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
    df.sort_values(by='Gmt time', inplace=True)

    # Reset index to ensure a clean index
    df.reset_index(drop=True, inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds some sample technical indicators using the `ta` library.
    Adjust the window periods and indicators to your liking.
    """
    # Ensure required column exists
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    # Moving Average Slope
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_20_slope'] = df['ma_20'].diff()

    # Fill any NaNs from indicator calculations
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df


def select_and_scale_features(
    df: pd.DataFrame, 
    feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, MinMaxScaler, List[str]]:
    """
    Given a DataFrame, selects the relevant columns and applies MinMax scaling.
    Returns the scaled array, the fitted scaler (for inversing later), and the list of columns used.
    """
    if feature_cols is None:
        # default feature set: O,H,L,C and a few indicators
        feature_cols = ['Open', 'High', 'Low', 'Close', 
                        'rsi', 'bb_high', 'bb_low', 'ma_20', 'ma_20_slope']

    data = df[feature_cols].values  # shape: (num_samples, num_features)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler, feature_cols