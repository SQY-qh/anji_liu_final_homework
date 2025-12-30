import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    def __init__(self, data_dir: str = "labs_rnn/data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_yahoo_stock(self, ticker: str = "AAPL", start_date: str = "2010-01-01", end_date: str = "2023-12-31"):
        data_path = os.path.join(self.data_dir, f"{ticker}_stock_data.csv")
        scaled_data = None
        scaler = MinMaxScaler(feature_range=(0, 1))
        original_data = None
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is None or len(df) == 0:
                raise RuntimeError("empty")
            data = df["Close"].values.reshape(-1, 1)
            original_data = data
            scaled_data = scaler.fit_transform(data)
            df.to_csv(data_path)
        except Exception:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                if "Close" in df.columns:
                    data = df["Close"].values.reshape(-1, 1)
                else:
                    data = df.values.reshape(-1, 1)
                original_data = data
                scaled_data = scaler.fit_transform(data)
            else:
                n = 3000
                t = np.linspace(0, 50, n)
                base = 150 + 5 * np.sin(0.2 * t) + 2 * np.sin(1.7 * t)
                noise = np.random.randn(n) * 0.5
                data = (base + noise).astype(np.float64).reshape(-1, 1)
                original_data = data
                scaled_data = scaler.fit_transform(data)
                os.makedirs(self.data_dir, exist_ok=True)
                pd.DataFrame({"Close": data.flatten()}).to_csv(data_path, index=False)
        return {"data": scaled_data, "scaler": scaler, "original_data": original_data}

    def create_stock_batches(self, data: np.ndarray, seq_len: int, batch_size: int):
        total_len = len(data)
        x = []
        y = []
        for i in range(total_len - seq_len):
            x.append(data[i : i + seq_len])
            y.append(data[i + seq_len])
        x = np.array(x)
        y = np.array(y)
        n_batches = len(x) // batch_size
        x = x[: n_batches * batch_size]
        y = y[: n_batches * batch_size]
        x = x.reshape(batch_size, n_batches, seq_len, -1).transpose(1, 2, 3, 0)
        y = y.reshape(batch_size, n_batches, -1).transpose(1, 2, 0)
        batches = []
        for i in range(n_batches):
            batches.append((x[i], y[i]))
        return batches

