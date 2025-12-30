import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from labs_rnn.utils.data_loader import DataLoader
from labs_rnn.models.min_gru import MinGRU
from labs_rnn.models.mamba import Mamba


model_type = "min_gru"
seq_len = 32
hidden_size = 128
output_size = 1
learning_rate = 0.01
batch_size = 32
n_epochs = 5


def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)


def train_min_gru():
    loader = DataLoader()
    data_info = loader.load_yahoo_stock(ticker="AAPL")
    batches = loader.create_stock_batches(data_info["data"], seq_len, batch_size)
    split = int(len(batches) * 0.8)
    train_batches = batches[:split]
    valid_batches = batches[split:]
    model = MinGRU(input_size=1, hidden_size=hidden_size, output_size=output_size)
    train_losses = []
    valid_losses = []
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for x, y in train_batches:
            h = model.zero_hidden(x.shape[2])
            outputs = []
            for t in range(seq_len):
                y_t, h = model.forward_step(x[t], h, t)
                outputs.append(y_t)
            y_pred = outputs[-1]
            loss = mse_loss(y_pred, y)
            total_loss += loss
            dy = (y_pred - y) / y_pred.shape[1]
            dh_next = np.zeros_like(h)
            for t in range(seq_len - 1, -1, -1):
                if t == seq_len - 1:
                    _, dh_next = model.backward_step(dy, dh_next, t)
                else:
                    zeros_dy = np.zeros_like(dy)
                    _, dh_next = model.backward_step(zeros_dy, dh_next, t)
            model.update(learning_rate, weight_decay=0.0001)
        avg_train = total_loss / max(1, len(train_batches))
        train_losses.append(avg_train)
        v_loss = 0.0
        for x, y in valid_batches:
            h = model.zero_hidden(x.shape[2])
            for t in range(seq_len):
                y_t, h = model.forward_step(x[t], h, t)
            y_pred = y_t
            v_loss += mse_loss(y_pred, y)
        avg_valid = v_loss / max(1, len(valid_batches))
        valid_losses.append(avg_valid)
        print(f"Epoch {epoch} | Train Loss {avg_train:.6f} | Valid Loss {avg_valid:.6f}")
    return {"train_losses": train_losses, "valid_losses": valid_losses}


def train_mamba():
    loader = DataLoader()
    data_info = loader.load_yahoo_stock(ticker="AAPL")
    batches = loader.create_stock_batches(data_info["data"], seq_len, batch_size)
    split = int(len(batches) * 0.8)
    train_batches = batches[:split]
    valid_batches = batches[split:]
    model = Mamba(input_size=1, hidden_size=hidden_size, output_size=output_size)
    train_losses = []
    valid_losses = []
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for x, y in train_batches:
            out = model.forward(x)
            y_pred = out[-1]
            loss = mse_loss(y_pred, y)
            total_loss += loss
            doutput = np.zeros_like(out)
            doutput[-1] = (y_pred - y) / y_pred.shape[1]
            model.backward(doutput)
            model.update(learning_rate, weight_decay=0.0001)
        avg_train = total_loss / max(1, len(train_batches))
        train_losses.append(avg_train)
        v_loss = 0.0
        for x, y in valid_batches:
            out = model.forward(x)
            y_pred = out[-1]
            v_loss += mse_loss(y_pred, y)
        avg_valid = v_loss / max(1, len(valid_batches))
        valid_losses.append(avg_valid)
        print(f"Epoch {epoch} | Train Loss {avg_train:.6f} | Valid Loss {avg_valid:.6f}")
    return {"train_losses": train_losses, "valid_losses": valid_losses}


if __name__ == "__main__":
    if model_type == "min_gru":
        train_min_gru()
    else:
        train_mamba()
