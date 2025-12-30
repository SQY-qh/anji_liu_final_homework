import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from labs_rnn.test_model import train_mamba


if __name__ == "__main__":
    train_mamba()
