import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from labs_rnn.models.mamba import Mamba


if __name__ == "__main__":
    seq_len = 8
    input_size = 1
    batch_size = 4
    x = np.random.randn(seq_len, input_size, batch_size)
    model = Mamba(input_size=1, hidden_size=16, output_size=1)
    out = model.forward(x)
    print(out.shape)
