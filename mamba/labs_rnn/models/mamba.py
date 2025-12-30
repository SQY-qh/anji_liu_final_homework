import numpy as np


class Mamba:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, state_size: int = 64, kernel_size: int = 4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_size = state_size
        self.kernel_size = kernel_size
        self.W_in = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / max(1, input_size))
        self.A = np.ones((state_size, 1)) * -1.0
        self.B = np.random.randn(hidden_size, state_size) * np.sqrt(2.0 / max(1, hidden_size))
        self.C = np.random.randn(hidden_size, state_size) * np.sqrt(2.0 / max(1, state_size))
        self.D = np.random.randn(hidden_size, 1) * 0.1
        self.W_out = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / max(1, hidden_size))
        self.b_out = np.zeros((output_size, 1))
        self.reset_grads()
        self.grad_history = []

    def reset_grads(self):
        self.dW_in = np.zeros_like(self.W_in)
        self.dB = np.zeros_like(self.B)
        self.dC = np.zeros_like(self.C)
        self.dD = np.zeros_like(self.D)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)

    def silu(self, x):
        z = np.clip(x, -50.0, 50.0)
        s = 1.0 / (1.0 + np.exp(-z))
        return x * s

    def dsilu(self, x):
        z = np.clip(x, -50.0, 50.0)
        s = 1.0 / (1.0 + np.exp(-z))
        return s + x * s * (1.0 - s)

    def forward(self, x):
        seq_len, input_size, batch_size = x.shape
        x_proj = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            x_proj[t] = np.dot(self.W_in, x[t])
        s = np.zeros((self.state_size, batch_size))
        y = np.zeros((seq_len, self.hidden_size, batch_size))
        self.s_history = []
        self.gate_history = []
        self.x_proj = x_proj
        self.x = x
        for t in range(seq_len):
            gate = np.clip(self.silu(x_proj[t]), -5.0, 5.0)
            s = s * np.exp(self.A) + np.dot(self.B.T, gate)
            u = np.dot(self.C, s) + self.D * x_proj[t]
            y[t] = gate * np.clip(u, -10.0, 10.0)
            self.s_history.append(s.copy())
            self.gate_history.append(gate.copy())
        output = np.zeros((seq_len, self.output_size, batch_size))
        for t in range(seq_len):
            output[t] = np.dot(self.W_out, y[t]) + self.b_out
        self.y = y
        self.output = output
        return output

    def backward(self, doutput):
        seq_len, _, batch_size = doutput.shape
        dy = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            self.dW_out += np.dot(doutput[t], self.y[t].transpose(1, 0))
            self.db_out += np.sum(doutput[t], axis=1, keepdims=True)
            dy[t] = np.dot(self.W_out.T, doutput[t])
        ds_next = np.zeros((self.state_size, batch_size))
        for t in range(seq_len - 1, -1, -1):
            gate = self.gate_history[t]
            s_t = self.s_history[t]
            x_p = self.x_proj[t]
            u = np.dot(self.C, s_t) + self.D * x_p
            du = gate * dy[t]
            dgate = u * dy[t]
            self.dC += np.dot(du, s_t.transpose(1, 0))
            ds = np.dot(self.C.T, du) + ds_next
            self.dD += np.sum(du * x_p, axis=1, keepdims=True)
            dxp_from_u = du * self.D
            dgate += np.dot(self.B, ds)
            dxp_from_gate = dgate * self.dsilu(x_p)
            dxp = dxp_from_u + dxp_from_gate
            self.dW_in += np.dot(dxp, self.x[t].transpose(1, 0))
            self.dB += np.dot(gate, ds.transpose(1, 0))
            ds_next = ds * np.exp(self.A)
        return None

    def update(self, lr: float, weight_decay: float = 0.0):
        self.W_in -= lr * (self.dW_in + weight_decay * self.W_in)
        self.B -= lr * (self.dB + weight_decay * self.B)
        self.C -= lr * (self.dC + weight_decay * self.C)
        self.D -= lr * (self.dD + weight_decay * self.D)
        self.W_out -= lr * (self.dW_out + weight_decay * self.W_out)
        self.b_out -= lr * self.db_out
        self.grad_history.append({
            "W_in": float(np.linalg.norm(self.dW_in)),
            "B": float(np.linalg.norm(self.dB)),
            "C": float(np.linalg.norm(self.dC)),
            "D": float(np.linalg.norm(self.dD)),
            "W_out": float(np.linalg.norm(self.dW_out)),
        })
        self.reset_grads()
