import numpy as np


class MinGRU:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W_z = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.W_r = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.W_h = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_z = np.zeros((hidden_size, 1))
        self.b_r = np.zeros((hidden_size, 1))
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        self.reset_grads()
        self.caches = {}

    def reset_grads(self):
        self.dW_z = np.zeros_like(self.W_z)
        self.dW_r = np.zeros_like(self.W_r)
        self.dW_h = np.zeros_like(self.W_h)
        self.dW_y = np.zeros_like(self.W_y)
        self.db_z = np.zeros_like(self.b_z)
        self.db_r = np.zeros_like(self.b_r)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def zero_hidden(self, batch_size: int):
        return np.zeros((self.hidden_size, batch_size))

    def forward_step(self, x_t, h_prev, t_idx):
        combined = np.concatenate([h_prev, x_t], axis=0)
        z = self.sigmoid(np.dot(self.W_z, combined) + self.b_z)
        r = self.sigmoid(np.dot(self.W_r, combined) + self.b_r)
        combined_r = np.concatenate([r * h_prev, x_t], axis=0)
        h_tilde = self.tanh(np.dot(self.W_h, combined_r) + self.b_h)
        h = (1.0 - z) * h_prev + z * h_tilde
        y = np.dot(self.W_y, h) + self.b_y
        self.caches[t_idx] = {
            "combined": combined,
            "z": z,
            "r": r,
            "combined_r": combined_r,
            "h_tilde": h_tilde,
            "h_prev": h_prev,
            "h": h,
        }
        return y, h

    def backward_step(self, dy, dh_next, t_idx):
        cache = self.caches[t_idx]
        h_prev = cache["h_prev"]
        h = cache["h"]
        z = cache["z"]
        r = cache["r"]
        combined = cache["combined"]
        combined_r = cache["combined_r"]
        h_tilde = cache["h_tilde"]
        self.dW_y += np.dot(dy, h.T)
        self.db_y += np.sum(dy, axis=1, keepdims=True)
        dh = np.dot(self.W_y.T, dy) + dh_next
        dh_tilde = dh * z
        dz = dh * (h_prev - h_tilde)
        dz_sigmoid = z * (1.0 - z) * dz
        self.dW_z += np.dot(dz_sigmoid, combined.T)
        self.db_z += np.sum(dz_sigmoid, axis=1, keepdims=True)
        dh_tilde_tanh = (1.0 - h_tilde ** 2) * dh_tilde
        self.dW_h += np.dot(dh_tilde_tanh, combined_r.T)
        self.db_h += np.sum(dh_tilde_tanh, axis=1, keepdims=True)
        dr_combined = np.dot(self.W_h.T, dh_tilde_tanh)
        dr = dr_combined[: self.hidden_size] * h_prev
        dr_sigmoid = r * (1.0 - r) * dr
        self.dW_r += np.dot(dr_sigmoid, combined.T)
        self.db_r += np.sum(dr_sigmoid, axis=1, keepdims=True)
        dx_combined = np.dot(self.W_z.T, dz_sigmoid) + np.dot(self.W_r.T, dr_sigmoid)
        dx = dx_combined[self.hidden_size :]
        dh_prev = dx_combined[: self.hidden_size] + dh * (1.0 - z)
        return dx, dh_prev

    def update(self, lr: float, weight_decay: float = 0.0, clip: float | None = 1.0):
        if clip is not None:
            for g in [self.dW_z, self.dW_r, self.dW_h, self.dW_y, self.db_z, self.db_r, self.db_h, self.db_y]:
                np.clip(g, -clip, clip, out=g)
        self.W_inplace_update(self.W_z, self.dW_z, lr, weight_decay)
        self.W_inplace_update(self.W_r, self.dW_r, lr, weight_decay)
        self.W_inplace_update(self.W_h, self.dW_h, lr, weight_decay)
        self.W_inplace_update(self.W_y, self.dW_y, lr, weight_decay)
        self.b_inplace_update(self.b_z, self.db_z, lr)
        self.b_inplace_update(self.b_r, self.db_r, lr)
        self.b_inplace_update(self.b_h, self.db_h, lr)
        self.b_inplace_update(self.b_y, self.db_y, lr)
        self.reset_grads()

    def W_inplace_update(self, W, dW, lr, wd):
        W -= lr * (dW + wd * W)

    def b_inplace_update(self, b, db, lr):
        b -= lr * db

