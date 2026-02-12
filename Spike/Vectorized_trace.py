import numpy as np
import math
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_CAP = ALPHABET.upper()
DIGITS = "0123456789"
SPECIAL_CHARS = " .,!?;:'\"()[]{}<>@#$%^&*-_=+|\\/`~"
END_TOKEN = "<END>"

LETTERS = list(ALPHABET + ALPHABET_CAP + DIGITS + SPECIAL_CHARS)
ALL_CHARS = LETTERS + [END_TOKEN]

INITIAL_SIGNAL = 5.0
DECAY_RATE = 0.9
STATE_DECAY = 0.0

STATE_DIM = 64
SIM_THRESHOLD = 0.85
TOP_K = 3
TEMPERATURE = 1.0
MAX_PROTOTYPES = 200

# ============================================================
# STATE PROJECTION
# ============================================================

STATE_SIZE = len(ALL_CHARS)
np.random.seed(42)
PROJECTION = np.random.randn(STATE_DIM, STATE_SIZE) / math.sqrt(STATE_SIZE)


def project_state(state_vector):
    v = PROJECTION @ state_vector
    norm = np.linalg.norm(v) + 1e-8
    return v / norm


# ============================================================
# PROTOTYPE CONNECTOR
# ============================================================


class PrototypeConnector:

    def __init__(self, output_char):
        self.output_char = output_char
        self.keys = []
        self.values = []
        self.counts = []

    def learn(self, key_vec, value):

        if not self.keys:
            self._add(key_vec, value)
            return

        sims = [np.dot(k, key_vec) for k in self.keys]
        idx = int(np.argmax(sims))
        best_sim = sims[idx]

        if best_sim > SIM_THRESHOLD:
            c = self.counts[idx]
            self.keys[idx] = (self.keys[idx] * c + key_vec) / (c + 1)
            self.keys[idx] /= np.linalg.norm(self.keys[idx]) + 1e-8
            self.values[idx] = (self.values[idx] * c + value) / (c + 1)
            self.counts[idx] += 1
        else:
            self._add(key_vec, value)

        if len(self.keys) > MAX_PROTOTYPES:
            remove_idx = int(np.argmin(self.counts))
            self._remove(remove_idx)

    def _add(self, key, value):
        self.keys.append(key.copy())
        self.values.append(value)
        self.counts.append(1)

    def _remove(self, idx):
        self.keys.pop(idx)
        self.values.pop(idx)
        self.counts.pop(idx)

    def transmit(self, key_vec):
        if not self.keys:
            return 0.0

        sims = np.array([np.dot(k, key_vec) for k in self.keys])

        top_idx = sims.argsort()[-TOP_K:]
        top_sims = sims[top_idx]

        weights = np.exp(top_sims / TEMPERATURE)
        weights /= weights.sum() + 1e-8

        weighted = 0.0
        for w, i in zip(weights, top_idx):
            weighted += w * self.values[i]

        return weighted


# ============================================================
# LAYERS
# ============================================================


class Layer:
    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def step(self, decay):
        self.signal *= decay

    def receive(self, val):
        self.signal += val


# ============================================================
# MODEL
# ============================================================


class TraceModel:

    def __init__(self):

        self.input_layers = {c: Layer(c) for c in ALL_CHARS}
        self.output_layers = {c: Layer(c) for c in ALL_CHARS}
        self.state_layers = {c: Layer(c) for c in ALL_CHARS}

        # sparse connectors
        self.connectors = defaultdict(dict)

    def reset(self):
        for d in [self.input_layers, self.output_layers, self.state_layers]:
            for l in d.values():
                l.signal = 0.0

    def step_all(self):
        for l in self.input_layers.values():
            l.step(DECAY_RATE)
        for l in self.output_layers.values():
            l.step(DECAY_RATE)
        for l in self.state_layers.values():
            l.step(STATE_DECAY)

    def state_vector(self):
        return np.array([self.state_layers[c].signal for c in ALL_CHARS])

    def learn(self, question, answer):

        for ch in question:
            if ch not in self.input_layers:
                return

        answer_tokens = list(answer) + [END_TOKEN]

        for ch in question:
            self.input_layers[ch].receive(INITIAL_SIGNAL)
            self.step_all()

        for out_ch in answer_tokens:

            key = project_state(self.state_vector())

            for qch in question:
                conn = self.connectors[qch].get(out_ch)
                if conn is None:
                    conn = PrototypeConnector(out_ch)
                    self.connectors[qch][out_ch] = conn

                val = self.input_layers[qch].signal
                conn.learn(key, val)

            self.state_layers[out_ch].receive(INITIAL_SIGNAL)
            self.step_all()

    def generate(self, max_len=200):

        output_text = []

        for _ in range(max_len):

            key = project_state(self.state_vector())

            for out_layer in self.output_layers.values():
                out_layer.signal = 0.0

            for qch, layer in self.input_layers.items():
                if layer.signal <= 0:
                    continue

                conns = self.connectors.get(qch, {})
                for out_ch, conn in conns.items():
                    val = conn.transmit(key)
                    self.output_layers[out_ch].receive(val * layer.signal)

            best_char = None
            best_signal = -1

            for ch, layer in self.output_layers.items():
                if layer.signal > best_signal:
                    best_signal = layer.signal
                    best_char = ch

            if best_char is None or best_char == END_TOKEN:
                break

            output_text.append(best_char)

            self.state_layers[best_char].receive(INITIAL_SIGNAL)
            self.step_all()

        return "".join(output_text)


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    model = TraceModel()

    model.learn("hello", "hi there")
    model.learn("how are you", "i am fine")

    for ch in "hello":
        model.input_layers[ch].receive(INITIAL_SIGNAL)
        model.step_all()

    print("Response:", model.generate())
