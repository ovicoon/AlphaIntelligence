import math
import csv
import re
import os
import sys
import bisect
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_CAP = ALPHABET.upper()
DIGITS = "0123456789"
SPECIAL_CHARS = " .,!?;:'\"()[]{}<>@#$%^&*-_=+|\\/`~"
LETTERS = ALPHABET + ALPHABET_CAP + DIGITS + SPECIAL_CHARS
END_TOKEN = "<END>"

DECAY_RATE = 0.9
STATE_DECAY_RATE = 0.0
SIMILARITY_K = 20.0
INITIAL_SIGNAL_STRENGTH = 5.0
MIN_SIGNAL_THRESHOLD = 0.0
STATE_INFLUENCE = 0.0
MAX_HISTORY = float("inf")

AUTO = False

# ============================================================
# CONNECTOR (Sparse + LogN)
# ============================================================


class Connector:
    __slots__ = ("output_id", "signals", "states")

    def __init__(self, output_id):
        self.output_id = output_id
        self.signals = []
        self.states = []

    def transmit(self, signal, current_state, output_layers):
        if not self.signals:
            return

        pos = bisect.bisect_left(self.signals, signal)

        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(self.signals):
            candidates.append(pos)

        if not candidates:
            return

        best_idx = min(candidates, key=lambda i: abs(self.signals[i] - signal))

        min_dist = abs(self.signals[best_idx] - signal)
        weight1 = max(0.0, 1.0 - min_dist / SIMILARITY_K)

        if STATE_INFLUENCE > 0:
            snapshot = self.states[best_idx]
            diff_sum = 0.0
            for i in range(len(snapshot)):
                diff_sum += abs(snapshot[i] - current_state[i])
            avg_diff = diff_sum / len(snapshot)
            weight2 = max(0.0, 1.0 - avg_diff / SIMILARITY_K)

            alpha = 1.0 - STATE_INFLUENCE
            beta = STATE_INFLUENCE
            final_weight = (weight1**alpha) * (weight2**beta)
        else:
            final_weight = weight1

        output_layers[self.output_id].signal += signal * final_weight

    def learn(self, signal, state_snapshot):
        pos = bisect.bisect_left(self.signals, signal)
        self.signals.insert(pos, signal)
        self.states.insert(pos, state_snapshot[:])

        if len(self.signals) > MAX_HISTORY:
            self.signals.pop(0)
            self.states.pop(0)


# ============================================================
# LAYERS
# ============================================================


class InputLayer:
    __slots__ = ("signal", "connections")

    def __init__(self):
        self.signal = 0.0
        self.connections = {}

    def step(self):
        self.signal *= DECAY_RATE

    def receive(self, amount):
        self.signal += amount


class OutputLayer:
    __slots__ = ("signal",)

    def __init__(self):
        self.signal = 0.0

    def step(self):
        self.signal *= DECAY_RATE


class StateLayer:
    __slots__ = ("signal",)

    def __init__(self):
        self.signal = 0.0

    def step(self):
        self.signal *= STATE_DECAY_RATE


# ============================================================
# GLOBAL INIT
# ============================================================

ALL_CHARS = list(LETTERS) + [END_TOKEN]

input_layers = {c: InputLayer() for c in ALL_CHARS}
output_layers = {c: OutputLayer() for c in ALL_CHARS}
state_layers = {c: StateLayer() for c in ALL_CHARS}

# ============================================================
# UTIL
# ============================================================


def step_all():
    for l in input_layers.values():
        l.step()
    for l in output_layers.values():
        l.step()
    for l in state_layers.values():
        l.step()


def reset_all():
    for l in input_layers.values():
        l.signal = 0.0
    for l in output_layers.values():
        l.signal = 0.0
    for l in state_layers.values():
        l.signal = 0.0


def tokenize(text):
    tokens = []
    i = 0
    L = len(text)
    end_len = len(END_TOKEN)

    while i < L:
        if text[i : i + end_len] == END_TOKEN:
            tokens.append(END_TOKEN)
            i += end_len
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def get_state_vector():
    return [state_layers[c].signal for c in ALL_CHARS]


def find_best_output():
    best_char = None
    best_val = 0.0
    for ch, layer in output_layers.items():
        if layer.signal > best_val:
            best_val = layer.signal
            best_char = ch
    return best_char, best_val


def parse_dialog(raw_text):
    pattern = r"'([^']+)'|\"([^\"]+)\""
    matches = re.findall(pattern, raw_text)
    dialog = []
    for match in matches:
        text = match[0] if match[0] else match[1]
        text = text.strip()
        if text:
            dialog.append(text)
    return dialog


# ============================================================
# LEARN
# ============================================================


def learn(question, answer):
    answer_tokens = tokenize(answer) + [END_TOKEN]

    for ch in question:
        if ch not in input_layers:
            return
        input_layers[ch].receive(INITIAL_SIGNAL_STRENGTH)
        step_all()

    for out_char in answer_tokens:
        if out_char not in output_layers:
            return

        state_snapshot = get_state_vector()

        for q_char in question:
            layer = input_layers[q_char]

            if out_char not in layer.connections:
                layer.connections[out_char] = Connector(out_char)

            layer.connections[out_char].learn(layer.signal, state_snapshot)

        state_layers[out_char].signal += INITIAL_SIGNAL_STRENGTH
        step_all()


# ============================================================
# GENERATE
# ============================================================


def generate(max_len=2048):
    result = []

    for _ in range(max_len):

        for l in output_layers.values():
            l.signal = 0.0

        current_state = get_state_vector()

        for ch, layer in input_layers.items():
            if layer.signal == 0:
                continue
            for conn in layer.connections.values():
                conn.transmit(layer.signal, current_state, output_layers)

        best_char, best_val = find_best_output()

        if best_val < MIN_SIGNAL_THRESHOLD:
            break
        if best_char == END_TOKEN:
            break

        result.append(best_char)

        state_layers[best_char].signal += INITIAL_SIGNAL_STRENGTH
        step_all()

    return "".join(result)


# ============================================================
# CSV LEARNING
# ============================================================


def learn_from_csv(file_path, max_dialogs=None):
    if not os.path.isfile(file_path):
        print("❌ file not found")
        return

    print(f"📂 learning from {file_path}")
    start = datetime.now()

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        total = 0
        pairs = 0

        for row in reader:
            if max_dialogs and total >= max_dialogs:
                break

            raw = row.get("dialog", "")
            dialog = parse_dialog(raw)

            if len(dialog) < 2:
                continue

            reset_all()

            for i in range(len(dialog) - 1):
                learn(dialog[i], dialog[i + 1])
                pairs += 1

            total += 1

            if total % 10 == 0:
                print(f"{total} dialogs learned")

            reset_all()

    elapsed = (datetime.now() - start).total_seconds()
    print(f"✅ done ({elapsed:.2f}s)")
    print(f"dialogs: {total}")
    print(f"pairs: {pairs}")
    if elapsed > 0:
        print(f"speed: {pairs/elapsed:.1f} pairs/sec")


# ============================================================
# CLI
# ============================================================


def main():
    print("Ultra Fast Trace AI")
    print("Commands:")
    print('  learn "Q" "A"')
    print("  learncsv <file.csv> [count]")
    print('  stimulate "Q"')
    print("  auto")
    print("  reset")
    print("  exit")

    while True:
        try:
            cmd = input(">>> ").strip()

            if cmd == "exit":
                break

            elif cmd.startswith("learn "):
                m = re.findall(r'"([^"]*)"', cmd)
                if len(m) >= 2:
                    learn(m[0], m[1])
                    print("learned")

            elif cmd.startswith("learncsv "):
                parts = cmd.split()
                file_path = parts[1]
                count = int(parts[2]) if len(parts) == 3 else None
                learn_from_csv(file_path, count)

            elif cmd.startswith("stimulate "):
                m = re.findall(r'"([^"]*)"', cmd)
                if m:
                    reset_all()
                    for ch in m[0]:
                        if ch in input_layers:
                            input_layers[ch].receive(INITIAL_SIGNAL_STRENGTH)
                            step_all()
                    print(generate())

            elif cmd == "auto":
                print("Auto mode")
                learn_from_csv("train.csv", max_dialogs=10)
                reset_all()
                q = "Hello"
                for ch in q:
                    input_layers[ch].receive(INITIAL_SIGNAL_STRENGTH)
                    step_all()
                print(generate())

            elif cmd == "reset":
                reset_all()
                print("reset")

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
