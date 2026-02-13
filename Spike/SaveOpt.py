import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import re, sys, bisect, pickle, csv
import numpy as np
from datetime import datetime

# ====================================================================
# CONFIG
# ====================================================================

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_CAP = ALPHABET.upper()
DIGITS = "0123456789"
SPECIAL_CHARS = " .,!?;:'\"()[]{}<>@#$%^&*-_=+|\\/`~"
LETTERS = ALPHABET + ALPHABET_CAP + DIGITS + SPECIAL_CHARS
END_TOKEN = "<END>"

DECAY_RATE = 0.78
STATE_DECAY_RATE = 0.48
SIMILARITY_K = 20
MIN_SIGNAL_THRESHOLD = 0.0
INITIAL_SIGNAL_STRENGTH = 5.0
STATE_INFLUENCE = 0.5
ALLOW_UNICODE = True

# ====================================================================
# CONNECTOR
# ====================================================================


class Connector:
    __slots__ = ("output_layer_id", "history_signals", "history_states")

    def __init__(self, output_layer_id, state_dim):
        self.output_layer_id = output_layer_id
        self.history_signals = np.zeros((0,), dtype=np.float32)
        self.history_states = np.zeros((0, state_dim), dtype=np.float32)

    def _ensure_dim(self, new_dim):
        """state_dim 변경 시 기존 히스토리 패딩"""
        if self.history_states.shape[1] < new_dim:
            diff = new_dim - self.history_states.shape[1]
            self.history_states = np.hstack(
                [
                    self.history_states,
                    np.zeros((self.history_states.shape[0], diff), dtype=np.float32),
                ]
            )

    def transmit(self, signal, output_layers_list, state_vector, output_idx_map):
        # 히스토리 배열 차원 확인
        self._ensure_dim(len(state_vector))
        if len(self.history_signals) == 0 or signal < MIN_SIGNAL_THRESHOLD:
            return

        pos = np.searchsorted(self.history_signals, signal)
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(self.history_signals):
            candidates.append(pos)

        min_idx = min(candidates, key=lambda x: abs(self.history_signals[x] - signal))
        weight1 = max(
            0.0, 1.0 - abs(self.history_signals[min_idx] - signal) / SIMILARITY_K
        )

        # vectorized state diff
        hist_vec = self.history_states[min_idx]
        if len(hist_vec) < len(state_vector):
            hist_vec = np.pad(
                hist_vec, (0, len(state_vector) - len(hist_vec)), "constant"
            )
        avg_diff = np.mean(np.abs(hist_vec - state_vector))
        weight2 = max(0.0, 1.0 - avg_diff / SIMILARITY_K)

        alpha = 1.0 - STATE_INFLUENCE
        beta = STATE_INFLUENCE
        final_weight = (weight1**alpha) * (weight2**beta)
        weighted_signal = signal * final_weight

        out_idx = output_idx_map.get(self.output_layer_id)
        if out_idx is not None:
            output_layers_list[out_idx].receive(weighted_signal)

    def learn(self, signal, state_vector):
        self._ensure_dim(len(state_vector))
        pos = np.searchsorted(self.history_signals, signal)
        # state_vector가 히스토리보다 길면 패딩
        vec = state_vector
        if len(vec) < self.history_states.shape[1]:
            vec = np.pad(vec, (0, self.history_states.shape[1] - len(vec)), "constant")
        self.history_signals = np.insert(self.history_signals, pos, signal)
        self.history_states = np.insert(
            self.history_states, pos, vec[np.newaxis, :], axis=0
        )


# ====================================================================
# LAYER CLASSES
# ====================================================================


class InputLayer:
    __slots__ = ("char", "signal", "connections")

    def __init__(self, char, output_layer_ids, state_dim):
        self.char = char
        self.signal = 0.0
        self.connections = [Connector(oid, state_dim) for oid in output_layer_ids]

    def step(self):
        self.signal *= DECAY_RATE

    def fire(self, output_layers_list, state_vector, output_idx_map):
        if self.signal < MIN_SIGNAL_THRESHOLD:
            return
        for conn in self.connections:
            conn.transmit(self.signal, output_layers_list, state_vector, output_idx_map)

    def receive(self, amount):
        self.signal += amount


class OutputLayer:
    __slots__ = ("char", "signal")

    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def step(self):
        self.signal *= DECAY_RATE

    def receive(self, amount):
        self.signal += amount


class StateLayer:
    __slots__ = ("char", "signal")

    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def step(self):
        self.signal *= STATE_DECAY_RATE

    def receive(self, amount):
        self.signal += amount


# ====================================================================
# MODEL
# ====================================================================


class AlphaIntelligence:
    def __init__(self):
        chars = list(LETTERS) + [END_TOKEN]
        self.state_dim = len(chars)

        self.output_layers = {c: OutputLayer(c) for c in chars}
        self.input_layers = {c: InputLayer(c, chars, self.state_dim) for c in chars}
        self.state_layers = {c: StateLayer(c) for c in chars}

        self._rebuild_cache()

    def _rebuild_cache(self):
        self.output_layers_list = list(self.output_layers.values())
        self.input_layers_list = list(self.input_layers.values())
        self.output_idx_map = {c: i for i, c in enumerate(self.output_layers.keys())}
        self.state_idx_map = {c: i for i, c in enumerate(self.state_layers.keys())}

    def step_all(self):
        for l in self.input_layers_list:
            l.step()
        for l in self.output_layers_list:
            l.step()
        for l in self.state_layers.values():
            l.step()

    def reset_all(self):
        for l in self.input_layers_list:
            l.signal = 0.0
        for l in self.output_layers_list:
            l.signal = 0.0
        for l in self.state_layers.values():
            l.signal = 0.0

    def get_state_vector(self):
        return np.array(
            [l.signal for l in self.state_layers.values()], dtype=np.float32
        )

    def stimulate(self, char, strength=INITIAL_SIGNAL_STRENGTH):
        l = self.input_layers.get(char)
        if l:
            l.receive(strength)

    def find_strongest_output(self):
        max_sig = -1.0
        idx = -1
        for i, l in enumerate(self.output_layers_list):
            if l.signal > max_sig:
                max_sig, idx = l.signal, i
        return (
            (self.output_layers_list[idx].char, max_sig)
            if idx >= 0
            else (None, max_sig)
        )

    def learn(self, question, answer):
        self.add_characters_from_text(question + answer)
        question_chars = list(question)
        answer_chars = list(answer) + [END_TOKEN]

        for c in question_chars:
            self.input_layers[c].receive(INITIAL_SIGNAL_STRENGTH)
        self.step_all()

        for ach in answer_chars:
            state_vec = self.get_state_vector()
            for c in question_chars:
                for conn in self.input_layers[c].connections:
                    if conn.output_layer_id == ach:
                        conn.learn(self.input_layers[c].signal, state_vec)
                        break
            self.state_layers[ach].receive(INITIAL_SIGNAL_STRENGTH)
            self.step_all()

    def add_characters_from_text(self, text):
        added = False
        for c in text:
            if c not in self.input_layers and ALLOW_UNICODE:
                self._add_char(c)
                added = True
        return added

    def _add_char(self, c):
        self.output_layers[c] = OutputLayer(c)
        self.state_layers[c] = StateLayer(c)
        self.input_layers[c] = InputLayer(
            c, list(self.output_layers.keys()), self.state_dim
        )
        for il in self.input_layers.values():
            if len(il.connections) < len(self.output_layers):
                il.connections.append(Connector(c, self.state_dim))
        self.state_dim = len(self.state_layers)
        self._rebuild_cache()

    def generate(self, question, max_length=4096, verbose=False):
        for c in question:
            self.stimulate(c)
            self.step_all()
        out_chars = []
        repeat, last = 0, None

        for _ in range(max_length):
            for l in self.output_layers_list:
                l.signal = 0.0
            state_vec = self.get_state_vector()
            for l in self.input_layers_list:
                l.fire(self.output_layers_list, state_vec, self.output_idx_map)
            c, s = self.find_strongest_output()
            if s < MIN_SIGNAL_THRESHOLD or c == END_TOKEN:
                break
            if c == last:
                repeat += 1
            else:
                repeat, last = 0, c
            if repeat >= 3:
                break
            out_chars.append(c)
            self.state_layers[c].receive(INITIAL_SIGNAL_STRENGTH)
            self.step_all()
        return "".join(out_chars)

    # ------------------- SAVE/LOAD -------------------

    def save(self, filepath):
        """벡터화된 NumPy 배열 그대로 저장"""
        data = {
            "config": {
                "DECAY_RATE": DECAY_RATE,
                "STATE_DECAY_RATE": STATE_DECAY_RATE,
                "SIMILARITY_K": SIMILARITY_K,
                "STATE_INFLUENCE": STATE_INFLUENCE,
            },
            "output_layers": self.output_layers,
            "input_layers": self.input_layers,
            "state_layers": self.state_layers,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print(f"✅ 저장: {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        model = cls.__new__(cls)
        model.output_layers = data["output_layers"]
        model.input_layers = data["input_layers"]
        model.state_layers = data["state_layers"]
        model._rebuild_cache()
        print(f"✅ 로드: {filepath}")
        return model


# ====================================================================
# CSV LEARNING
# ====================================================================


def parse_dialog(raw_text):
    return [
        m[0] if m[0] else m[1]
        for m in re.findall(r"'([^']+)'|\"([^\"]+)\"", raw_text)
        if m[0] or m[1]
    ]


def learn_from_csv(model, file_path, max_dialogs=None):
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, file_path)
    if not os.path.isfile(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return
    start = datetime.now()
    total_pairs = 0
    learned = 0
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_dialogs and learned >= max_dialogs:
                break
            raw = row.get("dialog", "")
            if not raw:
                continue
            dialog = parse_dialog(raw)
            if len(dialog) < 2:
                continue
            model.reset_all()
            for i in range(len(dialog) - 1):
                try:
                    model.learn(dialog[i], dialog[i + 1])
                    total_pairs += 1
                except:
                    continue
            learned += 1
            model.reset_all()
    print(
        f"학습 완료: {learned} 대화, {total_pairs} 쌍, 시간: {(datetime.now()-start).total_seconds():.1f}s"
    )


# ====================================================================
# CLI
# ====================================================================


def main():
    print("AlphaIntelligence v3.5 - GPU/Vectorized")
    model = AlphaIntelligence()
    while True:
        try:
            cmd = input(">>> ").strip()
            if cmd == "exit":
                break
            elif cmd.startswith("stimulate "):
                q = re.findall(r'"([^"]*)"', cmd[10:])
                q = q[0] if q else cmd[10:]
                model.reset_all()
                for c in q:
                    model.stimulate(c)
                    model.step_all()
                print(model.generate(q, verbose=True))
            elif cmd.startswith("learn "):
                m = re.findall(r'"([^"]*)"', cmd[6:])
                if len(m) >= 2:
                    model.learn(m[0], m[1])
                    print("✅ Learned")
            elif cmd.startswith("learncsv "):
                parts = cmd.split()
                maxd = None
                if len(parts) == 3:
                    maxd = int(parts[2])
                learn_from_csv(model, parts[1], maxd)
            elif cmd == "reset":
                model.reset_all()
                print("✅ Reset")
            elif cmd == "stats":
                total_hist = sum(
                    len(c.history_signals)
                    for l in model.input_layers_list
                    for c in l.connections
                )
                print(f"Chars:{len(model.input_layers)},History:{total_hist}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
