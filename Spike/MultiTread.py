import os
import csv
import re
import sys
import pickle
import gzip
import bz2
import lzma
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================================
# CONFIG
# ==========================================================

VERSION = "4.1-MP"

DECAY_RATE = 0.78
STATE_DECAY_RATE = 0.47
SIMILARITY_K = 67.0
STATE_INFLUENCE = 0.46
INITIAL_SIGNAL_STRENGTH = 5.0
MIN_SIGNAL_THRESHOLD = 0.0
END_TOKEN = "<END>"

LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()[]{}<>@#$%^&*-_=+|\\/`~"

NUM_PROCESSES = multiprocessing.cpu_count()

# ==========================================================
# CONNECTOR
# ==========================================================


class Connector:
    __slots__ = ("output_id", "history", "_sorted")

    def __init__(self, output_id):
        self.output_id = output_id
        self.history = []
        self._sorted = True

    def learn(self, signal, state_layers):
        snapshot = {
            k: v.signal
            for k, v in state_layers.items()
            if v.signal > MIN_SIGNAL_THRESHOLD
        }
        self.history.append((signal, snapshot))
        self._sorted = False

    def merge(self, other):
        if not other.history:
            return
        self.history.extend(other.history)
        self._sorted = False

    def _ensure_sorted(self):
        if not self._sorted:
            self.history.sort(key=lambda x: x[0])
            self._sorted = True

    def transmit(self, signal, output_layers, state_layers):
        if signal < MIN_SIGNAL_THRESHOLD or not self.history:
            return

        self._ensure_sorted()

        import bisect

        signals = [h[0] for h in self.history]
        pos = bisect.bisect_left(signals, signal)

        candidates = []
        if pos > 0:
            candidates.append(self.history[pos - 1])
        if pos < len(self.history):
            candidates.append(self.history[pos])

        if not candidates:
            return

        stored_signal, stored_state = min(candidates, key=lambda x: abs(x[0] - signal))

        diff = abs(stored_signal - signal)
        weight1 = max(0.0, 1.0 - diff / SIMILARITY_K)

        total = 0
        count = 0
        for k, v in stored_state.items():
            if k in state_layers:
                total += abs(v - state_layers[k].signal)
                count += 1

        weight2 = 1.0
        if count:
            weight2 = max(0.0, 1.0 - (total / count) / SIMILARITY_K)

        alpha = 1.0 - STATE_INFLUENCE
        beta = STATE_INFLUENCE

        final_weight = (weight1**alpha) * (weight2**beta)
        output_layers[self.output_id].signal += signal * final_weight


# ==========================================================
# LAYER
# ==========================================================


class Layer:
    __slots__ = ("char", "signal")

    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def decay(self, rate):
        self.signal *= rate


# ==========================================================
# MODEL
# ==========================================================


class AlphaIntelligence:

    def __init__(self):
        chars = list(LETTERS) + [END_TOKEN]

        self.input_layers = {c: Layer(c) for c in chars}
        self.output_layers = {c: Layer(c) for c in chars}
        self.state_layers = {c: Layer(c) for c in chars}

        self.connectors = {c: {o: Connector(o) for o in chars} for c in chars}

    # ------------------------------------------------------

    def reset(self):
        for d in (self.input_layers, self.output_layers, self.state_layers):
            for l in d.values():
                l.signal = 0.0

    # ------------------------------------------------------

    def step(self):
        for l in self.input_layers.values():
            l.decay(DECAY_RATE)
        for l in self.output_layers.values():
            l.decay(DECAY_RATE)
        for l in self.state_layers.values():
            l.decay(STATE_DECAY_RATE)

    # ------------------------------------------------------

    def learn(self, q, a):
        for ch in q:
            if ch not in self.input_layers:
                continue
            self.input_layers[ch].signal += INITIAL_SIGNAL_STRENGTH
            self.step()

        tokens = list(a) + [END_TOKEN]

        for out in tokens:
            if out not in self.output_layers:
                continue

            for ch in q:
                if ch not in self.connectors:
                    continue
                self.connectors[ch][out].learn(
                    self.input_layers[ch].signal,
                    self.state_layers,
                )

            self.state_layers[out].signal += INITIAL_SIGNAL_STRENGTH
            self.step()

    # ------------------------------------------------------

    def merge(self, other):
        for ch in self.connectors:
            for out in self.connectors[ch]:
                self.connectors[ch][out].merge(other.connectors[ch][out])

    # ------------------------------------------------------

    def generate(self, q, max_len=512):
        self.reset()

        for ch in q:
            if ch in self.input_layers:
                self.input_layers[ch].signal += INITIAL_SIGNAL_STRENGTH
                self.step()

        result = []

        for _ in range(max_len):

            for l in self.output_layers.values():
                l.signal = 0.0

            for ch, layer in self.input_layers.items():
                if layer.signal < MIN_SIGNAL_THRESHOLD:
                    continue
                for conn in self.connectors[ch].values():
                    conn.transmit(
                        layer.signal,
                        self.output_layers,
                        self.state_layers,
                    )

            best = max(self.output_layers.values(), key=lambda x: x.signal)

            if best.signal < MIN_SIGNAL_THRESHOLD:
                break

            if best.char == END_TOKEN:
                break

            result.append(best.char)

            self.state_layers[best.char].signal += INITIAL_SIGNAL_STRENGTH
            self.step()

        return "".join(result)

    # ------------------------------------------------------

    def save(self, path, compression="lzma"):
        data = self.__dict__

        if compression == "lzma":
            if not path.endswith(".xz"):
                path += ".xz"
            with lzma.open(path, "wb", preset=9) as f:
                pickle.dump(data, f)

        elif compression == "gzip":
            if not path.endswith(".gz"):
                path += ".gz"
            with gzip.open(path, "wb", compresslevel=9) as f:
                pickle.dump(data, f)

        elif compression == "bz2":
            if not path.endswith(".bz2"):
                path += ".bz2"
            with bz2.open(path, "wb", compresslevel=9) as f:
                pickle.dump(data, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f)

        print("Saved:", path)

    # ------------------------------------------------------

    @classmethod
    def load(cls, path):
        if path.endswith(".xz"):
            f = lzma.open(path, "rb")
        elif path.endswith(".gz"):
            f = gzip.open(path, "rb")
        elif path.endswith(".bz2"):
            f = bz2.open(path, "rb")
        else:
            f = open(path, "rb")

        data = pickle.load(f)
        f.close()

        model = cls()
        model.__dict__.update(data)
        return model


# ==========================================================
# MULTIPROCESS LEARNING
# ==========================================================


def learn_chunk(dialogs):
    model = AlphaIntelligence()
    pairs = 0

    for dialog in dialogs:
        for i in range(len(dialog) - 1):
            model.learn(dialog[i], dialog[i + 1])
            pairs += 1

    return model, pairs


def learn_from_csv(model, path, max_dialogs=None, processes=NUM_PROCESSES):

    dialogs = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if max_dialogs and len(dialogs) >= max_dialogs:
                break

            raw = row.get("dialog", "")
            matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", raw)

            dialog = []
            for m in matches:
                t = m[0] if m[0] else m[1]
                if t.strip():
                    dialog.append(t.strip())

            if len(dialog) >= 2:
                dialogs.append(dialog)

    if not dialogs:
        print("No dialogs found.")
        return

    chunk_size = max(1, len(dialogs) // processes)
    chunks = [dialogs[i : i + chunk_size] for i in range(0, len(dialogs), chunk_size)]

    start = datetime.now()

    total_pairs = 0

    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [executor.submit(learn_chunk, c) for c in chunks]

        for f in as_completed(futures):
            worker_model, pairs = f.result()
            model.merge(worker_model)
            total_pairs += pairs

    elapsed = (datetime.now() - start).total_seconds()

    print(f"Learned {total_pairs} pairs in {elapsed:.2f}s")
    print(f"{total_pairs/elapsed:.1f} pairs/sec")


# ==========================================================
# CLI
# ==========================================================

HELP = """
Commands:
  learn "Q" "A"
  learncsv <file> [count] [processes]
  stimulate "Q"
  save <file> [gzip|bz2|lzma|none]
  load <file>
  exit
"""


def main():
    global NUM_PROCESSES

    model = AlphaIntelligence()

    print("AlphaIntelligence v", VERSION)
    print("CPU:", NUM_PROCESSES)
    print(HELP)

    while True:
        try:
            cmd = input(">>> ").strip()

            if not cmd:
                continue

            if cmd in ("exit", "quit"):
                break

            elif cmd.startswith("learncsv "):
                parts = cmd.split()
                file_path = parts[1]
                count = int(parts[2]) if len(parts) >= 3 else None
                procs = int(parts[3]) if len(parts) >= 4 else NUM_PROCESSES

                learn_from_csv(model, file_path, count, procs)

            elif cmd.startswith("learn "):
                matches = re.findall(r'"([^"]*)"', cmd)
                if len(matches) >= 2:
                    model.learn(matches[0], matches[1])
                    print("Learned.")
                else:
                    print('Usage: learn "Q" "A"')

            elif cmd.startswith("stimulate "):
                matches = re.findall(r'"([^"]*)"', cmd)
                if matches:
                    print("Response:", model.generate(matches[0]))

            elif cmd.startswith("save "):
                parts = cmd.split()
                path = parts[1]
                comp = parts[2] if len(parts) >= 3 else "lzma"
                model.save(path, comp)

            elif cmd.startswith("load "):
                parts = cmd.split()
                model = AlphaIntelligence.load(parts[1])
                print("Loaded.")

            else:
                print("Unknown command.")

        except KeyboardInterrupt:
            break

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
