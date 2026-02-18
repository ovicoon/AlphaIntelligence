import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import csv
import re
import sys
import bisect
import pickle
import gzip
import bz2
import lzma
from datetime import datetime
from multiprocessing import Pool, cpu_count  # 🎯 멀티프로세스!
import copy

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "3.9-MP"  # Multi-Process

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_CAP = ALPHABET.upper()
DIGITS = "0123456789"
SPECIAL_CHARS = " .,!?;:'\"()[]{}<>@#$%^&*-_=+|\\/`~"
LETTERS = ALPHABET + ALPHABET_CAP + DIGITS + SPECIAL_CHARS
END_TOKEN = "<END>"

ALLOW_UNICODE = True

DECAY_RATE = 0.781
STATE_DECAY_RATE = 0.479
SIMILARITY_K = 67.433
MAX_HISTORY = float("inf")
MIN_SIGNAL_THRESHOLD = 0.0
INITIAL_SIGNAL_STRENGTH = 5.0

STATE_INFLUENCE = 0.468

NUM_PROCESSES = cpu_count()  # CPU 코어 수


# ============================================================================
# CONNECTOR
# ============================================================================


class Connector:
    __slots__ = ("output_layer_id", "history_signals", "history_states")

    def __init__(self, output_layer_id):
        self.output_layer_id = output_layer_id
        self.history_signals = []
        self.history_states = []

    def transmit(
        self,
        signal,
        output_layers_list,
        state_layers_dict,
        output_idx_map,
        state_idx_map,
    ):
        if not self.history_signals or signal < MIN_SIGNAL_THRESHOLD:
            return

        k = SIMILARITY_K
        pos = bisect.bisect_left(self.history_signals, signal)

        candidates = []
        if pos > 0:
            candidates.append((pos - 1, self.history_signals[pos - 1]))
        if pos < len(self.history_signals):
            candidates.append((pos, self.history_signals[pos]))

        if not candidates:
            return

        min_idx, min_signal = min(candidates, key=lambda x: abs(x[1] - signal))
        min_distance = abs(min_signal - signal)

        weight1 = max(0.0, 1.0 - min_distance / k)

        state_snapshot = self.history_states[min_idx]

        total_diff = 0.0
        count = 0

        for char, snapshot_signal in state_snapshot.items():
            current_layer = state_layers_dict.get(char)
            if current_layer:
                total_diff += abs(snapshot_signal - current_layer.signal)
                count += 1

        if count > 0:
            avg_diff = total_diff / count
            weight2 = max(0.0, 1.0 - avg_diff / k)
        else:
            weight2 = 1.0

        alpha = 1.0 - STATE_INFLUENCE
        beta = STATE_INFLUENCE

        final_weight = (weight1**alpha) * (weight2**beta)
        weighted_signal = signal * final_weight

        output_idx = output_idx_map.get(self.output_layer_id)
        if output_idx is not None:
            output_layers_list[output_idx].receive(weighted_signal)

    def learn(self, signal, state_layers_dict):
        state_snapshot = {
            char: layer.signal
            for char, layer in state_layers_dict.items()
            if layer.signal > MIN_SIGNAL_THRESHOLD
        }

        pos = bisect.bisect_left(self.history_signals, signal)
        self.history_signals.insert(pos, signal)
        self.history_states.insert(pos, state_snapshot)

        if len(self.history_signals) > MAX_HISTORY:
            self.history_signals.pop(0)
            self.history_states.pop(0)

    def merge(self, other):
        for signal, state in zip(other.history_signals, other.history_states):
            pos = bisect.bisect_left(self.history_signals, signal)
            self.history_signals.insert(pos, signal)
            self.history_states.insert(pos, state)

            if len(self.history_signals) > MAX_HISTORY:
                self.history_signals.pop(0)
                self.history_states.pop(0)


# ============================================================================
# LAYERS
# ============================================================================


class InputLayer:
    __slots__ = ("char", "signal", "connections")

    def __init__(self, char, output_layer_ids):
        self.char = char
        self.signal = 0.0
        self.connections = [Connector(oid) for oid in output_layer_ids]

    def step(self):
        self.signal *= DECAY_RATE

    def fire(
        self, output_layers_list, state_layers_dict, output_idx_map, state_idx_map
    ):
        if self.signal < MIN_SIGNAL_THRESHOLD:
            return

        for conn in self.connections:
            conn.transmit(
                self.signal,
                output_layers_list,
                state_layers_dict,
                output_idx_map,
                state_idx_map,
            )

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


# ============================================================================
# MODEL
# ============================================================================


class AlphaIntelligence:
    def __init__(self):
        initial_chars = list(LETTERS) + [END_TOKEN]

        self.output_layers = {char: OutputLayer(char) for char in initial_chars}
        self.input_layers = {
            char: InputLayer(char, initial_chars) for char in initial_chars
        }
        self.state_layers = {char: StateLayer(char) for char in initial_chars}

        self._rebuild_caches()

    def _rebuild_caches(self):
        self.output_layers_list = list(self.output_layers.values())
        self.input_layers_list = list(self.input_layers.values())

        self.output_idx_map = {
            char: i for i, char in enumerate(self.output_layers.keys())
        }
        self.state_idx_map = {
            char: i for i, char in enumerate(self.state_layers.keys())
        }

    def add_character(self, char):
        if char in self.input_layers:
            return False

        self.output_layers[char] = OutputLayer(char)
        self.state_layers[char] = StateLayer(char)

        output_ids = list(self.output_layers.keys())
        self.input_layers[char] = InputLayer(char, output_ids)

        for input_layer in self.input_layers.values():
            if len(input_layer.connections) < len(output_ids):
                input_layer.connections.append(Connector(char))

        self._rebuild_caches()
        return True

    def add_characters_from_text(self, text):
        added = []
        for char in text:
            if char not in self.input_layers and ALLOW_UNICODE:
                if self.add_character(char):
                    added.append(char)
        return added

    def step_all(self):
        for layer in self.input_layers_list:
            if layer.signal > MIN_SIGNAL_THRESHOLD:
                layer.step()

        for layer in self.output_layers_list:
            if layer.signal > MIN_SIGNAL_THRESHOLD:
                layer.step()

        for layer in self.state_layers.values():
            if layer.signal > MIN_SIGNAL_THRESHOLD:
                layer.step()

    def reset_all(self):
        for layer in self.input_layers_list:
            layer.signal = 0.0
        for layer in self.output_layers_list:
            layer.signal = 0.0
        for layer in self.state_layers.values():
            layer.signal = 0.0

    def stimulate(self, char, strength=INITIAL_SIGNAL_STRENGTH):
        layer = self.input_layers.get(char)
        if layer:
            layer.receive(strength)

    def find_strongest_output(self):
        max_signal = -1.0
        strongest_idx = -1

        for i, layer in enumerate(self.output_layers_list):
            if layer.signal > max_signal:
                max_signal = layer.signal
                strongest_idx = i

        if strongest_idx >= 0:
            return self.output_layers_list[strongest_idx].char, max_signal
        return None, max_signal

    def learn(self, question, answer):
        self.add_characters_from_text(question)
        self.add_characters_from_text(answer)
        self.add_characters_from_text(END_TOKEN)

        answer_tokens = list(answer) + [END_TOKEN]

        for char in question:
            layer = self.input_layers.get(char)
            if layer:
                layer.receive(INITIAL_SIGNAL_STRENGTH)
                self.step_all()

        for achar in answer_tokens:
            for char in question:
                layer = self.input_layers.get(char)
                if not layer:
                    continue

                for conn in layer.connections:
                    if conn.output_layer_id == achar:
                        conn.learn(layer.signal, self.state_layers)
                        break

            state_layer = self.state_layers.get(achar)
            if state_layer:
                state_layer.receive(INITIAL_SIGNAL_STRENGTH)

            self.step_all()

    def copy_structure(self):
        new_model = AlphaIntelligence()

        for char in self.input_layers.keys():
            if char not in new_model.input_layers:
                new_model.add_character(char)

        return new_model

    def merge_from(self, other_model):
        for char in other_model.input_layers.keys():
            if char not in self.input_layers:
                self.add_character(char)

        for char, other_input_layer in other_model.input_layers.items():
            if char not in self.input_layers:
                continue

            self_input_layer = self.input_layers[char]

            for i, other_conn in enumerate(other_input_layer.connections):
                if i >= len(self_input_layer.connections):
                    break

                self_conn = self_input_layer.connections[i]
                if self_conn.output_layer_id == other_conn.output_layer_id:
                    self_conn.merge(other_conn)

    def generate(self, question, max_length=4096, verbose=False):
        self.add_characters_from_text(question)

        for char in question:
            self.stimulate(char)
            self.step_all()

        output_text = []
        repeat_count = 0
        last_char = None

        if verbose:
            print("  Response: ", end="", flush=True)

        for _ in range(max_length):
            for layer in self.output_layers_list:
                layer.signal = 0.0

            for layer in self.input_layers_list:
                layer.fire(
                    self.output_layers_list,
                    self.state_layers,
                    self.output_idx_map,
                    self.state_idx_map,
                )

            strongest_char, max_signal = self.find_strongest_output()

            if max_signal < MIN_SIGNAL_THRESHOLD:
                break

            if strongest_char == END_TOKEN:
                break

            if strongest_char == last_char:
                repeat_count += 1
                if repeat_count >= 3:
                    if verbose:
                        print("  [repeat]", flush=True)
                    break
            else:
                repeat_count = 0
                last_char = strongest_char

            output_text.append(strongest_char)

            if verbose:
                print(strongest_char, end="", flush=True)

            state_layer = self.state_layers.get(strongest_char)
            if state_layer:
                state_layer.receive(INITIAL_SIGNAL_STRENGTH)

            self.step_all()

        if verbose:
            print()

        return "".join(output_text)

    def save(self, filepath, compression="lzma"):
        save_data = {
            "version": VERSION,
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

        if compression == "lzma":
            if not filepath.endswith(".xz"):
                filepath += ".xz"
            with lzma.open(filepath, "wb", preset=9) as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif compression == "bz2":
            if not filepath.endswith(".bz2"):
                filepath += ".bz2"
            with bz2.open(filepath, "wb", compresslevel=9) as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif compression == "gzip":
            if not filepath.endswith(".gz"):
                filepath += ".gz"
            with gzip.open(filepath, "wb", compresslevel=9) as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(filepath, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  Saved: {filepath} ({size_mb:.2f} MB, {compression or 'none'})")

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            for ext in [".xz", ".gz", ".bz2"]:
                if os.path.exists(filepath + ext):
                    filepath += ext
                    break
            else:
                raise FileNotFoundError(f"File not found: {filepath}")

        if filepath.endswith(".xz") or filepath.endswith(".lzma"):
            with lzma.open(filepath, "rb") as f:
                save_data = pickle.load(f)
            comp_type = "lzma"

        elif filepath.endswith(".bz2"):
            with bz2.open(filepath, "rb") as f:
                save_data = pickle.load(f)
            comp_type = "bz2"

        elif filepath.endswith(".gz"):
            with gzip.open(filepath, "rb") as f:
                save_data = pickle.load(f)
            comp_type = "gzip"

        else:
            with open(filepath, "rb") as f:
                save_data = pickle.load(f)
            comp_type = "none"

        model = cls.__new__(cls)
        model.output_layers = save_data["output_layers"]
        model.input_layers = save_data["input_layers"]
        model.state_layers = save_data["state_layers"]
        model._rebuild_caches()

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  Loaded: {filepath} ({size_mb:.2f} MB, {comp_type})")
        print(f"  Chars:  {len(model.input_layers)}")

        return model

    def get_stats(self):
        total_history = sum(
            len(conn.history_signals)
            for layer in self.input_layers_list
            for conn in layer.connections
        )

        return {
            "chars": len(self.input_layers),
            "total_history": total_history,
            "avg_history": (
                total_history / len(self.input_layers) if self.input_layers else 0
            ),
        }


# ============================================================================
# MULTI-PROCESSING WORKER
# ============================================================================


def _worker_learn_dialogs(args):
    """🎯 워커 프로세스 (pickle 가능하도록 최상위 함수)"""
    base_model_data, dialogs_chunk = args

    # 모델 복원
    worker_model = pickle.loads(base_model_data)

    pairs_learned = 0

    for dialog in dialogs_chunk:
        worker_model.reset_all()

        for i in range(len(dialog) - 1):
            q = dialog[i]
            a = dialog[i + 1]

            try:
                worker_model.learn(q, a)
                pairs_learned += 1
            except Exception:
                pass

        worker_model.reset_all()

    # 모델 직렬화하여 반환
    return pickle.dumps(worker_model), pairs_learned


def parse_dialog(raw_text):
    pattern = r"'([^']+)'|\"([^\"]+)\""
    matches = re.findall(pattern, raw_text)

    if not matches:
        return []

    dialog = []
    for match in matches:
        text = match[0] if match[0] else match[1]
        text = text.strip()
        if text:
            dialog.append(text)

    return dialog


def print_progress_bar(current, total, suffix="", length=40):
    if total == 0:
        return

    percent = 100 * (current / float(total))
    filled = int(length * current // total)
    bar = "#" * filled + "-" * (length - filled)

    sys.stdout.write(f"\r  [{bar}] {percent:5.1f}%  {suffix}")
    sys.stdout.flush()

    if current == total:
        print()


def learn_from_csv(model, file_path, max_dialogs=None, num_processes=NUM_PROCESSES):
    """🎯 멀티프로세스 학습"""
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.normpath(os.path.join(base_dir, file_path))

    if os.path.isdir(file_path):
        print(f"  Error: directory: {file_path}")
        return

    if not os.path.isfile(file_path):
        print(f"  Error: not found: {file_path}")
        return

    limit_text = f"{max_dialogs}" if max_dialogs else "all"
    print(
        f"  Learning: {os.path.basename(file_path)} ({limit_text} dialogs, {num_processes} processes)..."
    )

    start_time = datetime.now()

    # CSV 읽기
    dialogs = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if max_dialogs and len(dialogs) >= max_dialogs:
                break

            raw = row.get("dialog", "")
            if not raw:
                continue

            dialog = parse_dialog(raw)
            if len(dialog) >= 2:
                dialogs.append(dialog)

    if not dialogs:
        print("  No dialogs.")
        return

    total_dialogs = len(dialogs)

    # 청크 분할
    chunk_size = max(1, total_dialogs // num_processes)
    chunks = [dialogs[i : i + chunk_size] for i in range(0, total_dialogs, chunk_size)]

    # 베이스 모델 직렬화 (한 번만)
    base_model_data = pickle.dumps(model.copy_structure())

    # 멀티프로세스 학습
    total_pairs = 0

    with Pool(processes=num_processes) as pool:
        args_list = [(base_model_data, chunk) for chunk in chunks]

        results = []
        for i, result in enumerate(
            pool.imap_unordered(_worker_learn_dialogs, args_list)
        ):
            worker_model_data, pairs = result

            # 모델 역직렬화 및 병합
            worker_model = pickle.loads(worker_model_data)
            model.merge_from(worker_model)

            total_pairs += pairs

            suffix = f"{i+1}/{len(chunks)} chunks, {total_pairs} pairs"
            print_progress_bar(i + 1, len(chunks), suffix=suffix)

    elapsed = (datetime.now() - start_time).total_seconds()
    stats = model.get_stats()

    print(
        f"  Done: {total_dialogs} dialogs, {total_pairs} pairs, {stats['chars']} chars ({elapsed:.1f}s)"
    )
    print(f"  Speed: {total_pairs/elapsed:.1f} pairs/sec")


# ============================================================================
# CLI
# ============================================================================

DIVIDER = "-" * 50

HELP_TEXT = """
  Commands:
    learn "Q" "A"                     learn Q/A
    learncsv <file> [count] [procs]   learn from CSV (multi-process)
    stimulate "Q"                     generate
    save <file> [compression]         save (gzip|bz2|lzma|none)
    load <file>                       load
    reset                             reset
    stats                             stats
    processes [N]                     set process count
    help                              help
    exit                              quit
"""


def print_header():
    print()
    print(f"  AlphaIntelligence  v{VERSION}")
    print(f"  CPU Cores: {NUM_PROCESSES}")
    print(DIVIDER)


def main():
    global NUM_PROCESSES

    print_header()
    print(HELP_TEXT)

    model = AlphaIntelligence()
    running = True

    while running:
        try:
            cmd = input(">>> ").strip()

            if not cmd:
                continue

            if cmd in ("exit", "quit"):
                print("  Goodbye.")
                running = False

            elif cmd in ("help", "?"):
                print(HELP_TEXT)

            elif cmd.startswith("processes "):
                parts = cmd.split()
                if len(parts) == 2:
                    try:
                        NUM_PROCESSES = int(parts[1])
                        print(f"  Processes: {NUM_PROCESSES}")
                    except ValueError:
                        print("  Error: integer")
                else:
                    print(f"  Current: {NUM_PROCESSES}")

            elif cmd.startswith("stimulate "):
                rest = cmd[10:]
                matches = re.findall(r'"([^"]*)"', rest)
                question = matches[0] if matches else rest.strip()

                model.reset_all()

                for char in question:
                    model.stimulate(char)
                    model.step_all()

                model.generate(question, verbose=True)

            elif cmd.startswith("learn "):
                rest = cmd[6:]
                matches = re.findall(r'"([^"]*)"', rest)

                if len(matches) >= 2:
                    model.learn(matches[0], matches[1])
                    print("  Learned.")
                else:
                    print('  Usage: learn "Q" "A"')

            elif cmd.startswith("learncsv "):
                parts = cmd.split()
                if len(parts) >= 2:
                    file_path = parts[1]
                    max_dialogs = None
                    procs = NUM_PROCESSES

                    if len(parts) >= 3:
                        try:
                            max_dialogs = int(parts[2])
                        except ValueError:
                            print("  Error: count")
                            continue

                    if len(parts) >= 4:
                        try:
                            procs = int(parts[3])
                        except ValueError:
                            print("  Error: processes")
                            continue

                    learn_from_csv(model, file_path, max_dialogs, procs)

            elif cmd.startswith("save "):
                parts = cmd.split()
                filepath = parts[1] if len(parts) >= 2 else None
                compression = parts[2] if len(parts) >= 3 else "lzma"

                if not filepath:
                    print("  Usage: save <file> [gzip|bz2|lzma|none]")
                    continue

                if compression not in ("gzip", "bz2", "lzma", "none"):
                    print("  Error: compression")
                    continue

                model.save(filepath, compression)

            elif cmd.startswith("load "):
                parts = cmd.split()
                if len(parts) == 2:
                    model = AlphaIntelligence.load(parts[1])

            elif cmd == "reset":
                model.reset_all()
                print("  Reset.")

            elif cmd == "stats":
                stats = model.get_stats()
                print(f"\n  Stats")
                print(f"  {DIVIDER}")
                print(f"  Chars           {stats['chars']}")
                print(f"  History         {stats['total_history']}")
                print(f"  Avg             {stats['avg_history']:.1f}")
                print(f"  Processes       {NUM_PROCESSES}")
                print()

            else:
                print("  Unknown. Type 'help'.")

        except KeyboardInterrupt:
            print("\n  Goodbye.")
            running = False

        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
