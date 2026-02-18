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

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "3.7-compressed"

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_CAP = ALPHABET.upper()
DIGITS = "0123456789"
SPECIAL_CHARS = " .,!?;:'\"()[]{}<>@#$%^&*-_=+|\\/`~"
LETTERS = ALPHABET + ALPHABET_CAP + DIGITS + SPECIAL_CHARS
END_TOKEN = "<END>"

ALLOW_UNICODE = True

# Core parameters
DECAY_RATE = 0.781
STATE_DECAY_RATE = 0.479
SIMILARITY_K = 67.433
MAX_HISTORY = float("inf")
MIN_SIGNAL_THRESHOLD = 0.0
INITIAL_SIGNAL_STRENGTH = 5.0

STATE_INFLUENCE = 0.468

AUTO = False


# ============================================================================
# OPTIMIZED CONNECTOR (No NumPy)
# ============================================================================


class Connector:
    """Binary search optimized connector"""

    __slots__ = ("output_layer_id", "history_signals", "history_states")

    def __init__(self, output_layer_id):
        self.output_layer_id = output_layer_id
        self.history_signals = []
        self.history_states = []  # 🎯 딕셔너리 리스트로 저장

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

        # Binary search
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

        # 🎯 State comparison (Pure Python, 딕셔너리 기반)
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

        # Weighted geometric mean
        alpha = 1.0 - STATE_INFLUENCE
        beta = STATE_INFLUENCE

        final_weight = (weight1**alpha) * (weight2**beta)
        weighted_signal = signal * final_weight

        # Send signal
        output_idx = output_idx_map.get(self.output_layer_id)
        if output_idx is not None:
            output_layers_list[output_idx].receive(weighted_signal)

    def learn(self, signal, state_layers_dict):
        """🎯 딕셔너리로 저장"""
        # 활성화된 state만 저장 (압축)
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


# ============================================================================
# LAYER CLASSES
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
# MODEL MANAGER
# ============================================================================


class AlphaIntelligence:
    """Main AI model"""

    def __init__(self):
        initial_chars = list(LETTERS) + [END_TOKEN]

        self.output_layers = {char: OutputLayer(char) for char in initial_chars}
        self.input_layers = {
            char: InputLayer(char, initial_chars) for char in initial_chars
        }
        self.state_layers = {char: StateLayer(char) for char in initial_chars}

        self._rebuild_caches()

    def _rebuild_caches(self):
        """Rebuild caches"""
        self.output_layers_list = list(self.output_layers.values())
        self.input_layers_list = list(self.input_layers.values())

        self.output_idx_map = {
            char: i for i, char in enumerate(self.output_layers.keys())
        }
        self.state_idx_map = {
            char: i for i, char in enumerate(self.state_layers.keys())
        }

    def add_character(self, char):
        """Add new character"""
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
        """Add unknown characters"""
        added = []
        for char in text:
            if char not in self.input_layers and ALLOW_UNICODE:
                if self.add_character(char):
                    added.append(char)
        return added

    def step_all(self):
        """Step all layers"""
        for layer in self.input_layers_list:
            if layer.signal > MIN_SIGNAL_THRESHOLD:
                layer.step()

        for layer in self.output_layers_list:
            if layer.signal > MIN_SIGNAL_THRESHOLD:
                layer.step()

        for char, layer in self.state_layers.items():
            if layer.signal > MIN_SIGNAL_THRESHOLD:
                layer.step()

    def reset_all(self):
        """Reset all signals"""
        for layer in self.input_layers_list:
            layer.signal = 0.0
        for layer in self.output_layers_list:
            layer.signal = 0.0
        for layer in self.state_layers.values():
            layer.signal = 0.0

    def stimulate(self, char, strength=INITIAL_SIGNAL_STRENGTH):
        """Stimulate input"""
        layer = self.input_layers.get(char)
        if layer:
            layer.receive(strength)

    def find_strongest_output(self):
        """Find strongest output"""
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
        """Learn Q&A"""
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

    def generate(self, question, max_length=4096, verbose=False):
        """Generate response"""
        self.add_characters_from_text(question)

        for char in question:
            self.stimulate(char)
            self.step_all()

        output_text = []
        repeat_count = 0
        last_char = None

        if verbose:
            print("Response: ", end="", flush=True)

        for step in range(max_length):
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
                        print(f"\n[반복 감지]", flush=True)
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
        """
        compression: "gzip", "bz2", "lzma", None
        """
        save_data = {
            "version": "3.8",
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

        # 압축 방식별 저장
        if compression == "lzma":
            if not filepath.endswith(".xz"):
                filepath += ".xz"
            # LZMA 최대 압축
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
            # 압축 없음
            with open(filepath, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ 저장: {filepath} ({file_size:.2f} MB, {compression or 'none'})")

    @classmethod
    def load(cls, filepath):
        """자동 압축 감지"""
        # 파일 찾기
        if not os.path.exists(filepath):
            for ext in [".xz", ".gz", ".bz2"]:
                if os.path.exists(filepath + ext):
                    filepath += ext
                    break
            else:
                raise FileNotFoundError(f"파일 없음: {filepath}")

        # 확장자로 압축 방식 감지
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

        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ 로드: {filepath} ({file_size:.2f} MB, {comp_type})")
        print(f"   문자: {len(model.input_layers)}개")

        return model

    def get_stats(self):
        """Statistics"""
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
# CSV LEARNING
# ============================================================================


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


def print_progress_bar(current, total, prefix="", suffix="", length=40):
    if total == 0:
        return

    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = "█" * filled_length + "-" * (length - filled_length)

    sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}")
    sys.stdout.flush()

    if current == total:
        print()


def learn_from_csv(model, file_path, max_dialogs=None):
    """CSV에서 학습 (깔끔한 UI)"""
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.normpath(os.path.join(base_dir, file_path))

    if os.path.isdir(file_path):
        print(f"❌ 폴더: {file_path}")
        return

    if not os.path.isfile(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return

    # 시작 메시지
    limit_text = f"{max_dialogs}개" if max_dialogs else "전체"
    print(f"📂 {os.path.basename(file_path)} 학습 중... ({limit_text})")

    start_time = datetime.now()

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        total_learned = 0
        total_pairs = 0
        error_count = 0

        for row_index, row in enumerate(reader, 1):
            if max_dialogs and total_learned >= max_dialogs:
                break

            raw = row.get("dialog", "")
            if not raw:
                continue

            dialog = parse_dialog(raw)

            if len(dialog) < 2:
                continue

            model.reset_all()

            for i in range(len(dialog) - 1):
                q = dialog[i]
                a = dialog[i + 1]

                try:
                    model.learn(q, a)
                    total_pairs += 1
                except Exception as e:
                    error_count += 1

            total_learned += 1

            # 프로그레스바만 표시
            if max_dialogs:
                suffix = f"{total_learned}/{max_dialogs} | {total_pairs} 쌍"
                print_progress_bar(total_learned, max_dialogs, prefix="", suffix=suffix)

            model.reset_all()

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        # 완료 메시지
        stats = model.get_stats()
        print(
            f"\n✅ 완료: {total_learned}개 대화, {total_pairs}개 쌍, {stats['chars']}개 문자 ({elapsed:.1f}초)"
        )


# ============================================================================
# CLI
# ============================================================================


def main():
    print("=" * 60)
    print(VERSION)
    print("=" * 60)
    print("Commands:")
    print('  learn "Q" "A"')
    print("  learncsv <file.csv> [count]")
    print('  stimulate "Q"')
    print("  save <file.gzip>")
    print("  load <file.gzip>")
    print("  reset")
    print("  stats")
    print("  exit")
    print("=" * 60)

    model = AlphaIntelligence()
    running = True

    while running:
        try:
            cmd = input("\n>>>").strip()

            if not cmd:
                continue

            if cmd == "exit":
                print("Goodbye!")
                running = False

            elif cmd.startswith("stimulate "):
                rest = cmd[10:]
                matches = re.findall(r'"([^"]*)"', rest)
                question = matches[0] if matches else rest.strip()

                model.reset_all()

                for char in question:
                    model.stimulate(char)
                    model.step_all()

                result = model.generate(question, verbose=True)

            elif cmd.startswith("learn "):
                rest = cmd[6:]
                matches = re.findall(r'"([^"]*)"', rest)

                if len(matches) >= 2:
                    question = matches[0]
                    answer = matches[1]
                    model.learn(question, answer)
                    print(f"✅ Learned")
                else:
                    print('Usage: learn "Q" "A"')

            elif cmd.startswith("learncsv "):
                parts = cmd.split()
                if len(parts) >= 2:
                    file_path = parts[1]
                    max_dialogs = None

                    if len(parts) == 3:
                        try:
                            max_dialogs = int(parts[2])
                        except ValueError:
                            print("❌ 숫자")
                            continue

                    learn_from_csv(model, file_path, max_dialogs)

            elif cmd.startswith("save "):
                parts = cmd.split()
                if len(parts) >= 2:
                    filepath = parts[1]
                    compression = parts[2] if len(parts) >= 3 else "gzip"

                    if compression not in ["gzip", "bz2", "lzma", "none"]:
                        print("❌ compression: gzip, bz2, lzma, none")
                        continue

                    model.save(filepath, compression)

            elif cmd.startswith("load "):
                parts = cmd.split()
                if len(parts) == 2:
                    model = AlphaIntelligence.load(parts[1])

            elif cmd == "reset":
                model.reset_all()
                print("✅ Reset")

            elif cmd == "stats":
                stats = model.get_stats()
                print(f"\n📊 Stats:")
                print(f"  문자: {stats['chars']}")
                print(f"  History: {stats['total_history']}")
                print(f"  평균: {stats['avg_history']:.1f}")

            else:
                print("❌ Unknown")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            running = False
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
