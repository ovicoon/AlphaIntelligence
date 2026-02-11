import math
import csv
import re
import os
import sys
import bisect
import numpy as np
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_CAP = ALPHABET.upper()
DIGITS = "0123456789"
SPECIAL_CHARS = " .,!?;:'\"()[]{}<>@#$%^&*-_=+|\\/`~"
LETTERS = ALPHABET + ALPHABET_CAP + DIGITS + SPECIAL_CHARS
END_TOKEN = "<END>"

ALLOW_UNICODE = True

# Core parameters
DECAY_RATE = 0.9
STATE_DECAY_RATE = 0.0
SIMILARITY_K = 20
MAX_HISTORY = float("inf")
MIN_SIGNAL_THRESHOLD = 0.0
INITIAL_SIGNAL_STRENGTH = 5.0

# State weight 조절
STATE_INFLUENCE = 0.0  # State 영향력 (0~1)

AUTO = False


# ============================================================================
# OPTIMIZED CONNECTOR with Binary Search
# ============================================================================


class Connector:
    """🚀 이진 탐색 최적화 버전"""

    __slots__ = ("output_layer_id", "history_signals", "history_states")

    def __init__(self, output_layer_id):
        self.output_layer_id = output_layer_id
        self.history_signals = []  # 정렬 유지
        self.history_states = []

    def transmit(
        self,
        signal,
        output_layers_list,
        state_layers_list,
        output_idx_map,
        state_idx_map,
    ):
        if not self.history_signals:
            return

        k = SIMILARITY_K

        # 🎯 Weight 1: 이진 탐색으로 가장 가까운 신호 찾기
        pos = bisect.bisect_left(self.history_signals, signal)

        # 양쪽 후보 확인 (최대 2개)
        candidates = []

        if pos > 0:
            candidates.append((pos - 1, self.history_signals[pos - 1]))
        if pos < len(self.history_signals):
            candidates.append((pos, self.history_signals[pos]))

        if not candidates:
            return

        # 가장 가까운 것 선택
        min_idx, min_signal = min(candidates, key=lambda x: abs(x[1] - signal))
        min_distance = abs(min_signal - signal)

        weight1 = max(0.0, 1.0 - min_distance / k)

        # 🎯 Weight 2: 해당 인덱스의 state만 비교
        current_state = np.array([layer.signal for layer in state_layers_list])
        state_snapshot = self.history_states[min_idx]

        diffs = np.abs(state_snapshot - current_state)
        avg_diff = np.mean(diffs)
        weight2 = max(0.0, 1.0 - avg_diff / k)

        # 🎯 가중 기하평균
        alpha = 1.0 - STATE_INFLUENCE
        beta = STATE_INFLUENCE

        final_weight = (weight1**alpha) * (weight2**beta)

        weighted_signal = signal * final_weight

        # 신호 전송
        output_idx = output_idx_map.get(self.output_layer_id)
        if output_idx is not None:
            output_layers_list[output_idx].receive(weighted_signal)

    def learn(self, signal, state_layers_list):
        """🎯 정렬 유지하며 삽입"""
        state_snapshot = np.array([layer.signal for layer in state_layers_list])

        # 이진 탐색으로 삽입 위치 찾기
        pos = bisect.bisect_left(self.history_signals, signal)

        # 정렬 유지하며 삽입
        self.history_signals.insert(pos, signal)
        self.history_states.insert(pos, state_snapshot)

        # 최대 크기 제한
        if len(self.history_signals) > MAX_HISTORY:
            # 가장 오래된 것 제거 (첫 번째)
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
        self, output_layers_list, state_layers_list, output_idx_map, state_idx_map
    ):
        for conn in self.connections:
            conn.transmit(
                self.signal,
                output_layers_list,
                state_layers_list,
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
# GLOBALS
# ============================================================================

ALL_CHARS = list(LETTERS) + [END_TOKEN]

output_layers = {char: OutputLayer(char) for char in ALL_CHARS}
input_layers = {char: InputLayer(char, ALL_CHARS) for char in ALL_CHARS}
state_layers = {char: StateLayer(char) for char in ALL_CHARS}

output_layers_list = list(output_layers.values())
input_layers_list = list(input_layers.values())
state_layers_list = list(state_layers.values())

output_idx_map = {char: i for i, char in enumerate(output_layers.keys())}
state_idx_map = {char: i for i, char in enumerate(state_layers.keys())}


# ============================================================================
# UTILITIES
# ============================================================================


def step_all_layers():
    for layer in input_layers_list:
        layer.step()
    for layer in output_layers_list:
        layer.step()
    for layer in state_layers_list:
        layer.step()


def stimulate(char, strength=INITIAL_SIGNAL_STRENGTH):
    layer = input_layers.get(char)
    if layer:
        layer.receive(strength)


def reset_all_layers():
    for layer in input_layers_list:
        layer.signal = 0.0
    for layer in output_layers_list:
        layer.signal = 0.0
    for layer in state_layers_list:
        layer.signal = 0.0


def tokenize(text):
    tokens = []
    i = 0
    text_len = len(text)
    end_len = len(END_TOKEN)

    while i < text_len:
        if text[i : i + end_len] == END_TOKEN:
            tokens.append(END_TOKEN)
            i += end_len
        else:
            tokens.append(text[i])
            i += 1

    return tokens


def find_strongest_output():
    max_signal = -1.0
    strongest_idx = -1

    for i, layer in enumerate(output_layers_list):
        if layer.signal > max_signal:
            max_signal = layer.signal
            strongest_idx = i

    if strongest_idx >= 0:
        return output_layers_list[strongest_idx].char, max_signal
    return None, max_signal


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


# ============================================================================
# CORE FUNCTIONS
# ============================================================================


def generate_response(max_length=4096):
    output_text = []
    repeat_count = 0
    last_char = None

    print("Response: ", end="", flush=True)

    for _ in range(max_length):
        for layer in output_layers_list:
            layer.signal = 0.0

        for layer in input_layers_list:
            layer.fire(
                output_layers_list, state_layers_list, output_idx_map, state_idx_map
            )

        strongest_char, max_signal = find_strongest_output()

        if max_signal < MIN_SIGNAL_THRESHOLD:
            break

        if strongest_char == END_TOKEN:
            break

        if strongest_char == last_char:
            repeat_count += 1
            if repeat_count >= 3:
                print(f"\n[반복 감지: '{last_char}']", flush=True)
                break
        else:
            repeat_count = 0
            last_char = strongest_char

        output_text.append(strongest_char)
        print(strongest_char, end="", flush=True)

        state_idx = state_idx_map.get(strongest_char)
        if state_idx is not None:
            state_layers_list[state_idx].receive(INITIAL_SIGNAL_STRENGTH)

        step_all_layers()

    print()
    result = "".join(output_text)
    return result


def learn(question, answer):
    answer_tokens = tokenize(answer) + [END_TOKEN]

    # 미등록 문자 검사
    missing_chars = []
    for char in question:
        if char not in input_layers:
            missing_chars.append(char)

    if missing_chars:
        unique_missing = list(set(missing_chars))
        print(f"❌ Question 미등록 문자: {unique_missing}")
        print(f"   건너뜀: {question[:50]}...")
        return

    for achar in answer_tokens:
        if achar not in output_layers or achar not in state_layers:
            print(f"❌ Answer 미등록 문자: [{achar}]")
            print(f"   건너뜀: {answer[:50]}...")
            return

    # Question 입력
    for char in question:
        layer = input_layers[char]
        layer.receive(INITIAL_SIGNAL_STRENGTH)
        step_all_layers()

    # Answer 학습
    for achar in answer_tokens:
        for char in question:
            layer = input_layers[char]

            for conn in layer.connections:
                if conn.output_layer_id == achar:
                    conn.learn(layer.signal, state_layers_list)
                    break

        state_idx = state_idx_map.get(achar)
        if state_idx is not None:
            state_layers_list[state_idx].receive(INITIAL_SIGNAL_STRENGTH)

        step_all_layers()


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


def learn_from_csv(file_path, max_dialogs=None):
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.normpath(os.path.join(base_dir, file_path))

    if os.path.isdir(file_path):
        print(f"❌ 폴더: {file_path}")
        candidates = [f for f in os.listdir(file_path) if f.lower().endswith(".csv")]
        if candidates:
            print("CSV 파일:")
            for c in candidates:
                print(f"  - {c}")
        return

    if not os.path.isfile(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return

    limit_text = f"{max_dialogs}개" if max_dialogs else "전체"
    print(f"📂 CSV 학습: {os.path.basename(file_path)} ({limit_text})")
    print(f"🚀 이진 탐색 최적화 적용\n")

    start_time = datetime.now()

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        total_learned = 0
        total_pairs = 0
        error_count = 0
        skipped_count = 0

        for row_index, row in enumerate(reader, 1):
            if max_dialogs and total_learned >= max_dialogs:
                print(f"\n⏸️ {max_dialogs}개 완료")
                break

            raw = row.get("dialog", "")
            if not raw:
                continue

            dialog = parse_dialog(raw)

            if len(dialog) < 2:
                continue

            if row_index <= 3:
                print(f"샘플 {row_index}: {len(dialog)}턴")
                for i, turn in enumerate(dialog[:2], 1):
                    preview = turn[:50] + "..." if len(turn) > 50 else turn
                    print(f"  턴{i}: {preview}")
                print()

            if row_index == 4:
                print("📖 학습 시작...\n")

            reset_all_layers()

            # 미등록 문자 검사
            skip_dialog = False
            for i in range(len(dialog) - 1):
                q = dialog[i]
                a = dialog[i + 1]

                for char in q:
                    if char not in input_layers:
                        skip_dialog = True
                        break

                if skip_dialog:
                    break

                a_tokens = tokenize(a) + [END_TOKEN]
                for achar in a_tokens:
                    if achar not in output_layers or achar not in state_layers:
                        skip_dialog = True
                        break

                if skip_dialog:
                    break

            if skip_dialog:
                skipped_count += 1
                if skipped_count <= 3:
                    print(f"⚠️ 행 {row_index} 건너뜀")
                continue

            # 학습
            for i in range(len(dialog) - 1):
                q = dialog[i]
                a = dialog[i + 1]

                try:
                    learn(q, a)
                    total_pairs += 1
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"⚠️ 오류 (행{row_index}): {str(e)[:50]}")

            total_learned += 1

            # 프로그레스바
            if max_dialogs:
                suffix = f"{total_learned}/{max_dialogs} ({total_pairs} 쌍)"
                print_progress_bar(
                    total_learned, max_dialogs, prefix="진행", suffix=suffix
                )

            reset_all_layers()

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"\n✅ 학습 완료 ({elapsed:.1f}초)")
        print(f"  - 대화: {total_learned}개")
        print(f"  - 건너뜀: {skipped_count}개")
        print(f"  - Q-A 쌍: {total_pairs}개")
        print(f"  - 오류: {error_count}개")
        print(f"  - 속도: {total_pairs/elapsed:.1f} 쌍/초")


# ============================================================================
# CLI
# ============================================================================


def main():
    print("=" * 60)
    print("AlphaIntelligence v3.0 - Binary Search Optimized")
    print("🚀 O(N) → O(log N) 속도 개선")
    print("=" * 60)
    print("Commands:")
    print('  learn "Q" "A"                 - Learn Q&A')
    print("  learncsv <file.csv> [count]   - Learn from CSV")
    print('  stimulate "Q"                 - Generate')
    print("  reset                         - Reset")
    print("  stats                         - Stats")
    print("  exit                          - Exit")
    print("=" * 60)
    print(f"\n⚙️ 설정:")
    print(f"  - STATE_INFLUENCE: {STATE_INFLUENCE}")
    print(f"  - SIMILARITY_K: {SIMILARITY_K}")
    print("=" * 60)

    running = True

    while running:
        try:
            if AUTO:
                cmd = "auto"
            else:
                cmd = input("\n>>> ").strip()

            if not cmd:
                continue

            if cmd == "exit":
                print("Goodbye!")
                running = False

            elif cmd == "auto":
                print("🤖 Auto mode")
                learn_from_csv("Alpha/DataSet/archive/train.csv", max_dialogs=10)
                reset_all_layers()

                question = "Say , Jim , how about going for a few beers after dinner ?"
                for char in question:
                    stimulate(char)
                    step_all_layers()

                generate_response()
                break

            elif cmd.startswith("stimulate "):
                rest = cmd[10:]
                matches = re.findall(r'"([^"]*)"', rest)
                question = matches[0] if matches else rest.strip()

                for char in question:
                    stimulate(char)
                    step_all_layers()

                generate_response()

            elif cmd.startswith("learn "):
                rest = cmd[6:]
                matches = re.findall(r'"([^"]*)"', rest)

                if len(matches) >= 2:
                    question = matches[0]
                    answer = matches[1]
                    learn(question, answer)
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
                            if max_dialogs <= 0:
                                print("❌ 양수")
                                continue
                        except ValueError:
                            print("❌ 정수")
                            continue

                    learn_from_csv(file_path, max_dialogs)
                else:
                    print("Usage: learncsv <file.csv> [count]")

            elif cmd == "reset":
                reset_all_layers()
                print("✅ Reset")

            elif cmd == "stats":
                total_history = sum(
                    len(conn.history_signals)
                    for layer in input_layers_list
                    for conn in layer.connections
                )

                print(f"\n📊 Stats:")
                print(f"  - Chars: {len(input_layers)}")
                print(f"  - History: {total_history}")
                print(f"  - Avg/char: {total_history / len(input_layers):.1f}")
                print(f"  - 🚀 Binary search: ON")

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
