import math
import csv
import re
import os
import numpy as np

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Character set
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_CAP = ALPHABET.upper()
DIGITS = "0123456789"
SPECIAL_CHARS = " .,!?;:'\"()[]{}<>@#$%^&*-_=+|\\/`~"
LETTERS = ALPHABET + ALPHABET_CAP + DIGITS + SPECIAL_CHARS
END_TOKEN = "<END>"

# 유니코드 지원 (한글, 중국어 등)
ALLOW_UNICODE = True

# Core parameters
DECAY_RATE = 0.9
STATE_DECAY_RATE = 0.1
SIMILARITY_K = 20
MAX_HISTORY = float("inf")  # 무제한 기록
MIN_SIGNAL_THRESHOLD = 0.01
INITIAL_SIGNAL_STRENGTH = 5.0

AUTO = True


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
# CONNECTOR CLASS
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
        state_layers_list,
        output_idx_map,
        state_idx_map,
    ):
        if not self.history_signals:
            return

        k = SIMILARITY_K

        # Weight 1: 신호 강도 유사도 (numpy 벡터화)
        history_signals_arr = np.array(self.history_signals)
        distances = np.abs(history_signals_arr - signal)
        min_distance = np.min(distances)
        weight1 = max(0.0, 1.0 - min_distance / k)

        # Weight 2: StateLayer 패턴 유사도 (numpy 벡터화)
        current_state = np.array([layer.signal for layer in state_layers_list])

        if self.history_states:
            history_states_arr = np.array(self.history_states)
            diffs = np.abs(history_states_arr - current_state)
            avg_diff = np.mean(diffs)
            weight2 = max(0.0, 1.0 - avg_diff / k)
        else:
            weight2 = 1.0

        # 최종 가중치
        final_weight = math.sqrt(weight1 * weight2)
        weighted_signal = signal * final_weight

        # 신호 전송
        output_idx = output_idx_map.get(self.output_layer_id)
        if output_idx is not None:
            output_layers_list[output_idx].receive(weighted_signal)

    def learn(self, signal, state_layers_list):
        state_snapshot = np.array([layer.signal for layer in state_layers_list])
        self.history_signals.append(signal)
        self.history_states.append(state_snapshot)

        if len(self.history_signals) > MAX_HISTORY:
            self.history_signals.pop(0)
            self.history_states.pop(0)


# ============================================================================
# GLOBAL LAYER INITIALIZATION
# ============================================================================

ALL_CHARS = list(LETTERS) + [END_TOKEN]

output_layers = {char: OutputLayer(char) for char in ALL_CHARS}
input_layers = {char: InputLayer(char, ALL_CHARS) for char in ALL_CHARS}
state_layers = {char: StateLayer(char) for char in ALL_CHARS}

# 리스트 변환 (빠른 순회용)
output_layers_list = list(output_layers.values())
input_layers_list = list(input_layers.values())
state_layers_list = list(state_layers.values())

# 인덱스 맵 (빠른 검색용)
output_idx_map = {char: i for i, char in enumerate(output_layers.keys())}
state_idx_map = {char: i for i, char in enumerate(state_layers.keys())}


# ============================================================================
# UTILITY FUNCTIONS
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


# ============================================================================
# CORE AI FUNCTIONS
# ============================================================================


def generate_response(max_length=200):
    """응답 생성"""
    output_text = []
    repeat_count = 0
    last_char = None

    print("Response: ", end="", flush=True)  # ✅ 시작 출력

    for _ in range(max_length):
        # OutputLayer 초기화
        for layer in output_layers_list:
            layer.signal = 0.0

        # InputLayer fire
        for layer in input_layers_list:
            layer.fire(
                output_layers_list, state_layers_list, output_idx_map, state_idx_map
            )

        strongest_char, max_signal = find_strongest_output()

        # 신호가 너무 약하면 종료
        if max_signal < MIN_SIGNAL_THRESHOLD:
            break

        # END 토큰이면 종료
        if strongest_char == END_TOKEN:
            break

        # ✅ 동일 문자 3회 연속 반복 시 종료
        if strongest_char == last_char:
            repeat_count += 1
            if repeat_count >= 3:
                print(f"\n[반복 감지: '{last_char}' x{repeat_count+1}]", flush=True)
                break
        else:
            repeat_count = 0
            last_char = strongest_char

        output_text.append(strongest_char)
        print(strongest_char, end="", flush=True)  # ✅ 실시간 한글자씩 출력

        # StateLayer에 자극
        state_idx = state_idx_map.get(strongest_char)
        if state_idx is not None:
            state_layers_list[state_idx].receive(INITIAL_SIGNAL_STRENGTH)

        step_all_layers()

    print()  # ✅ 줄바꿈
    result = "".join(output_text)
    return result


def parse_dialog(raw_text):
    """따옴표로 묶인 발화 추출"""
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


def learn(question, answer):
    """학습 - 동적 레이어 생성 제거"""
    answer_tokens = tokenize(answer) + [END_TOKEN]

    # ✅ 1단계: 미등록 문자 확인 (Question)
    missing_chars = []
    for char in question:
        if char not in input_layers:
            missing_chars.append(char)

    if missing_chars:
        unique_missing = list(set(missing_chars))
        print(f"❌ Question에 미등록 문자 발견: {unique_missing}")
        print(f"   학습 건너뜀: {question[:50]}...")
        return

    # ✅ 2단계: 미등록 문자 확인 (Answer)
    for achar in answer_tokens:
        if achar not in output_layers:
            print(f"❌ Answer에 미등록 문자 발견: [{achar}]")
            print(f"   학습 건너뜀: {answer[:50]}...")
            return
        if achar not in state_layers:
            print(f"❌ StateLayer에 미등록 문자 발견: [{achar}]")
            print(f"   학습 건너뜀: {answer[:50]}...")
            return

    # ✅ 3단계: Question 입력
    for char in question:
        layer = input_layers[char]
        layer.receive(INITIAL_SIGNAL_STRENGTH)
        step_all_layers()

    # ✅ 4단계: Answer 학습
    for achar in answer_tokens:
        # Question의 각 문자에서 Answer 문자로 학습
        for char in question:
            layer = input_layers[char]

            # 해당 OutputLayer로 가는 커넥터 찾아서 학습
            for conn in layer.connections:
                if conn.output_layer_id == achar:
                    conn.learn(layer.signal, state_layers_list)
                    break

        # StateLayer 자극
        state_idx = state_idx_map.get(achar)
        if state_idx is not None:
            state_layers_list[state_idx].receive(INITIAL_SIGNAL_STRENGTH)

        step_all_layers()


def learn_from_csv(file_path, max_dialogs=None):
    """CSV 파일에서 학습

    Args:
        file_path: CSV 파일 경로
        max_dialogs: 학습할 최대 대화 개수 (None이면 전체)
    """
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.normpath(os.path.join(base_dir, file_path))

    if os.path.isdir(file_path):
        print(f"❌ 폴더 경로입니다: {file_path}")
        candidates = [f for f in os.listdir(file_path) if f.lower().endswith(".csv")]
        if candidates:
            print("사용 가능한 CSV:")
            for c in candidates:
                print(f"  - {c}")
        return

    if not os.path.isfile(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return

    limit_text = f"{max_dialogs}개" if max_dialogs else "전체"
    print(f"📂 CSV 학습 시작: {os.path.basename(file_path)} ({limit_text})\n")

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        total_learned = 0
        total_pairs = 0
        error_count = 0
        skipped_count = 0  # ✅ 건너뛴 대화 카운트

        for row_index, row in enumerate(reader, 1):
            # 최대 개수 제한
            if max_dialogs and total_learned >= max_dialogs:
                print(f"\n⏸️ {max_dialogs}개 대화 학습 완료, 중단")
                break

            raw = row.get("dialog", "")
            if not raw:
                continue

            dialog = parse_dialog(raw)

            if len(dialog) < 2:
                continue

            # 첫 3개 샘플
            if row_index <= 3:
                print(f"행 {row_index}: {len(dialog)}개 턴")
                for i, turn in enumerate(dialog[:3], 1):
                    preview = turn[:60] + "..." if len(turn) > 60 else turn
                    print(f"  턴{i}: {preview}")
                print()

            if row_index == 4:
                print("📖 학습 시작...\n")

            reset_all_layers()

            # ✅ 대화 전체 미등록 문자 사전 검사
            skip_dialog = False
            for i in range(len(dialog) - 1):
                q = dialog[i]
                a = dialog[i + 1]

                # Question 검사
                for char in q:
                    if char not in input_layers:
                        skip_dialog = True
                        break

                if skip_dialog:
                    break

                # Answer 검사
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
                    print(f"⚠️ 행 {row_index} 건너뜀 (미등록 문자)")
                continue

            # ✅ 학습 진행
            for i in range(len(dialog) - 1):
                q = dialog[i]
                a = dialog[i + 1]

                try:
                    learn(q, a)
                    total_pairs += 1
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:
                        print(f"⚠️ 학습 오류 (행{row_index}): {str(e)[:50]}")

            total_learned += 1

            # 진행률 표시
            if total_learned % 10 == 0:
                if max_dialogs:
                    percent = int((total_learned / max_dialogs) * 100)
                    print(
                        f"[{total_learned}/{max_dialogs}] {percent}% - {total_pairs}쌍 학습 완료"
                    )
                else:
                    print(f"[{total_learned}개] {total_pairs}쌍 학습 완료")

            reset_all_layers()

        print(f"\n✅ 학습 완료")
        print(f"  - 대화: {total_learned}개")
        print(f"  - 건너뜀: {skipped_count}개")
        print(f"  - Q-A 쌍: {total_pairs}개")
        print(f"  - 오류: {error_count}개")
        print(f"  - 문자 종류: {len(input_layers)}개")


def main():
    print("=" * 60)
    print("AlphaIntelligence - Trace-Based AI")
    print("=" * 60)
    print("Commands:")
    print('  learn "question" "answer"     - Learn Q&A pair')
    print("  learncsv <file.csv> [count]   - Learn from CSV (optional count)")
    print('  stimulate "question"          - Generate response')
    print("  reset                         - Reset all layers")
    print("  stats                         - Show statistics")
    print("  exit                          - Exit program")
    print("=" * 60)
    print("\nExamples:")
    print('  stimulate "Say , Jim , how about going"')
    print('  learn "hello" "hi there"')
    print("  learncsv train.csv 100        - Learn first 100 dialogs")
    print("=" * 60)

    running = True

    while running:
        try:
            if AUTO:
                print("\n>>> auto")
                cmd = "auto"
            else:
                cmd = input("\n>>> ").strip()

            if not cmd:
                continue

            if cmd == "exit":
                print("Goodbye!")
                running = False

            elif cmd == "auto":
                print("Auto mode activated")

                learn_from_csv("Alpha/DataSet/archive/train.csv", max_dialogs=1)
                reset_all_layers()

                question = "Say , Jim , how about going for a few beers after dinner ?"

                for char in question:
                    stimulate(char)
                    step_all_layers()

                generate_response()
                break

            elif cmd.startswith("stimulate "):
                rest = cmd[10:]  # "stimulate " 제거

                # 따옴표 파싱
                matches = re.findall(r'"([^"]*)"', rest)
                if matches:
                    question = matches[0]
                else:
                    # 따옴표 없으면 전체를 question으로
                    question = rest.strip()

                for char in question:
                    stimulate(char)
                    step_all_layers()

                generate_response()

            elif cmd.startswith("learn "):
                rest = cmd[6:]  # "learn " 제거

                # 따옴표 파싱
                matches = re.findall(r'"([^"]*)"', rest)
                if len(matches) >= 2:
                    question = matches[0]
                    answer = matches[1]
                    learn(question, answer)
                    print(f"✅ Learned: {question} -> {answer}")
                else:
                    print('Usage: learn "question" "answer"')

            elif cmd.startswith("learncsv "):
                parts = cmd.split()
                if len(parts) >= 2:
                    file_path = parts[1]
                    max_dialogs = None

                    # 개수 파라미터 확인
                    if len(parts) == 3:
                        try:
                            max_dialogs = int(parts[2])
                            if max_dialogs <= 0:
                                print("❌ 개수는 양수여야 합니다")
                                continue
                        except ValueError:
                            print("❌ 개수는 정수여야 합니다")
                            continue

                    learn_from_csv(file_path, max_dialogs)
                else:
                    print("Usage: learncsv <file.csv> [count]")

            elif cmd == "reset":
                reset_all_layers()
                print("✅ All layers reset")

            elif cmd == "stats":
                print(f"\n📊 Statistics:")
                print(f"  - InputLayers: {len(input_layers)}")
                print(f"  - OutputLayers: {len(output_layers)}")
                print(f"  - StateLayers: {len(state_layers)}")

                total_history = 0
                for layer in input_layers_list:
                    for conn in layer.connections:
                        total_history += len(conn.history_signals)

                print(f"  - Total connection histories: {total_history}")
                print(f"  - Unicode support: {'ON' if ALLOW_UNICODE else 'OFF'}")

            else:
                print("❌ Unknown command")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            running = False
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
