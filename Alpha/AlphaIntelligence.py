import math
import csv
import re
import os

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
SIMILARITY_K = 10.0
MAX_HISTORY = 100
MIN_SIGNAL_THRESHOLD = 1e-10
INITIAL_SIGNAL_STRENGTH = 1.0


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

    def fire(self, output_layers, state_layers):
        for conn in self.connections:
            conn.transmit(self.signal, output_layers, state_layers)

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
        self.signal *= DECAY_RATE

    def receive(self, amount):
        self.signal += amount


# ============================================================================
# CONNECTOR CLASS
# ============================================================================


class Connector:
    __slots__ = ("output_layer_id", "history")

    def __init__(self, output_layer_id):
        self.output_layer_id = output_layer_id
        self.history = []

    def transmit(self, signal, output_layers, state_layers):
        if not self.history:
            return

        history = self.history
        k = SIMILARITY_K

        # Weight 1: 신호 강도 유사도
        min_distance = float("inf")
        for history_signal, _ in history:
            distance = abs(history_signal - signal)
            if distance < min_distance:
                min_distance = distance

        weight1 = max(0.0, 1.0 - min_distance / k)

        # Weight 2: StateLayer 패턴 유사도
        total_diff = 0.0
        count = 0

        for _, state_snapshot in history:
            for char, snapshot_signal in state_snapshot.items():
                state_layer = state_layers.get(char)
                if state_layer:
                    total_diff += abs(snapshot_signal - state_layer.signal)
                    count += 1

        if count > 0:
            avg_diff = total_diff / count
            weight2 = max(0.0, 1.0 - avg_diff / k)
        else:
            weight2 = 1.0

        # 최종 가중치
        final_weight = math.sqrt(weight1 * weight2)
        weighted_signal = signal * final_weight

        # 신호 전송
        if self.output_layer_id in output_layers:
            output_layers[self.output_layer_id].receive(weighted_signal)

    def learn(self, signal, state_layers):
        state_snapshot = {char: layer.signal for char, layer in state_layers.items()}
        self.history.append((signal, state_snapshot))

        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)


# ============================================================================
# GLOBAL LAYER INITIALIZATION
# ============================================================================

ALL_CHARS = list(LETTERS) + [END_TOKEN]

output_layers = {char: OutputLayer(char) for char in ALL_CHARS}
input_layers = {char: InputLayer(char, ALL_CHARS) for char in ALL_CHARS}
state_layers = {char: StateLayer(char) for char in ALL_CHARS}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def step_all_layers():
    for layer in input_layers.values():
        layer.step()
    for layer in output_layers.values():
        layer.step()
    for layer in state_layers.values():
        layer.step()


def stimulate(char, strength=INITIAL_SIGNAL_STRENGTH):
    layer = input_layers.get(char)
    if layer:
        layer.receive(strength)


def reset_all_layers():
    for layer in input_layers.values():
        layer.signal = 0.0
    for layer in output_layers.values():
        layer.signal = 0.0
    for layer in state_layers.values():
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
    strongest_char = None
    max_signal = -1.0

    for char, layer in output_layers.items():
        if layer.signal > max_signal:
            max_signal = layer.signal
            strongest_char = char

    return strongest_char, max_signal


# ============================================================================
# CORE AI FUNCTIONS
# ============================================================================


def generate_response(max_length=200):
    """응답 생성"""
    output_text = []

    for _ in range(max_length):
        # OutputLayer 초기화
        for layer in output_layers.values():
            layer.signal = 0.0

        # InputLayer fire
        for layer in input_layers.values():
            layer.fire(output_layers, state_layers)

        strongest_char, max_signal = find_strongest_output()

        # 신호가 너무 약하면 종료
        if max_signal < MIN_SIGNAL_THRESHOLD:
            break

        # END 토큰이면 종료
        if strongest_char == END_TOKEN:
            break

        output_text.append(strongest_char)

        # StateLayer에 자극
        if strongest_char in state_layers:
            state_layers[strongest_char].receive(INITIAL_SIGNAL_STRENGTH)

        step_all_layers()

    result = "".join(output_text)
    print(f"Response: {result}")
    return result


def learn(question, answer):
    """학습 - 동적 레이어 생성 지원"""
    answer_tokens = tokenize(answer) + [END_TOKEN]

    # 1단계: Question 입력
    for char in question:
        # 새로운 문자면 동적 생성
        if char not in input_layers:
            if not ALLOW_UNICODE:
                continue

            # InputLayer 생성
            input_layers[char] = InputLayer(char, list(output_layers.keys()))

            # StateLayer 생성
            if char not in state_layers:
                state_layers[char] = StateLayer(char)

        layer = input_layers[char]
        layer.receive(INITIAL_SIGNAL_STRENGTH)
        step_all_layers()

    # 2단계: Answer 학습
    for achar in answer_tokens:
        # OutputLayer 동적 생성
        if achar not in output_layers:
            if not ALLOW_UNICODE and achar != END_TOKEN:
                continue

            output_layers[achar] = OutputLayer(achar)

            # StateLayer 생성
            if achar not in state_layers:
                state_layers[achar] = StateLayer(achar)

            # 기존 InputLayer에 새 커넥터 추가
            for ilayer in input_layers.values():
                ilayer.connections.append(Connector(achar))

        # Question의 각 문자에서 Answer 문자로 학습
        for char in question:
            if char not in input_layers:
                continue

            layer = input_layers[char]

            # 해당 OutputLayer로 가는 커넥터 찾아서 학습
            for conn in layer.connections:
                if conn.output_layer_id == achar:
                    conn.learn(layer.signal, state_layers)
                    break

        # StateLayer 자극
        if achar in state_layers:
            state_layers[achar].receive(INITIAL_SIGNAL_STRENGTH)

        step_all_layers()


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


def learn_from_csv(file_path):
    """CSV 파일에서 학습"""
    # 절대 경로 변환
    if not os.path.isabs(file_path):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.normpath(os.path.join(base_dir, file_path))

    # 폴더인지 확인
    if os.path.isdir(file_path):
        print(f"❌ 폴더 경로입니다: {file_path}")
        candidates = [f for f in os.listdir(file_path) if f.lower().endswith(".csv")]
        if candidates:
            print("사용 가능한 CSV:")
            for c in candidates:
                print(f"  - {c}")
        return

    # 파일 존재 확인
    if not os.path.isfile(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return

    print(f"📂 CSV 학습 시작: {os.path.basename(file_path)}\n")

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        total_learned = 0
        total_pairs = 0
        error_count = 0

        for row_index, row in enumerate(reader, 1):
            raw = row.get("dialog", "")
            if not raw:
                continue

            dialog = parse_dialog(raw)

            if len(dialog) < 2:
                continue

            # 첫 3개 대화 샘플 출력
            if row_index <= 3:
                print(f"행 {row_index}: {len(dialog)}개 턴")
                for i, turn in enumerate(dialog[:3], 1):
                    preview = turn[:60] + "..." if len(turn) > 60 else turn
                    print(f"  턴{i}: {preview}")
                print()

            reset_all_layers()

            # 연속 턴을 Q-A 쌍으로 학습
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

            # 진행률 표시 (500행마다)
            if row_index % 500 == 0:
                print(f"진행: {row_index}행, {total_pairs}쌍, 오류 {error_count}개")

            reset_all_layers()

        print(f"\n✅ 학습 완료")
        print(f"  - 대화: {total_learned}개")
        print(f"  - Q-A 쌍: {total_pairs}개")
        print(f"  - 오류: {error_count}개")
        print(f"  - 학습된 문자 종류: {len(input_layers)}개")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    print("=" * 60)
    print("AlphaIntelligence - Trace-Based AI")
    print("=" * 60)
    print("Commands:")
    print("  learn <question> <answer>  - Learn Q&A pair")
    print("  learncsv <file.csv>       - Learn from CSV file")
    print("  stimulate <question>       - Generate response")
    print("  reset                      - Reset all layers")
    print("  stats                      - Show statistics")
    print("  exit                       - Exit program")
    print("=" * 60)

    running = True

    while running:
        try:
            cmd = input("\n>>> ").strip()

            if not cmd:
                continue

            if cmd == "exit":
                print("Goodbye!")
                running = False

            elif cmd.startswith("stimulate "):
                parts = cmd.split(" ", 1)
                if len(parts) == 2:
                    question = parts[1]
                    reset_all_layers()

                    # Question 입력
                    for char in question:
                        stimulate(char)
                        step_all_layers()

                    # 응답 생성
                    generate_response()
                else:
                    print("Usage: stimulate <question>")

            elif cmd.startswith("learn "):
                parts = cmd.split(" ", maxsplit=2)
                if len(parts) == 3:
                    question = parts[1]
                    answer = parts[2]
                    learn(question, answer)
                    print(f"✅ Learned: {question} -> {answer}")
                else:
                    print("Usage: learn <question> <answer>")

            elif cmd.startswith("learncsv "):
                parts = cmd.split(" ", 1)
                if len(parts) == 2:
                    file_path = parts[1]
                    learn_from_csv(file_path)
                else:
                    print("Usage: learncsv <file.csv>")

            elif cmd == "reset":
                reset_all_layers()
                print("✅ All layers reset")

            elif cmd == "stats":
                print(f"\n📊 Statistics:")
                print(f"  - InputLayers: {len(input_layers)}")
                print(f"  - OutputLayers: {len(output_layers)}")
                print(f"  - StateLayers: {len(state_layers)}")

                # 커넥터 히스토리 통계
                total_history = 0
                for layer in input_layers.values():
                    for conn in layer.connections:
                        total_history += len(conn.history)

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
