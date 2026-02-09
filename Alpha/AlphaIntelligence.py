import math
import csv

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

# Core parameters
DECAY_RATE = 0.9  # 망각 상수 (signal decay per step)
SIMILARITY_K = 10.0  # 최대 거리 함수 (유사도 계산 상수)
MAX_HISTORY = 100  # 최대 Connector 히스토리 개수

# Signal thresholds
MIN_SIGNAL_THRESHOLD = 1e-10  # 최소 신호 강도 (이보다 약하면 무시)

# Learning constants
INITIAL_SIGNAL_STRENGTH = 1.0  # 초기 신호 강도


# ============================================================================
# LAYER CLASSES
# ============================================================================


class InputLayer:
    """입력 레이어: 사용자에게 들은 것"""

    __slots__ = ("char", "signal", "connections")  # 메모리 최적화

    def __init__(self, char, output_layer_ids):
        self.char = char
        self.signal = 0.0
        # Connector 리스트 사전 할당
        self.connections = [Connector(oid) for oid in output_layer_ids]

    def step(self):
        """신호 감쇠"""
        self.signal *= DECAY_RATE

    def fire(self, output_layers, state_layers):
        """신호 전달 (최적화: 리스트 순회)"""
        for conn in self.connections:
            conn.transmit(self.signal, output_layers, state_layers)

    def receive(self, amount):
        """신호 수신 (inplace 연산)"""
        self.signal += amount


class OutputLayer:
    """출력 레이어: 말할까? 라고 생각한 것"""

    __slots__ = ("char", "signal")  # 메모리 최적화

    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def step(self):
        """신호 감쇠"""
        self.signal *= DECAY_RATE

    def receive(self, amount):
        """신호 수신 (inplace 연산)"""
        self.signal += amount


class StateLayer:
    """상태 레이어: 내가 말한 것을 내가 들은 것"""

    __slots__ = ("char", "signal")  # 메모리 최적화

    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def step(self):
        """신호 감쇠"""
        self.signal *= DECAY_RATE

    def receive(self, amount):
        """신호 수신 (inplace 연산)"""
        self.signal += amount


# ============================================================================
# CONNECTOR CLASS
# ============================================================================


class Connector:
    """InputLayer와 OutputLayer를 연결하는 커넥터"""

    __slots__ = ("output_layer_id", "history")  # 메모리 최적화

    def __init__(self, output_layer_id):
        self.output_layer_id = output_layer_id
        self.history = []  # [(signal, state_snapshot), ...]

    def transmit(self, signal, output_layers, state_layers):
        """신호 전달 (유사도 기반 가중치 적용)"""
        if not self.history:
            return

        # 최적화: 지역 변수로 참조 저장
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

        # 최종 가중치 (기하평균)
        final_weight = math.sqrt(weight1 * weight2)

        # 신호 전송
        weighted_signal = signal * final_weight
        output_layers[self.output_layer_id].receive(weighted_signal)

    def learn(self, signal, state_layers):
        """학습: history에 (신호, StateLayer 스냅샷) 저장"""
        # StateLayer 스냅샷 생성 (최적화: dict comprehension)
        state_snapshot = {char: layer.signal for char, layer in state_layers.items()}

        # History 추가
        self.history.append((signal, state_snapshot))

        # History 크기 제한
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)


# ============================================================================
# GLOBAL LAYER INITIALIZATION
# ============================================================================

# 전체 문자 집합 (알파벳 + END 토큰)
ALL_CHARS = list(LETTERS) + [END_TOKEN]

# 레이어 생성 (최적화: dict comprehension)
output_layers = {char: OutputLayer(char) for char in ALL_CHARS}
input_layers = {char: InputLayer(char, ALL_CHARS) for char in ALL_CHARS}
state_layers = {char: StateLayer(char) for char in ALL_CHARS}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def step_all_layers():
    """모든 레이어의 step() 호출 (최적화: 한 번의 순회)"""
    for layer in input_layers.values():
        layer.step()
    for layer in output_layers.values():
        layer.step()
    for layer in state_layers.values():
        layer.step()


def stimulate(char, strength=INITIAL_SIGNAL_STRENGTH):
    """InputLayer 자극"""
    layer = input_layers.get(char)
    if layer:
        layer.receive(strength)


def reset_all_layers():
    """모든 레이어 초기화"""
    for layer in input_layers.values():
        layer.signal = 0.0
    for layer in output_layers.values():
        layer.signal = 0.0
    for layer in state_layers.values():
        layer.signal = 0.0


def tokenize(text):
    """문자열을 개별 문자와 END 토큰으로 분리"""
    tokens = []
    i = 0
    text_len = len(text)
    end_len = len(END_TOKEN)

    while i < text_len:
        # END 토큰 확인
        if text[i : i + end_len] == END_TOKEN:
            tokens.append(END_TOKEN)
            i += end_len
        else:
            tokens.append(text[i])
            i += 1

    return tokens


def find_strongest_output():
    """OutputLayer에서 가장 강한 신호를 가진 문자 찾기"""
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


def generate_response():
    """응답 생성 (출력 순서도)"""
    while True:
        # OutputLayer 초기화
        for layer in output_layers.values():
            layer.signal = 0.0

        # 전체 InputLayer fire
        for layer in input_layers.values():
            layer.fire(output_layers, state_layers)

        # 가장 강한 출력 찾기
        strongest_char, max_signal = find_strongest_output()

        # 출력
        print(f"Strongest Output: ('{strongest_char}', {max_signal})")

        # 신호가 너무 약하면 종료
        if max_signal < MIN_SIGNAL_THRESHOLD:
            print("신호가 너무 약해서 종료합니다.")
            break

        # END 토큰이면 종료
        if strongest_char == END_TOKEN:
            break

        # StateLayer에 출력한 문자 자극
        state_layers[strongest_char].receive(INITIAL_SIGNAL_STRENGTH)

        # Step
        step_all_layers()


def learn(question, answer):
    """학습 (학습 순서도)"""
    # Answer 토큰화 + END 토큰 추가
    answer_tokens = tokenize(answer) + [END_TOKEN]

    # Q의 각 char을 step을 호출하면서 Input
    for char in question:
        layer = input_layers.get(char)
        if layer:
            layer.receive(INITIAL_SIGNAL_STRENGTH)
            step_all_layers()

    # For each character in A
    for achar in answer_tokens:
        # Q의 각 char에서 achar로 가는 커넥터 학습
        for char in question:
            layer = input_layers.get(char)
            if layer:
                for conn in layer.connections:
                    if conn.output_layer_id == achar:
                        conn.learn(layer.signal, state_layers)

        # StateLayer에 achar 자극
        state_layers[achar].receive(INITIAL_SIGNAL_STRENGTH)

        # Step
        step_all_layers()


def learn_from_csv(file_path):
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialog = ast.literal_eval(row["dialog"])

            reset_all_layers()  # 대화 시작

            for i in range(len(dialog) - 1):
                q = dialog[i].strip()
                a = dialog[i + 1].strip()

                learn(q, a)
                if (i + 1) / (len(dialog) - 1) >= 0.1:  # 10%마다 진행 상황 출력
                    print(f"{(i+1)/(len(dialog)-1) * 100}%")

            reset_all_layers()  # 대화 종료


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    """메인 루프"""
    print("=" * 60)
    print("AlphaIntelligence - Trace-Based AI")
    print("=" * 60)
    print("Commands:")
    print("  learn <question> <answer>  - Learn Q&A pair")
    print("  learncsv <file.csv>       - Learn Q&A pairs from CSV file")
    print("  stimulate <question>       - Generate response")
    print("  reset                      - Reset all layers")
    print("  exit                       - Exit program")
    print("=" * 60)

    running = True

    while running:
        try:
            cmd = input(">>> ").strip()

            if not cmd:
                continue

            if cmd == "exit":
                print("Goodbye!")
                running = False

            elif cmd.startswith("stimulate "):
                parts = cmd.split(" ", 1)
                if len(parts) == 2:
                    question = parts[1]
                    # Q의 각 char을 step을 호출하면서 Input
                    for char in question:
                        stimulate(char)
                        step_all_layers()
                    # 응답 생성
                    generate_response()
                else:
                    print("Usage: stimulate <question>")

            elif cmd.startswith("learn "):
                parts = cmd.split(" ")
                if len(parts) == 3:
                    question = parts[1]
                    answer = parts[2]
                    learn(question, answer)
                    print(f"Learned: {question} -> {answer}")
                else:
                    print("Usage: learn <question> <answer>")

            elif cmd.startswith("learncsv "):
                parts = cmd.split(" ")
                if len(parts) == 2:
                    file_path = parts[1]
                    learn_from_csv(file_path)
                else:
                    print("Usage: learncsv <file.csv>")

            elif cmd == "reset":
                reset_all_layers()
                print("All layers have been reset.")

            else:
                print("Unknown command. Type 'exit' to quit.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            running = False
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
