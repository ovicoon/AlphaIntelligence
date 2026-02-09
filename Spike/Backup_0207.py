import math


LETTERS = "abcdefghijklmnopqrstuvwxyz"
END_TOKEN = "<END>"


class InputLayer:
    def __init__(self, input, output_layers):
        self.char = input
        self.signal = 0.0

        self.connections = []
        for id in output_layers.keys():
            self.connections.append(Connector(id))

    def step(self):
        self.signal *= 0.9

    def fire(self, output_layers, state_layers):
        for conn in self.connections:
            conn.transmit(self.signal, output_layers, state_layers)

    def receive(self, amount):
        self.signal += amount


class OutputLayer:
    def __init__(self, output):
        self.char = output
        self.signal = 0.0

    def step(self):
        self.signal *= 0.9

    def receive(self, amount):
        self.signal += amount


class StateLayer:
    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def step(self):
        self.signal *= 0.9

    def receive(self, amount):
        self.signal += amount


class Connector:
    def __init__(self, output_layer_id):
        self.output_layer = output_layer_id
        self.history = []

    def transmit(self, signal, output_layers, state_layers):
        if self.history:
            temp = []
            for history in self.history:
                history_signal = history[0]
                temp.append(history_signal)

            closest = min(temp, key=lambda x: abs(x - signal))
            distance = abs(closest - signal)

            weight1 = max(0.0, 1.0 - distance / 10.0)

            temp = []
            for history in self.history:
                _, state_snapshot = history
                for char, snapshot_signal in state_snapshot.items():
                    if char in state_layers:
                        temp.append(abs(snapshot_signal - state_layers[char].signal))

            if temp:
                weight2 = max(0.0, 1.0 - (sum(temp) / len(temp)) / 10.0)
            else:
                weight2 = 1.0

            weight = (weight1 * weight2) ** 0.5
            sent = signal * weight
            output_layers[self.output_layer].receive(sent)

    def learn(self, signal, state_layers):
        state_snapshot = {char: layer.signal for char, layer in state_layers.items()}
        self.history.append((signal, state_snapshot))
        if len(self.history) > 100:
            self.history.pop(0)


# END 토큰을 포함한 모든 문자로 레이어 생성
all_chars = list(LETTERS) + [END_TOKEN]

output_layers = {}
for text in all_chars:
    output_layers[text] = OutputLayer(text)

input_layers = {}
for text in all_chars:
    input_layers[text] = InputLayer(text, output_layers)

state_layers = {}
for text in all_chars:
    state_layers[text] = StateLayer(text)


def step_all_layers():
    for layer in input_layers.values():
        layer.step()
    for layer in output_layers.values():
        layer.step()
    for layer in state_layers.values():
        layer.step()


def stimulate(char, strength=1.0):
    if char in input_layers:
        input_layers[char].receive(strength)


def print_output_states():
    states = {
        char: layer.signal
        for char, layer in output_layers.items()
        if layer.signal > 0.1
    }
    print("Output Layer States:", states)


def tokenize(text):
    """문자열을 개별 문자와 END 토큰으로 분리"""
    tokens = []
    i = 0
    while i < len(text):
        if text[i : i + len(END_TOKEN)] == END_TOKEN:
            tokens.append(END_TOKEN)
            i += len(END_TOKEN)
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def print_():
    while True:
        # OutputLayer 초기화
        for layer in output_layers.values():
            layer.signal = 0.0

        # 전체 InputLayer을 shoot (fire)
        for layer in input_layers.values():
            layer.fire(output_layers, state_layers)

        # 가장 강한 출력 찾기
        strongest = None
        max_signal = -1
        for char, layer in output_layers.items():
            if layer.signal > max_signal:
                max_signal = layer.signal
                strongest = (char, layer.signal)

        if strongest and strongest[0] == END_TOKEN:
            print(strongest[0])
        else:
            print(strongest[0], end="")

        # <END> 타이핑되면 종료
        if strongest and strongest[0] == END_TOKEN:
            break

        # StateLayer에 타이핑한 글자 자극
        if strongest:
            state_layers[strongest[0]].receive(1.0)

        # step 호출
        step_all_layers()


running = True
while running:
    cmd = input(">>>")
    if cmd == "exit":
        running = False
        continue
    elif cmd.startswith("stimulate "):
        parts = cmd.split(" ")
        if len(parts) == 3:
            text = parts[1]
            strength = float(parts[2])
            for char in text:
                stimulate(char, strength)
                step_all_layers()
            for layer in input_layers.values():
                layer.fire(output_layers, state_layers)
        elif len(parts) == 2:
            text = parts[1]
            # Q의 각 char을 step을 호출하면서 Input
            for char in text:
                stimulate(char)
                step_all_layers()
            print_()

    elif cmd.startswith("learn "):
        parts = cmd.split(" ")
        if len(parts) == 3:
            question = parts[1]
            answer = parts[2]
            answer_tokens = tokenize(answer) + [END_TOKEN]

            # Q의 각 char을 step을 호출하면서 Input
            for char in question:
                if char in input_layers:
                    input_layers[char].receive(1.0)
                    step_all_layers()

            # for achar in A
            for achar in answer_tokens:
                # Q의 각 char에서 achar로 가는 커넥터를 찾는다
                for char in question:
                    if char in input_layers:
                        for conn in input_layers[char].connections:
                            if conn.output_layer == achar:
                                # 커넥터에 history 추가
                                conn.learn(input_layers[char].signal, state_layers)

                # StateLayer에 현재 achar 자극
                state_layers[achar].receive(1.0)

                # step 호출
                step_all_layers()
    elif cmd.startswith("learn1 "):
        print("Warning: Learn1 is deprecated.")
        parts = cmd.split(" ")
        if len(parts) == 3:
            question = parts[1]
            answer = parts[2]
            answer_tokens = tokenize(answer) + [END_TOKEN]

            # Q의 각 char을 step을 호출하면서 Input
            for char in question:
                if char in input_layers:
                    input_layers[char].receive(1.0)
                    step_all_layers()

            # for achar in A
            for achar in answer_tokens:
                # Q의 각 char에서 achar로 가는 커넥터를 찾는다
                for char in question:
                    if char in input_layers:
                        for conn in input_layers[char].connections:
                            if conn.output_layer == achar:
                                # 커넥터에 history 추가
                                conn.learn(input_layers[char].signal, state_layers)

                # step 호출
                step_all_layers()

                # OutputLayer 초기화 후 fire
                for layer in output_layers.values():
                    layer.signal = 0.0

                for layer in input_layers.values():
                    layer.fire(output_layers, state_layers)

                # OutputLayer에서 가장 큰 char 찾기
                strongest_char = None
                max_signal = -1
                for char, layer in output_layers.items():
                    if layer.signal > max_signal:
                        max_signal = layer.signal
                        strongest_char = char

                # StateLayer에 현재 가장 신호가 높은 char을 자극
                if strongest_char:
                    state_layers[strongest_char].receive(1.0)

    elif cmd == "reset":
        for layer in input_layers.values():
            layer.signal = 0.0
        for layer in output_layers.values():
            layer.signal = 0.0
        for layer in state_layers.values():
            layer.signal = 0.0
        print("All layers have been reset.")
    else:
        print("Unknown command")
