import math


LETTERS = "abcdefghijklmnopqrstuvwxyz"


class InputLayer:
    def __init__(self, input, output_layers):
        self.char = input
        self.signal = 0.0

        self.connections = []
        for id in output_layers.keys():
            self.connections.append(Connector(id))

    def step(self):
        self.signal *= 0.9  # 신호 감쇠

    def fire(self):
        for conn in self.connections:
            conn.transmit(self.signal)

    def receive(self, amount):
        self.signal += amount


class OutputLayer:
    def __init__(self, output):
        self.char = output
        self.signal = 0.0

    def step(self):
        self.signal *= 0.9  # 신호 감쇠

    def receive(self, amount):
        self.signal += amount


class StateLayer:
    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def step(self):
        self.signal *= 0.9  # 신호 감쇠

    def receive(self, amount):
        self.signal += amount


class Connector:
    def __init__(self, output_layer_id):
        self.output_layer = output_layer_id

        self.history = []  # 과거 학습에서 신호 강도 저장

    def transmit(self, signal):
        if self.history:
            temp = []
            for history in self.history:
                history = history[0]
                temp.append(history)

            closest = min(temp, key=lambda x: abs(x - signal))
            distance = abs(closest - signal)

            weight1 = max(0.0, 1.0 - distance / 10.0)  # 거리에 따른 가중치 계산
            temp = []
            for history in self.history:
                trash, keep = history
                for s in keep.values():
                    temp.append(abs(s.signal - state_layers[s.char].signal))

            weight2 = max(
                0.0, 1.0 - (sum(temp) / len(temp)) / 10.0
            )  # 과거 변화에 따른 가중치 계산
            weight = (weight1 * weight2) ** 0.5  # 최종 가중치 계산
            sent = signal * weight  # 가중치 적용
            output_layers[self.output_layer].receive(sent)

    def learn(self, signal):
        self.history.append((signal, state_layers))
        if len(self.history) > 100:
            self.history.pop(0)  # 오래된 기록 제거


output_layers = {}
for text in LETTERS:
    output_layers[text] = OutputLayer(text)

input_layers = {}
for text in LETTERS:
    input_layers[text] = InputLayer(text, output_layers)

state_layers = {}
for text in LETTERS:
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


def print_():
    while True:
        for layer in output_layers.values():
            layer.signal = 0.0
        for layer in input_layers.values():
            layer.fire()
        step_all_layers()

        strongest = None
        for char, layer in output_layers.items():
            if strongest is None or layer.signal > strongest[1]:
                strongest = (char, layer.signal)

        print("Strongest Output:", strongest)
        for char, layer in state_layers.items():
            if char == strongest[0]:
                layer.receive(1.0)


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
                layer.fire()
        elif len(parts) == 2:
            text = parts[1]
            for char in text:
                stimulate(char)
                step_all_layers()
            print_()

    elif cmd.startswith("learn "):
        parts = cmd.split(" ")
        if len(parts) == 3:
            question = parts[1]
            answer = parts[2]
            for char in question:
                if char in input_layers:
                    for a_char in answer:
                        for conn in input_layers[char].connections:
                            if conn.output_layer == a_char:
                                conn.learn(input_layers[char].signal)
                        state_layers[a_char].receive(1.0)

    elif cmd == "reset":
        for layer in input_layers.values():
            layer.signal = 0.0
        for layer in output_layers.values():
            layer.signal = 0.0
        for layer in state_layers.values():
            layer.signal = 0.0
        print("All layers have been reset.")
