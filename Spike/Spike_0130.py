import math

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DONE = "#"
ALL_CHARS = LETTERS + DONE


# -----------------------
# Layers
# -----------------------


class InputLayer:
    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def receive(self, amount):
        self.signal += amount

    def step(self):
        self.signal *= 0.8


class StateLayer:
    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def receive(self, amount):
        self.signal += amount

    def step(self):
        self.signal *= 0.85


class OutputLayer:
    def __init__(self, char):
        self.char = char
        self.signal = 0.0

    def receive(self, amount):
        self.signal += amount

    def step(self):
        self.signal *= 0.7


# -----------------------
# Connector
# -----------------------


class Connector:
    def __init__(self, out_char):
        self.out_char = out_char
        self.history = []  # (input_signal, state_snapshot)

    def similarity(self, input_signal, state_layers):
        if not self.history:
            return 0.0

        scores = []
        for past_signal, past_state in self.history:
            d1 = abs(past_signal - input_signal)

            d2 = 0.0
            for c in past_state:
                d2 += abs(past_state[c] - state_layers[c].signal)
            d2 /= len(past_state)

            # 기하 평균 (AND 성향)
            score = math.sqrt(max(0.0, 1 - d1) * max(0.0, 1 - d2))
            scores.append(score)

        return max(scores)

    def transmit(self, input_signal, state_layers, output_layers):
        w = self.similarity(input_signal, state_layers)
        if w > 0:
            output_layers[self.out_char].receive(input_signal * w)

    def learn(self, input_signal, state_layers):
        snapshot = {c: state_layers[c].signal for c in state_layers}
        self.history.append((input_signal, snapshot))
        if len(self.history) > 200:
            self.history.pop(0)


# -----------------------
# Brain
# -----------------------


class Brain:
    def __init__(self):
        self.input_layers = {c: InputLayer(c) for c in LETTERS}
        self.state_layers = {c: StateLayer(c) for c in ALL_CHARS}
        self.output_layers = {c: OutputLayer(c) for c in ALL_CHARS}

        self.connectors = {c: {o: Connector(o) for o in ALL_CHARS} for c in LETTERS}

    def reset_signals(self):
        for layer in (
            list(self.input_layers.values())
            + list(self.state_layers.values())
            + list(self.output_layers.values())
        ):
            layer.signal = 0.0

    def step_all(self):
        for l in self.input_layers.values():
            l.step()
        for l in self.state_layers.values():
            l.step()
        for l in self.output_layers.values():
            l.step()

    # -----------------------
    # Learning
    # -----------------------

    def learn(self, question, answer):
        self.reset_signals()

        for q in question:
            if q in self.input_layers:
                self.input_layers[q].receive(1.0)
                self.step_all()

        for a in answer + DONE:
            for q in question:
                if q in self.input_layers:
                    self.connectors[q][a].learn(
                        self.input_layers[q].signal,
                        self.state_layers,
                    )
            self.state_layers[a].receive(1.0)
            self.step_all()

    # -----------------------
    # Ask / Generate
    # -----------------------

    def ask(self, text, max_len=20):
        self.reset_signals()

        # 입력 한 번만 처리
        for c in text:
            if c in self.input_layers:
                self.input_layers[c].receive(1.0)

        self.step_all()

        result = ""
        last_char = None

        for _ in range(max_len):
            # fire - 질문 입력만 사용
            for c in text:  # 모든 입력이 아니라 text의 문자들만
                if c in self.input_layers:
                    for o in self.connectors[c]:
                        self.connectors[c][o].transmit(
                            self.input_layers[c].signal,
                            self.state_layers,
                            self.output_layers,
                        )

            # choose
            best = max(
                self.output_layers.values(),
                key=lambda x: x.signal,
            )

            if best.char == DONE or best.signal < 0.05:
                break

            # refractory (폭주 방지)
            if best.char == last_char:
                self.state_layers[best.char].signal *= 0.3

            result += best.char
            last_char = best.char

            self.state_layers[best.char].receive(0.5)
            self.step_all()

        return result


# -----------------------
# CLI
# -----------------------

brain = Brain()

while True:
    cmd = input(">>> ").strip()
    if cmd == "exit":
        break

    if cmd.startswith("learn "):
        _, q, a = cmd.split()
        brain.learn(q, a)
        print("learned")

    elif cmd.startswith("ask "):
        _, q = cmd.split()
        ans = brain.ask(q)
        print("answer:", ans)
