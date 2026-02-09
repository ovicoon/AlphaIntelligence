import math

LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ---------- Layers ----------


class InputLayer:
    def __init__(self, char):
        self.char = char
        self.signal = 0.0
        self.connections = []

    def connect(self, output_layer, state_layers):
        self.connections.append(Connector(output_layer, state_layers))

    def step(self):
        self.signal *= 0.9

    def fire(self):
        for conn in self.connections:
            conn.transmit(self.signal)

    def receive(self, amount):
        self.signal += amount


class OutputLayer:
    def __init__(self, char):
        self.char = char
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


# ---------- Connector ----------


class Connector:
    def __init__(self, output_layer, state_layers):
        self.output_layer = output_layer
        self.state_layers = state_layers
        self.weight = 1.0
        self.history = []

    def transmit(self, signal):
        sent = signal * self.weight
        self.output_layer.receive(sent)

    def learn(self, signal):
        # 간단한 Hebbian-style learning
        self.history.append(signal)
        if len(self.history) > 50:
            self.history.pop(0)

        avg_signal = sum(self.history) / len(self.history)
        self.weight += 0.05 * avg_signal
        self.weight = max(0.0, min(self.weight, 5.0))


# ---------- Network Init ----------

output_layers = {c: OutputLayer(c) for c in LETTERS}
state_layers = {c: StateLayer(c) for c in LETTERS}
input_layers = {c: InputLayer(c) for c in LETTERS}

for inp in input_layers.values():
    for out in output_layers.values():
        inp.connect(out, state_layers)


# ---------- Utilities ----------


def step_all():
    for layer in input_layers.values():
        layer.step()
    for layer in output_layers.values():
        layer.step()
    for layer in state_layers.values():
        layer.step()


def stimulate(text, strength=1.0):
    for c in text:
        if c in input_layers:
            input_layers[c].receive(strength)


def fire_all_inputs():
    for layer in input_layers.values():
        layer.fire()


def strongest_output():
    return max(output_layers.values(), key=lambda l: l.signal)


# ---------- Main Loop ----------


def main():
    print("Type: stimulate <text> [strength], learn <q> <a>, reset, exit")

    while True:
        cmd = input(">>> ").strip()

        if cmd == "exit":
            break

        elif cmd.startswith("stimulate"):
            parts = cmd.split()
            text = parts[1]
            strength = float(parts[2]) if len(parts) == 3 else 1.0

            stimulate(text, strength)
            fire_all_inputs()
            step_all()

            out = strongest_output()
            print("Strongest Output:", out.char)
            state_layers[out.char].receive(1.0)

        elif cmd.startswith("learn"):
            _, q, a = cmd.split()
            for qc in q:
                if qc in input_layers:
                    for conn in input_layers[qc].connections:
                        if conn.output_layer.char == a:
                            conn.learn(input_layers[qc].signal)

        elif cmd == "reset":
            for d in (input_layers, output_layers, state_layers):
                for layer in d.values():
                    layer.signal = 0.0
            print("Reset complete.")

        else:
            print("Unknown command")


if __name__ == "__main__":
    main()
