import random
import time
import matplotlib.pyplot as plt

plt.ion()


# 소문자 a-z 문자열
LETTERS = "abcdefghijklmnopqrstuvwxyz"


# ---------------------------
# 뉴런 클래스
# ---------------------------
class Neuron:
    def __init__(self, nid):
        self.id = nid
        self.activation = random.uniform(0.0, 0.1)
        self.signal = {}  # {signal_id: amount}
        self.fatigue = 0.0
        self.bias = random.uniform(-0.02, 0.02)
        self.synapses = {}  # {target_id: weight}

    def step(self, noise_level, neurons):
        # 1. 신호 → 활성도에 영향
        self.activation += sum(self.signal.values()) * 0.1

        # 2. 발화
        self.maybe_fire(neurons)

        # 3. 신호 감쇠 (메아리)
        for i in self.signal:
            self.signal[i] *= 0.99

        # 4. 노이즈 + bias
        self.activation += random.gauss(0, noise_level) + self.bias

        # 5. 피로
        self.activation -= self.fatigue
        self.activation = max(0.0, self.activation)

        self.fatigue += self.activation * 0.01
        self.fatigue *= 0.95

        # 6. 시냅스 가소성 감소
        for t in self.synapses:
            self.synapses[t] *= 0.999  # 아주 느린 망각

    def maybe_fire(self, neurons):
        if self.activation > 0.4 and sum(self.signal.values()) > 0.01:
            for target_id, weight in self.synapses.items():
                sent = {}  # 보낼 신호 초기화
                for i in self.signal:
                    sent[i] = self.signal[i] * weight
                neurons[target_id].receive(sent)

                # 시냅스 강화
                self.synapses[target_id] += 0.01 * sum(sent.values())
                self.synapses[target_id] = min(self.synapses[target_id], 2.0)

            # 자기 신호 감소
            for i in self.signal:
                self.signal[i] *= 0.3

    def receive(self, amount):
        for i in amount:
            self.signal[i] = min(self.signal.get(i, 0) + amount[i], 5.0)


# ---------------------------
# 경쟁 구조
# ---------------------------
def competition(neurons, survive_ratio=0.2, inhibition_strength=0.5):
    sorted_neurons = sorted(neurons, key=lambda n: n.activation, reverse=True)
    cutoff = max(1, int(len(neurons) * survive_ratio))

    for i, neuron in enumerate(sorted_neurons):
        if i < cutoff:
            # 상위 강화
            neuron.activation *= 1.05
        else:
            # 하위 억제
            neuron.activation *= inhibition_strength


# ---------------------------
# 글로벌 상태
# ---------------------------
class GlobalState:
    def __init__(self):
        self.noise_level = 0.05

    def step(self):
        # noise 조금 흔들기
        self.noise_level *= random.uniform(0.98, 1.02)
        self.noise_level = min(max(self.noise_level, 0.01), 0.2)


# ---------------------------
# Input 시스템
# ---------------------------
class InputSystem:
    def __init__(self, neurons, neurons_per_char=5, strength=1):
        self.neurons = neurons
        self.neurons_per_char = neurons_per_char
        self.strength = strength

    def stimulate(self, char):
        # 문자 하나당 랜덤 뉴런 몇 개 자극
        targets = random.sample(self.neurons, self.neurons_per_char)
        for n in targets:
            n.signal[char] = min(n.signal.get(char, 0) + self.strength, 5.0)


# ---------------------------
# 시각화 함수
# ---------------------------
def visualize(neurons, step):
    xs = [n.id for n in neurons]
    ys = [0] * len(neurons)

    activations = [n.activation for n in neurons]
    signals = [n.signal for n in neurons]

    plt.clf()
    plt.scatter(
        xs,
        ys,
        s=[sum(s.values()) * 500 for s in signals],  # 신호 → 크기
        c=activations,  # 활성 → 색
        cmap="viridis",
        alpha=0.8,
    )

    plt.ylim(-1, 1)
    plt.title(f"Tick {step}")
    plt.colorbar(label="Activation")
    plt.pause(0.01)


# ---------------------------
# 메인 루프
# ---------------------------
def main():
    NUM_NEURONS = 50
    TICKS = 200

    neurons = [Neuron(i) for i in range(NUM_NEURONS)]
    global_state = GlobalState()
    input_system = InputSystem(neurons)

    for n in neurons:
        targets = random.sample(range(NUM_NEURONS), k=random.randint(2, 5))
        for t in targets:
            if t != n.id:
                n.synapses[t] = random.uniform(0.05, 0.3)

    for tick in range(TICKS):
        if tick % 6 == 0:
            input_system.stimulate(random.choice(LETTERS))

        # 각 뉴런 업데이트
        for n in neurons:
            n.step(global_state.noise_level, neurons)

        # 경쟁 적용
        competition(neurons)

        # 글로벌 상태 업데이트
        global_state.step()

        # 시각화
        visualize(neurons, tick)

        time.sleep(0.05)


if __name__ == "__main__":
    main()
