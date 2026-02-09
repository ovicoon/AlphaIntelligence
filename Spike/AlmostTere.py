# AlphaIntelligence v3.0 — 전체 재작성 (버그 수정판)
# - 출력 어휘를 토큰 리스트로 관리 (특수 토큰 분리)
# - END 토큰을 단일 토큰으로 처리
# - END 토큰에서 다른 토큰으로 가는 연결을 만들지 않아 self-loop 방지
# - Input 신호는 첫 예측 후 제거 -> 상태 기반 생성으로 순서 보장

import math

# 기본 문자 집합 (입력)
LETTERS = list("abcdefghijklmnopqrstuvwxyz ")  # input vocab (문자 단위)
END_TOKEN = "<END>"

# 출력 어휘는 문자 + 특수 토큰(단일 항목)
OUTPUT_TOKENS = LETTERS + [END_TOKEN]  # 리스트 합치기 (END_TOKEN은 한 항목)


# ---------------------------
# Connector: 가중치 기반
# ---------------------------
class Connector:
    def __init__(self, output_token):
        self.output_token = output_token
        self.weight = 0.0

    def transmit_fraction(self, source_signal, total_weight):
        if total_weight <= 0.0 or self.weight <= 0.0:
            return 0.0
        frac = self.weight / total_weight
        return source_signal * frac

    def learn(self, amount, lr=1.0, max_w=500.0):
        self.weight += amount * lr
        if self.weight > max_w:
            self.weight = max_w

    def decay(self, rate=0.999):
        self.weight *= rate
        if self.weight < 1e-8:
            self.weight = 0.0


# ---------------------------
# Layers
# ---------------------------
class InputLayer:
    def __init__(self, token, output_tokens):
        self.token = token
        self.signal = 0.0
        # connections: token -> Connector (to each output token)
        self.connections = {out: Connector(out) for out in output_tokens}

    def step(self):
        # 입력은 빠르게 사라지도록 한다
        self.signal *= 0.5
        if self.signal < 1e-4:
            self.signal = 0.0

    def fire(self, output_layers):
        if self.signal <= 0.0:
            return
        total_w = sum(conn.weight for conn in self.connections.values())
        if total_w <= 0.0:
            return
        for conn in self.connections.values():
            amt = conn.transmit_fraction(self.signal, total_w)
            if amt > 0.0:
                output_layers[conn.output_token].receive(amt)

    def receive(self, amount):
        self.signal = min(self.signal + amount, 20.0)


class StateLayer:
    def __init__(self, token, output_tokens, allow_outgoing=True):
        self.token = token
        self.signal = 0.0
        # if allow_outgoing is False (for END token), do not create outgoing connectors
        if allow_outgoing:
            self.connections = {out: Connector(out) for out in output_tokens}
        else:
            self.connections = {}  # no outgoing connections (END token)

    def step(self):
        # 상태는 비교적 오래 유지
        self.signal *= 0.75
        if self.signal < 1e-4:
            self.signal = 0.0

    def fire(self, output_layers):
        if self.signal <= 0.0:
            return
        total_w = sum(conn.weight for conn in self.connections.values())
        if total_w <= 0.0:
            return
        for conn in self.connections.values():
            amt = conn.transmit_fraction(self.signal, total_w)
            if amt > 0.0:
                output_layers[conn.output_token].receive(amt)

    def receive(self, amount):
        self.signal = min(self.signal + amount, 40.0)


class OutputLayer:
    def __init__(self, token):
        self.token = token
        self.signal = 0.0

    def step(self):
        self.signal *= 0.90
        if self.signal < 1e-6:
            self.signal = 0.0

    def receive(self, amount):
        self.signal += amount


# ---------------------------
# Create layers
# ---------------------------
output_layers = {tok: OutputLayer(tok) for tok in OUTPUT_TOKENS}
input_layers = {tok: InputLayer(tok, OUTPUT_TOKENS) for tok in LETTERS}
# For state layers: allow_outgoing False for END_TOKEN to prevent END -> anything
state_layers = {}
for tok in OUTPUT_TOKENS:
    allow = tok != END_TOKEN
    state_layers[tok] = StateLayer(tok, OUTPUT_TOKENS, allow_outgoing=allow)


# ---------------------------
# Utilities
# ---------------------------
def step_all():
    for l in input_layers.values():
        l.step()
    for l in state_layers.values():
        l.step()
    for l in output_layers.values():
        l.step()
    # connector decay
    for layer in list(input_layers.values()) + list(state_layers.values()):
        for conn in layer.connections.values():
            conn.decay(0.9995)


def reset():
    for l in input_layers.values():
        l.signal = 0.0
    for l in state_layers.values():
        l.signal = 0.0
    for l in output_layers.values():
        l.signal = 0.0


def fire_all(use_input=True, use_state=True):
    if use_input:
        for l in input_layers.values():
            l.fire(output_layers)
    if use_state:
        for l in state_layers.values():
            l.fire(output_layers)


def get_top_outputs(n=5):
    items = [(tok, l.signal) for tok, l in output_layers.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:n]


# ---------------------------
# Learning (Q -> A)
# ---------------------------
def learn(Q, A, verbose=True):
    """
    Q: 입력 문자열 (문자 단위, 예: 'apple')
    A: 정답 문자열 (문자 단위, 예: 'yes')
    """
    if verbose:
        print(f"학습: '{Q}' → '{A}'")
    reset()

    # 1) Q 입력
    for ch in Q:
        if ch in input_layers:
            input_layers[ch].receive(6.0)  # 강하게 입력
    for _ in range(3):
        step_all()
    if verbose:
        print("  Q 입력 완료")

    # 2) A 순차 학습 (끝에 END_TOKEN 추가)
    A_tokens = list(A) + [END_TOKEN]
    for i, a_tok in enumerate(A_tokens):
        if verbose:
            print(f"  {i+1}. '{a_tok}'", end="")

        # 현재 활성화된 신호들
        current_inputs = {
            c: layer.signal for c, layer in input_layers.items() if layer.signal > 0.05
        }
        current_states = {
            c: layer.signal for c, layer in state_layers.items() if layer.signal > 0.05
        }

        # Input -> a_tok 학습
        for q_tok, strength in current_inputs.items():
            q_layer = input_layers[q_tok]
            if a_tok in q_layer.connections:
                # input에서 얻은 신호를 기반으로 weight 증가 (input은 조금 약한 학습)
                q_layer.connections[a_tok].learn(strength, lr=0.7, max_w=500.0)

        # State -> a_tok 학습
        for s_tok, strength in current_states.items():
            s_layer = state_layers[s_tok]
            if a_tok in s_layer.connections:
                # state에서의 신호는 다음 토큰을 가르치는 데 더 중요
                s_layer.connections[a_tok].learn(strength, lr=1.0, max_w=800.0)

        # 로그
        inp_chars = ",".join(sorted(current_inputs.keys()))
        sta_chars = ",".join(sorted(current_states.keys()))
        if verbose:
            print(f" [I:{inp_chars}] [S:{sta_chars}]")

        # 정답 토큰을 상태로 활성화
        if a_tok in state_layers:
            state_layers[a_tok].receive(12.0)

        # 시간 진행 (감쇠)
        for _ in range(2):
            step_all()

    if verbose:
        print("  학습 완료\n")


# ---------------------------
# Generate
# ---------------------------
def generate(prompt, max_len=40, verbose=True):
    """
    prompt: 입력 문자열 (예: 'apple')
    """
    reset()
    if verbose:
        print(f"입력: '{prompt}'")

    # 프롬프트 입력
    for ch in prompt:
        if ch in input_layers:
            input_layers[ch].receive(6.0)
    for _ in range(3):
        step_all()

    result = []
    prev_tok = None
    repeat_count = 0

    for i in range(max_len):
        # 출력층 초기화
        for l in output_layers.values():
            l.signal = 0.0

        # 첫 스텝에는 input+state, 이후는 state만 사용
        use_input = i == 0
        fire_all(use_input=use_input, use_state=True)

        top3 = get_top_outputs(3)
        top3_str = ", ".join([f"{tok}({s:.2f})" for tok, s in top3])
        chosen_tok, strength = top3[0]

        # 디버그
        input_total = sum(l.signal for l in input_layers.values())
        state_total = sum(l.signal for l in state_layers.values())
        if verbose:
            print(
                f"  {i+1}. '{chosen_tok}' ({strength:.2f}) [{top3_str}] I:{input_total:.2f} S:{state_total:.2f}"
            )

        # 생성 시작 후 입력층 신호 제거 -> 순서 보장
        if use_input:
            for l in input_layers.values():
                l.signal = 0.0

        # 종료 토큰 발견
        if chosen_tok == END_TOKEN:
            if verbose:
                print("  → 완료 (END_TOKEN)")
            break

        # 약한 출력이면 중단
        if strength < 0.05:
            if verbose:
                print("  → 약한 신호, 중단")
            break

        # 반복 방지 (같은 토큰 3번 연속이면 중단)
        if chosen_tok == prev_tok:
            repeat_count += 1
            if repeat_count >= 2:
                if verbose:
                    print("  → 반복 감지, 중단")
                break
        else:
            repeat_count = 0

        # 결과 추가
        result.append(chosen_tok)

        # 선택한 토큰을 상태로 활성화 (다음 예측에 사용)
        if chosen_tok in state_layers:
            state_layers[chosen_tok].receive(12.0)

        prev_tok = chosen_tok

        # 시간 진행
        for _ in range(3):
            step_all()

    out = "".join(result)
    if verbose:
        print(f"\n✓ 생성 결과: '{out}'")
    return out


# ---------------------------
# 상태 출력
# ---------------------------
def show_state():
    print("\n=== 상태 요약 ===")
    print("Input (상위 8):")
    inputs = sorted(
        [(c, l.signal) for c, l in input_layers.items() if l.signal > 0.001],
        key=lambda x: x[1],
        reverse=True,
    )[:8]
    for c, s in inputs:
        print(f"  {repr(c)}: {s:.3f}")

    print("State (상위 8):")
    states = sorted(
        [(c, l.signal) for c, l in state_layers.items() if l.signal > 0.001],
        key=lambda x: x[1],
        reverse=True,
    )[:8]
    for c, s in states:
        print(f"  {repr(c)}: {s:.3f}")

    print("Top Output (상위 8):")
    tops = get_top_outputs(8)
    for c, s in tops:
        print(f"  {repr(c)}: {s:.3f}")


# ---------------------------
# CLI
# ---------------------------
def repl():
    print("AlphaIntelligence v3.0")
    print(
        "Commands: learn <Q> <A>   |   gen <prompt>   |   state   |   reset   |   exit"
    )
    print()
    while True:
        try:
            cmd = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            break
        if not cmd:
            continue
        if cmd == "exit":
            break
        elif cmd == "reset":
            reset()
            print("✓ reset")
        elif cmd == "state":
            show_state()
        elif cmd.startswith("learn "):
            parts = cmd.split(maxsplit=2)
            if len(parts) == 3:
                Q, A = parts[1], parts[2]
                learn(Q, A, verbose=True)
            else:
                print("사용법: learn <Q> <A>")
        elif cmd.startswith("gen "):
            parts = cmd.split(maxsplit=1)
            if len(parts) == 2:
                _ = generate(parts[1], verbose=True)
            else:
                print("사용법: gen <prompt>")
        else:
            print("?")


if __name__ == "__main__":
    repl()
