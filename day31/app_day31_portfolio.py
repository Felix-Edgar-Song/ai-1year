print("=== Day31 2-Month Final Portfolio ===")
table = [
    ("Day24 baseline", "902MiB / 17 replicas / P8 5W / 375M 757MB"),
    ("Day25 vLLM", "608MiB / 10 replicas / EngineCore 360MiB overhead"),
    ("Day26 torch.compile 580", "580.8MiB / 844.3MiB Peak / 24L 1024dim"),
    ("Day27 233MiB", "233.4MiB / 265.5MiB Peak / 12L 768dim / smashed 400MiB"),
    ("Day28 SGLang", "FAIL CUDA 13.0 vs 12.9 mismatch / sm_120 not supported / torch.compile final"),
    ("Day29 50 prod", "233.4MiB / 445.5MiB Peak / 12400MiB=12.1GiB<16GiB OK / Least VRAM LB"),
    ("Day30 Final", "233.4MiB / 265.5MiB Peak / 12422MiB=12.1GiB<16GiB OK / 74% cut 17→50 3x / 2-month best"),
]
for k,v in table:
    print(f"{k:20s} | {v}")

print("\n--- ai-1year/README.md에 붙일 비교표 (markdown) ---")
print("| Day | Alloc | Peak | Replicas | Note |")
print("|---|---|---|---|---|")
print("| Day24 | 902MiB | - | 17 | baseline 375M |")
print("| Day25 vLLM | 608MiB | - | 10 | EngineCore 360MiB overhead |")
print("| Day27 | 233.4MiB | 265.5MiB | 50 | 74% cut, smashed 400MiB |")
print("| Day30 Final | 233.4MiB | 265.5MiB | 50 | 12.1GiB<16GiB OK |")

print("\n--- Meta E5 3문장 ---")
print("1. 50개: 233MiB x50=11.6GiB+15MiB overhead=12.1GiB<16GiB, Least VRAM LB 8000-8049 /vram health")
print("2. int8: 124M 스케일 테이블로 232→515MiB 증가, 7B+만 효과 - Day27 로그 증명")
print("3. torch.compile: SGLang flashinfer sm_120 미지원 + vLLM 360MiB 고정 10개만, 233MiB Peak 265MiB 유일 <400MiB +50개")
