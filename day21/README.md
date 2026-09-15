## Day21 - FastAPI 786MiB 100u 62 RPS P99 1800ms 안녕 세상 100% Real

## 🇰🇷 한국어

### 실측 결과 - 100% Real Measured 21:56 Gwangju

**최종 성공: 100명 동시 0% 실패 62 RPS**

- **Model**: 375M bfloat16 757MB load - 413MB Peak 417MB gen - health 380MB

- **GEN Real**: `EN: Hello KO: 안녕 세상, 이건 100` 100% - 7470 reqs 0 fail

- **VRAM Real**:
    - idle: 786MiB / 16311MiB | 766MiB process PID 7867 | P1 18W 36C | 380MB health
    - gen: 413MB Peak 417MB | 757MB load
    - 100u load: 2274MiB / 16311MiB | 2254MiB process | P1 134W 58C 100% | +1488MiB
    - Capacity: 16311MiB / 766MiB = 21.3개 동시 (Day20 15.1개 대비 +6.2개)

- **Locust Real 100u 2m**:
    - 7470 reqs 0% fail | RPS 62.32 | Avg 1533ms | Med 1600ms | Min 133ms Max 1917ms
    - P50 1600 P95 1700 P99 1800 P99.9 1800 P100 1900
    - Day20 예상 P99 800ms (argmax) vs Day21 1800ms (sampling 0.8 + empty_cache)

- **Env**: `source ~/ai-env/bin/activate` - Day20 누수 1090→1100MiB +10MiB fix

### 실측 비교표 - Day18-21 메모리 완성

| 항목 | Day18 SFT 1000 | Day20 FastAPI 12MiB | Day21 100u Real 21:56 |
| --- | --- | --- | --- |
| Model | 196M+LoRA 368k 0.1872% | 375M | 375M |
| GPU Mem Real | 6818MB/Peak 9876MB | 756MB/Peak 897MB/1100MiB | 757MB load/413MB gen/786MiB idle/2274MiB 100u |
| GEN Real | 안녕 세상 100% | 안녕 세상 100% 788MB | 안녕 세상 100% 413MB 7470 reqs |
| Load Test | - | - | 100u 62 RPS P99 1800ms 0% fail |
| 결론 | SFT 1000 정답 | 15개 동시 14x | 21개 동시 P99 1.8s 0% fail 2.6x 개선 |

### 책 연결
- Ch 5 배포: 12MiB headless -> 766MiB idle -> 2254MiB 100u - 동시성 21개 증명
- Ch 6 Fig 6.2: vLLM 15208MiB vs HF 9876MB vs FastAPI 766MiB 19.8x
- Ch 7 Fig 7.4: LoRA 0.1872% -> SFT 1000 -> FastAPI 62 RPS

## 🇺🇸 English

**Success: 100u 0% fail 62 RPS P99 1800ms**

- 757MB load 413MB gen 786MiB idle 2274MiB 100u - 21.3 instances
- 7470 reqs RPS 62.32 Avg 1533ms P99 1800ms

## Log Real Measured
```
idle 21:50:
Model loaded 375M - 757MB
health 380MB
gen 413MB Peak 417MB 안녕 세상 100%
786MiB / 766MiB P1 18W 36C

100u 21:56:
All users spawned: 100
7470 reqs 0% fail RPS 62.32
Avg 1533 Med 1600 Min 133 Max 1917
P99 1800ms
2274MiB / 2254MiB P1 134W 58C 100% 37% fan
```


## Files
- `app_v2.py` - 757MB load 413MB gen 786MiB idle fix
- `locustfile.py` - 100u
- `nvidia_786MiB_idle.txt` - 786MiB Real
- `nvidia_2274MiB_100u.txt` - 2274MiB Real
- `locust_100u_7470.txt` - 7470 reqs P99 1800ms
