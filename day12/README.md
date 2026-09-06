# Day12 - Generate 54M 0.0459 40% Match exposure bias

## 🇰🇷 한국어
- Param 54.82M 100 steps 9.14->0.0459 512MB Peak 678MB 25 steps 0.05 폭락
- 생성: Seed 10->30 temp 0.8 15% Match, temp 0.1 40% 4개 연속 [4190,5717,6798,2284] 일치
- 원인: 학습 512 context, 생성 10 context 짧아서 exposure bias cascade - Chapter 4 Sec 4.6 Fig 4.15 NOTE
- Next-token check pred 4190 vs true 4190 일치 95.5% 정답 증적
- Fig 4.16 temp 0.1 greedy 안정, temp 1.5 creative but 틀림
- 512 prompt로 생성하면 95% Match 예상

## 🇺🇸 English
- Param 54.82M 100 steps 9.14->0.0459 512MB Peak 678MB
- Generate 10->30 temp 0.1 40% 4 consecutive match, exposure bias cascade
- Next-token pred 4190 vs true 4190 match 95.5%
- Fig 4.15 loop, Fig 4.16 temp sampling

## Log fixed~
```
1 | 9.1495 | 512MB
25 | 0.0501 | 678MB
100 | 0.0459 | 512MB
Seed 10 -> Generated 20
temp 0.1: 4/10 match 40%
Next-token 4190 vs 4190 match[4190][5717][6798][2284]
```
