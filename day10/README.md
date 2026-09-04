# Day10 - Vocab 32K Fix Attempt 10.52->9.59 plateau

## 🇰🇷 한국어
- Param 91.68M 동일, lr 1e-3, accum 4, eff batch 8, steps 500
- 결과: 10.5209->9.5937 12.5s 660MB Peak 1390MB 평지
- 실패 원인: vocab 32000에서 y=x copy는 각 토큰 평균 0.6번 등장, 희귀 토큰이라 500 steps로 외울 수 없음
- Chapter 2 Sec 2.2 Fig 2.5 frequency, Chapter 5 Sec 5.2 Fig 5.6 rare tokens plateau 증적
- 의미: 큰 vocab은 1T 토큰 필요, 20문장 copy 과제는 부적절 - 다음 Day11은 vocab 8000으로 낮추거나 반복 데이터로 변경 필요
- wandb: https://wandb.ai/lightel-test/ai-1year/runs/2e04vkmc loss 평지 그래프

## 🇺🇸 English
- Param 91.68M same, lr 1e-3 accum 4 steps 500
- Result: 10.5209->9.5937 12.5s 660MB Peak 1390MB plateau
- Cause: vocab 32000 y=x copy each token avg 0.6 times, rare tokens can't memorize in 500 steps
- Ch2 Sec2.2 Fig2.5 frequency, Ch5 Sec5.2 Fig5.6 rare plateau
- Insight: large vocab needs 1T tokens, 20 sentences copy is wrong task - Day11 vocab 8000 or repeated data

## Log
1 | 10.5209 | 480MB | 826MB
500 | 9.5937 | 660MB | 1390MB Peak
GPU: 5060 Ti 16GB 104MiB idle
Total 12.5s

