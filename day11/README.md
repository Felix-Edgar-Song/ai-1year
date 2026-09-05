# Day11 - Vocab 8K Next-Token 9.14->0.046 plateau escape

## 🇰🇷 한국어
- Param 91.68M->54.82M vocab 32K->8K 75% cut embed 24.5M->6.1M
- 결과: loss 9.1495->0.0463 500 steps 6.6s CUDA 404MB Peak 788MB
- 탈출 원인: copy 0.6번 등장 -> next-token 2.5번 + 문맥 반복 20번, frequency up
- Chapter 2 Sec 2.3 Fig 2.10 vocab memory linear, Sec 2.2 Fig 2.5 frequency
- Chapter 4 Sec 4.4 Fig 4.3 next-token prediction, Chapter 5 Sec 5.1 Fig 5.2 pretraining curve 9->0.1
- 의미: 큰 vocab rare token 평지는 next-token + 반복 + 작은 vocab으로 해결, 1390MB->788MB -600MB
- wandb: https://wandb.ai/lightel-test/ai-1year/runs/y16el4nz loss █▁▁▁ 완벽 낙하

## 🇺🇸 English
- Param 91.68M->54.82M vocab 32K->8K 75% cut
- Result: loss 9.1495->0.0463 500 steps 6.6s CUDA 404MB Peak 788MB
- Escape: copy 0.6 freq -> next-token 2.5 freq + context repeat 20x
- Ch2 Sec2.3 Fig2.10 vocab memory, Ch2 Sec2.2 Fig2.5 frequency, Ch4 Sec4.4 Fig4.3 next-token, Ch5 Sec5.1 Fig5.2
- Insight: rare token plateau solved by next-token + repeat + small vocab, 1390MB->788MB
- wandb: loss drop 9.14->0.15 in 50 steps 0.8s

## Log
```
Param: 54.82M - Day10 91.68M -> Day11 54.7M vocab 75% cut
1 | 9.1495 | 0.2s | 294MB | 428MB
50 | 0.1505 | 0.8s | 511MB | 788MB
500 | 0.0463 | 6.6s | 404MB | 788MB
Final 0.0463 | Total 6.6s | GPU 5060 Ti 104MiB idle
```
