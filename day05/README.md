# Day5 - AdamW vs SGD 10 steps

## 🇰🇷 한국어
- Param 768, batch [2,768] - Day3 DataLoader
- 실험1 lr=0.01: SGD 3.5→132→5024→897조 발산(overflow), AdamW 3.5→0.0022 수렴
  → lr 크면 SGD 터짐, AdamW adaptive로 안정적
- 실험2 lr=0.001 Fixed: SGD 0.4531→0.0000 4 steps 수렴, AdamW 0.4531→0.0010→0.0288 진동
  → 간단한 768→1에선 SGD 더 빠름, AdamW는 m+v 관성 + decay 0.01로 0 근처에서 규제
- CUDA: 64.01MB (model+grad+m+v+context), AdamW는 SGD보다 m+v로 2배 상태 저장
- 의미: Toy에선 SGD 승리, Transformer 590K에선 AdamW 승리 - 200 steps 67초는 AdamW 덕분
- Meta 면접: "Adam vs AdamW 차이? weight_decay 분리"

## 🇺🇸 English
- Param 768, batch [2,768] from Day3
- Exp1 lr=0.01: SGD 3.5→897T divergence, AdamW 3.5→0.0022 convergence
  → SGD explodes with large lr, AdamW adaptive stable
- Exp2 lr=0.001 Fixed: SGD 0.4531→0.0000 in 4 steps, AdamW 0.4531→0.0010→0.0288 oscillating
  → SGD faster on simple 768→1, AdamW has momentum + decay 0.01 regularizing near 0
- CUDA: 64.01MB (model+grad+m+v+context), AdamW stores 2x states (m+v)
- Insight: SGD wins toy, AdamW wins Transformer 590K - enables 200 steps 67s
- Meta interview: "Adam vs AdamW? weight_decay decoupled"

## Log Exp1 lr=0.01 (divergence proves need for adaptive)~

## Log Exp2 lr=0.001 Fixed (convergence)


