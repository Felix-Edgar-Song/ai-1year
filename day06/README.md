# Day6 - 200 steps reproduction Phase1 Complete

## 🇰🇷 한국어
- Param 0.66M (128 hidden, 4 layers, vocab 500) - Day2 0.59M 맞춤
- 고정 데이터 y=x 복사 과제 20개 반복 - 랜덤 y 버그 수정 (ln(500)=6.21 상한)
- 결과: loss 6.3851→0.0071 200 steps 0.9s CUDA 68.1MB Peak 73.2MB
- 67s vs 0.9s 차이: hidden 768 vs 128, vocab 32000 vs 500, seq 512 vs 32
- 5대 요소 합체: bf16 50% + 0.59M + DataLoader + backward + AdamW
- 의미: Phase1 클리어 - Llama 학습 루프 200 steps 완주, 0.0999 목표 14배 초과

## 🇺🇸 English
- Param 0.66M (128 hidden, 4 layers, vocab 500) - Day2 0.59M target
- Fixed dataset y=x copy task 20 samples repeat - fix random y bug (ln(500)=6.21 bound)
- Result: loss 6.3851→0.0071 200 steps 0.9s CUDA 68.1MB Peak 73.2MB
- 67s vs 0.9s diff: hidden 768 vs 128, vocab 32k vs 500, seq 512 vs 32
- 5 elements combined: bf16 50% + 0.59M + DataLoader + backward + AdamW
- Insight: Phase1 complete - Llama training loop 200 steps, 14x better than 0.0999 target

## Log
