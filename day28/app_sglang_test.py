import torch
print(f"CUDA: {torch.cuda.is_available()}, SM: {torch.cuda.get_device_capability()}")
# SGLang 233MiB 모델 로드 시도 - Blackwell 호환 체크
try:
    import sglang as sgl
    print("SGLang import OK - Blackwell sm_120 test")
except Exception as e:
    print(f"SGLang fail: {e} -> torch.compile 233MiB가 Blackwell 최종 정답")
