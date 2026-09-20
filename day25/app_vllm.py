from fastapi import FastAPI
app = FastAPI()

print("Loading vLLM custom...")
from vllm import LLM
llm = LLM(
    model="/app/hf_model",
    dtype="bfloat16",
    gpu_memory_utilization=0.45,
    max_model_len=512,
    trust_remote_code=True,
    enforce_eager=True
)
print("vLLM loaded!")

@app.get("/health")
def health():
    import torch
    return {"vram": torch.cuda.memory_allocated()/1024**2, "status":"ok"}

@app.get("/generate")
def gen(prompt: str = "EN: Hello KO:"):
    from vllm import SamplingParams
    out = llm.generate([prompt], SamplingParams(max_tokens=12, temperature=0.0))
    return {"gen": out[0].outputs[0].text}
