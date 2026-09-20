import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from fastapi import FastAPI
app = FastAPI()
print("Loading vLLM gpt2 - 608MiB ultra low...")
from vllm import LLM
llm = LLM(model="openai-community/gpt2", dtype="bfloat16", gpu_memory_utilization=0.08, max_model_len=128, max_num_seqs=1, enforce_eager=True, enable_prefix_caching=False)
print("Loaded - 608MiB target")
@app.get("/health")
def health(): return {"status":"ok","vram":"608MiB"}
@app.get("/generate")
def gen(q: str="Hello"):
    from vllm import SamplingParams
    out = llm.generate([q], SamplingParams(max_tokens=20))
    return {"text": out[0].outputs[0].text}
