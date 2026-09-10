import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16).cuda()
prompt = "Hello world, this is"
inputs = tok(prompt, return_tensors="pt").to("cuda")
start = time.time()
out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(f"HF { (time.time()-start)*1000:.0f}ms", tok.decode(out[0]))
