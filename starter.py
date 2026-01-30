import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/medgemma-27b-text-it"  # example model on HF Hub :contentReference[oaicite:0]{index=0}

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   # adjust if needed
    device_map="auto",
)

prompt = "Summarize the likely causes of iron deficiency anemia in adults."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(out[0], skip_special_tokens=True))