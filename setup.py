from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor

model_id = "microsoft/Phi-3.5-vision-instruct"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=4
)

# Test input: an image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Shaki_waterfall.jpg/800px-Shaki_waterfall.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Generate a prompt
messages = [
    {"role": "user", "content": "<|image_1|>\nDescribe the scene in detail."}
]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

# Inference
generation_args = {"max_new_tokens": 100, "temperature": 0.7, "do_sample": True}
outputs = model.generate(**inputs)
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print("Model Response:", response)
