from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")

image_path = "object.png"
try:
    pil_image = Image.open(image_path)
except FileNotFoundError:
    print("Error: Image not found.")
    exit()

inputs = processor(pil_image, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
description = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Generated description: {description}")
