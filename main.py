# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

# Load processor and model
processor = AutoImageProcessor.from_pretrained("tangocrazyguy/resnet-50-finetuned-cats_vs_dogs")
model = AutoModelForImageClassification.from_pretrained("tangocrazyguy/resnet-50-finetuned-cats_vs_dogs")

# Load and process the image
image_path = "test_images/images1.jpeg"

# Check if image exists
if not os.path.exists(image_path):
    print(f"Fehler: Bild nicht gefunden unter {image_path}")
    exit(1)

# Load image
image = Image.open(image_path)
print(f"Bild geladen: {image_path}")
print(f"Bildgröße: {image.size}")

# Process image
inputs = processor(image, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits[0], dim=0)

# Get prediction results
predicted_class_id = predictions.argmax().item()
confidence = predictions[predicted_class_id].item()

# Map class ID to label (assuming 0=cat, 1=dog based on typical cats_vs_dogs models)
class_labels = {0: "Katze", 1: "Hund"}
predicted_label = class_labels.get(predicted_class_id, f"Unbekannte Klasse {predicted_class_id}")

print(f"\nVorhersage: {predicted_label}")
print(f"Vertrauen: {confidence:.2%}")
print(f"\nAlle Wahrscheinlichkeiten:")
for class_id, label in class_labels.items():
    prob = predictions[class_id].item()
    print(f"  {label}: {prob:.2%}")