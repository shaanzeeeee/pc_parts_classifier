import os
import random
import torch
from torchvision import transforms
from PIL import Image
from models import get_efficientnet_b0
from config import DEVICE

def main():
    test_dir = os.path.join('data', 'processed', 'test')
    if not os.path.exists(test_dir):
        print("Test directory not found.")
        return
        
    CLASSES = sorted(os.listdir(test_dir))
    MODEL_NAME = "efficientnet_b0"
    MODEL_WEIGHTS = "best_model_efficientnetb0.pth"
    
    print(f"\n--- Initializing High-Performance {MODEL_NAME} ---")
    model = get_efficientnet_b0(num_classes=len(CLASSES), feature_extract=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_images = []
    for cls_name in CLASSES:
        cls_path = os.path.join(test_dir, cls_name)
        if os.path.exists(cls_path):
            for file_name in os.listdir(cls_path):
                all_images.append((os.path.join(cls_path, file_name), cls_name))

    if not all_images:
        print("No images found in test directory.")
        return

    # Select 1 random image
    example = random.choice(all_images)
    img_path, actual_class = example

    print(f"\nTargeting raw file: {img_path}")
    print("Executing Diagnostic Neural Pipeline...\n")

    with torch.no_grad():
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        confidence, predicted_idx = torch.max(probs, 1)
        predicted_class = CLASSES[predicted_idx.item()]
        conf_val = confidence.item()
        
        if conf_val < 0.40:
            predicted_class = "Uncertain"
            
        print("======== RESULTS ========")
        print(f"Algorithm Prediction  : {predicted_class}")
        print(f"Network Confidence    : {conf_val*100:.2f}%")
        print(f"Actual Baseline Truth : {actual_class}")
        print("=========================\n")

if __name__ == "__main__":
    main()
