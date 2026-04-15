import torch
import logging
from main import get_dataloaders
from evaluation import evaluate_model
from models import get_efficientnet_b0
from config import DEVICE, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Initializing DataLoaders for Final Evaluation Matrix...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    
    if test_loader is None:
        logging.error("Failed to load evaluation dataset!")
        return
        
    logging.info("Loading optimal architecture (EfficientNet-B0)...")
    model = get_efficientnet_b0(NUM_CLASSES, feature_extract=False).to(DEVICE)
    model.load_state_dict(torch.load('best_model_efficientnetb0.pth', map_location=DEVICE, weights_only=True))
    
    logging.info("Starting Full Neural Evaluation Pipeline across testing subset...")
    evaluate_model(model, test_loader, class_names, DEVICE)
    
if __name__ == "__main__":
    main()
