import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, MEAN, STD, DEVICE, NUM_CLASSES
from models import get_resnet50, get_mobilenet_v2, get_efficientnet_b0
from evaluation import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_dataloaders():
    """
    Sets up transforms and DataLoaders for PyTorch.
    $x_{norm} = \\frac{x - \\mu}{\\sigma}$
    """
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    
    try:
        train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
        val_data = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
        test_data = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, test_loader, train_data.classes
    except BaseException as e:
        logging.error(f"Failed to load datasets. Ensure preprocessing script has been run: {e}")
        return None, None, None, None

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=EPOCHS):
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct.double() / len(val_loader.dataset)
        
        logging.info(f'Epoch {epoch}/{num_epochs - 1} - Train Loss: {epoch_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
        
        # Scheduler Step
        scheduler.step(val_loss)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
                
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def main():
    logging.info("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    
    if train_loader is None:
        return
        
    logging.info(f"Classes Found: {class_names}")
    
    models_dict = {
        "ResNet-50": get_resnet50,
        "MobileNetV2": get_mobilenet_v2,
        "EfficientNet-B0": get_efficientnet_b0
    }
    
    for model_name, model_fn in models_dict.items():
        logging.info(f"========== Training {model_name} ==========")
        torch.cuda.empty_cache() # Clear GPU Cache
        
        logging.info(f"Initializing {model_name} on {DEVICE}...")
        model = model_fn(NUM_CLASSES).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        
        # We need to pass the model_name to save distinct weights
        best_val_loss = float('inf')
        early_stop_patience = 5
        patience_counter = 0
        model_save_path = f'best_model_{model_name.replace("-", "").lower()}.pth'
        
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == labels.data)
                    
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct.double() / len(val_loader.dataset)
            logging.info(f'{model_name} Epoch {epoch}/{EPOCHS - 1} - Train Loss: {epoch_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch}")
                    break
                    
        # Load best weights for Evaluation
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        
        logging.info(f"Evaluating {model_name} on Test Set...")
        evaluate_model(model, test_loader, class_names, DEVICE)
        
    logging.info("Multi-Model Pipeline Complete.")

if __name__ == "__main__":
    main()
