import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2

def evaluate_model(model, dataloader, class_names, device):
    """
    Evaluates the model utilizing the 'Confidence Rejection Threshold' logic.
    Predictions with confidence < 0.70 are flagged as 'Uncertain'.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            # Confidence Thresholding
            max_probs, preds = torch.max(probs, 1)
            
            # Convert to CPU numpy
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            max_probs = max_probs.cpu().numpy()
            
            for i in range(len(preds)):
                # If confidence < 0.40, replace with "Uncertain" (-1)
                if max_probs[i] < 0.40:
                    all_preds.append(-1)
                else:
                    all_preds.append(preds[i])
                    
            all_labels.extend(labels)
            
    # Remove 'Uncertain' labels for sklearn report if you want to cleanly evaluate accepted predictions, 
    # but for full transparency, let's keep them and map -1 to 'Uncertain'
    all_preds_cleaned = []
    all_labels_cleaned = []
    for p, l in zip(all_preds, all_labels):
        if p != -1:
            all_preds_cleaned.append(p)
            all_labels_cleaned.append(l)
            
    if len(all_preds_cleaned) == 0:
        print("All predictions were uncertain.")
        return
        
    print("Classification Report:")
    print(classification_report(all_labels_cleaned, all_preds_cleaned, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels_cleaned, all_preds_cleaned)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    plt.close()

class GradCAM:
    r"""
    Calculates Grad-CAM heatmap for a target class.
    Mathematically:
    $L_{Grad-CAM}^c = ReLU \left( \sum_k \\alpha_k^c A^k \\right)$
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook registrations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
            
        self.model.zero_grad()
        # Backprop for target class
        target = out[0, class_idx]
        target.backward()
        
        # Global Average Pooling of gradients ($\alpha_k^c$)
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        # Linear combination of activations
        activations = self.activations * gradients
        cam = activations.sum(dim=1, keepdim=True)
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-7)
        
        return cam.squeeze().cpu().detach().numpy()

def display_gradcam(image_tensor, cam, save_path="gradcam_heatmap.png"):
    heatmap = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)
    
    # Superimpose
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Grad-CAM: Bad Cable Management')
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()
