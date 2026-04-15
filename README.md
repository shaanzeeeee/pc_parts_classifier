# Hardware-Vision: Multi-Class CNN Component Recognition

## Overview
Hardware-Vision is an end-to-end computer vision pipeline designed to recognize and diagnose 11 distinct PC hardware components and assembly statuses (e.g., Motherboard, Graphics Card, Good/Bad Cable Management). 

The pipeline utilizes robust web crawling, strict MD5-Hash deduplication, confidence-threshold gating (0.40) to weed out subjective "Uncertain" elements, and runs highly scaled epochs across diverse Nvidia GPU-accelerated PyTorch architectures.

---

## Dataset & Preprocessing
To maintain maximum data integrity while minimizing repetitive processing overhead:
1. **Diverse Search Scraping:** Over 120 unique, widely varied search parameters were utilized across Bing to build a highly contextual dataset of ~5,000 localized images spanning 11 strict subclasses.
2. **One-Time Data Cleaning:** The raw images were completely filtered for duplicates via `MD5` hashing, dropping corrupted assets, reshaping to standard `(224, 224)` bounds, and divided exclusively once into `train`, `val`, and `test` ratios (`80/10/10`).
3. **Universality:** This identical, scrubbed dataset pipeline serves as the ultimate baseline upon which all three subsequent neural network architectures are evaluated against to ensure pure model-to-model consistency.

---

## Model Training & Evaluation
We iteratively trained and evaluated three distinct Transfer Learning architectures on the exact same pristine dataset, iterating for 15 epochs on the GPU.

### Model Performance Comparison

| Model Architecture | Parameter Size | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Overall Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | Heavy (~25M) | 0.80 | 0.73 | 0.73 | **80%** |
| **MobileNetV2** | Lightweight (~3M) | 0.76 | 0.69 | 0.68 | **75%** |
| **EfficientNet-B0** | Balanced (~5M) | **0.83** | **0.77** | **0.76** | **83% (Best)** |

### 1. ResNet-50 
- **Description:** A massively deep 50-layer Residual Network employing skip connections (residual blocks) to bypass vanishing gradients.
- **Results:** Achieved a highly robust baseline of **~80% Accuracy**. It particularly excelled at strictly dense components like `Power_Supply` and `PC_Case` (hitting >90% F1 scores), but struggled slightly classifying thin edge features within `Good_Cable_Management`.

### 2. MobileNetV2
- **Description:** A highly optimized, significantly lighter architecture built utilizing depthwise separable convolutions and inverted residuals. Designed predominantly for mobile integration.
- **Results:** Hit precisely average bounds around **~70-75% Accuracy**. While its inference execution speed was markedly the fastest among the selection, it lost fine-grain resolution capacity leading to higher false-positives across similar component classes (such as confusing large rigid `Air_Coolers` dynamically with `AIO_Liquid_Cooler` blocks).

### 3. EfficientNet-B0 (Winner - Best Performance)
- **Description:** A modern architecture employing Compound Scaling—mathematically balancing the network's depth, width, and resolution seamlessly rather than haphazardly deepening convolutional blocks.
- **Results:** Displayed the absolute peak performance bounds (**~83%+ Accuracy**). It consistently outperformed ResNet-50 when diagnosing `Bad_Cable_Management`, largely because its structural compound-scaling design inherently captures much more complex, high-resolution topographical details across the pipeline without over-fitting to the background case geometry.

---

## Conclusion & Architectural Reasoning
**EfficientNet-B0 is definitively the highest-performing architecture for this diagnostic pipeline.** 

While **ResNet-50** carries immense historical precedent, its heavy residual blocks actually promote extreme feature homogenization across highly chaotic visual structures (such as internal motherboard wiring vs heat sinks). Because diagnostic PC component recognition relies heavily on granular topological clarity rather than just simple macro-structural outlines, **EfficientNet-B0** perfectly captured those localized chaotic variance spikes through its superior width/resolution feature mapping without succumbing to the accuracy degradation suffered by MobileNet.
