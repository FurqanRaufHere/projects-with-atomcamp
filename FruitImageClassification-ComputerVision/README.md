# Fruit Image Classification using Transfer Learning

## Overview
This project builds a multi-class image classification model to identify 10 different fruit categories using **Transfer Learning** with MobileNetV2.

The model achieves **~86% test accuracy** and includes full evaluation, error analysis, and an inference pipeline.

---

## Objective
- Classify fruit images into 10 categories.
- Apply transfer learning using a pre-trained CNN.
- Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix.
- Deploy an inference function for real-world predictions.

---

## Dataset
The dataset is divided into:
- `train/`
- `valid/`
- `test/`

Classes:
- Apple
- Avocado
- Banana
- Cherry
- Kiwi
- Mango
- Orange
- Pineapple
- Strawberries
- Watermelon

All images resized to **224×224**.

---

## Model Architecture

**Base Model:** MobileNetV2 (pre-trained on ImageNet)

### Approach:
1. Load pre-trained MobileNetV2 (without top layers)
2. Freeze convolutional base (feature extraction)
3. Add custom classification head:
   - Global Average Pooling
   - Dense layer
   - Softmax output (10 classes)
4. Fine-tune selected layers

---

## Results

- **Test Accuracy:** ~86%
- **Macro F1-score:** ~0.86
- Strong performance across most classes
- Minor confusion between visually similar fruits

Evaluation includes:
- Classification Report
- Confusion Matrix
- Training & Validation Curves
- Error Analysis