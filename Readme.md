# Seatbelt Detection System

This project implements a **seatbelt detection system** using YOLO for windshield detection and cropping, and a DenseNet model to classify whether a person is **wearing a seatbelt** or **not**. The system processes video frames, detects windshields, crops them, and classifies seatbelt usage for each detected windshield.

---

## Overview

The project combines the power of object detection and image classification:
1. **YOLO**:
   - Detects and crops the windshield area in a given video or image.
2. **DenseNet**:
   - Classifies the cropped image into two categories:
     - `Wearing Seatbelt` (Class 0)
     - `Not Wearing Seatbelt` (Class 1 - Violation)

---

## Video Demos

### Demo 1
[![Demo 1](https://img.youtube.com/vi/64xSVSXQAUE/0.jpg)](https://www.youtube.com/watch?v=64xSVSXQAUE)

### Demo 2
[![Demo 2](https://img.youtube.com/vi/gEB4qA7gZG4/0.jpg)](https://www.youtube.com/watch?v=gEB4qA7gZG4)

### Demo 3
[![Demo 3](https://img.youtube.com/vi/v7ML6Is3vEI/0.jpg)](https://www.youtube.com/watch?v=v7ML6Is3vEI)

---

## TODOs

- **Dataset**:
  - Include links and descriptions of the dataset used for training and testing.

- **Models**:
  - Add details about the YOLO and DenseNet models, including architecture and pretrained weights.

- **Requirements**:
  - Provide a `requirements.txt` file with all necessary dependencies.

---

## Future Updates

- **Performance Metrics**:
  - Add accuracy, precision, recall, F1 Score, and confusion matrix.
- **Optimization**:
  - Improve model performance on edge cases (e.g., glare, occlusions).

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
