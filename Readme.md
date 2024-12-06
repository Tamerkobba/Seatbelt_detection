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


## License
MIT License

Copyright (c) 2024 Tamer Kobba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
