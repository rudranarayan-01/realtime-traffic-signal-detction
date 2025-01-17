# Traffic Sign Detection Project

This project implements a real-time traffic sign detection system using Python, OpenCV, and TensorFlow. It uses a convolutional neural network (CNN) to classify traffic signs and display predictions in real-time through a webcam.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

The goal of this project is to detect and classify traffic signs in real-time. The system is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains 43 classes of traffic signs.

### Key Features
- Preprocessing of traffic sign images.
- Training a CNN for traffic sign classification.
- Real-time traffic sign detection using a webcam.

## Dataset

The project uses the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/) dataset. Download and extract the dataset into the `traffic_sign_dataset/` directory. Ensure the dataset is structured as follows:

```
traffic_sign_dataset/
    0/
    1/
    2/
    ...
    42/
```

## Requirements

Install the following Python libraries:

- OpenCV
- TensorFlow
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install opencv-python opencv-python-headless tensorflow numpy matplotlib
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rudranarayan-01/traffic-sign-detection.git
   cd traffic-sign-detection
   ```

2. Prepare the dataset:
   - Download the GTSRB dataset.
   - Extract it into the `traffic_sign_dataset/` directory.


## Usage

### Training the Model

Run the training script to train the CNN model on the dataset:

```bash
Traffic_Sign_Recognition.ipynb
```

The trained model will be saved as `traffic_sign_model.h5`.

### Real-Time Detection

Use the real-time detection script to detect traffic signs using a webcam:

```bash
python test.py --source 0
```

Press `q` to exit the real-time detection window.

## Model Architecture

The CNN model consists of:

- **Conv2D Layers**: Extract spatial features from images.
- **MaxPooling2D Layers**: Reduce spatial dimensions.
- **Dense Layers**: Perform classification.
- **Dropout Layers**: Prevent overfitting.

The model uses the Adam optimizer and categorical cross-entropy loss.

## Results

- **Accuracy**: Achieved an accuracy of ~95% on the test set.
- **Real-Time Performance**: Processes webcam frames efficiently.

## Future Improvements

- Implement data augmentation to improve generalization.
- Use a pre-trained model like MobileNet for higher accuracy.
- Extend the system to detect multiple traffic signs in a single frame.

---

Feel free to contribute to this project by submitting issues or pull requests. Happy coding!
(https://github.com/rudranarayan-01)
