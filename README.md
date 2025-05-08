# Face Mask Detection System

This project implements a real-time face mask detection system using Python, OpenCV, and TensorFlow. The system can detect faces in a video stream and determine whether the person is wearing a mask or not.

## Prerequisites

- Python 3.7 or higher
- Webcam or video input device

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the required model files:
   - Create a directory named `face_detector`
   - Download the face detection model files:
     - [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
     - [res10_300x300_ssd_iter_140000.caffemodel](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
   - Place these files in the `face_detector` directory

4. Download the mask detection model:
   - Download the pre-trained mask detection model from [here](https://github.com/chandrikadeb7/Face-Mask-Detection/raw/master/mask_detector.model)
   - Place it in the root directory of the project

## Usage

1. Run the mask detection script:
```bash
python mask_detection.py
```

2. The program will open your webcam and start detecting faces and masks in real-time.
3. Press 'q' to quit the program.

## Features

- Real-time face detection
- Mask/no-mask classification
- Visual feedback with bounding boxes and labels
- Confidence score display

## Notes

- The system uses a pre-trained model for mask detection
- The face detection is performed using OpenCV's DNN face detector
- The mask detection model is based on MobileNetV2 architecture
- Green boxes indicate detected masks, red boxes indicate no mask 