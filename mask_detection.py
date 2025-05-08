import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Grab the dimensions of the frame and construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Only make a predictions if at least one face was detected
    if len(faces) > 0:
        # For faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # Return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def main():
    # Check if required files exist
    required_files = [
        "face_detector/deploy.prototxt",
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        "mask_detector.model"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file '{file}' not found!")
            print("Please run download_models.py first to download the required model files.")
            return

    print("Loading face detector model...")
    # Load face detector model
    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    try:
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        print("Face detector model loaded successfully!")
    except Exception as e:
        print(f"Error loading face detector model: {str(e)}")
        return

    print("Loading mask detector model...")
    # Load the face mask detector model
    try:
        maskNet = load_model("mask_detector.model")
        print("Mask detector model loaded successfully!")
    except Exception as e:
        print(f"Error loading mask detector model: {str(e)}")
        return

    print("Initializing video capture...")
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device!")
        return

    print("Starting video stream... Press 'q' to quit.")
    while True:
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video capture device!")
            break

        # Detect faces and predict masks
        try:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            continue

        # Loop over the detected face locations and their corresponding predictions
        for (box, pred) in zip(locs, preds):
            # Unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # Determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Show the output frame
        cv2.imshow("Frame", frame)
        
        # Check for window close button
        if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
            break
            
        # Check for 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Q key pressed, exiting...")
            break

    print("Cleaning up...")
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()