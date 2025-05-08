import urllib.request
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename} successfully!")

# Create face_detector directory if it doesn't exist
if not os.path.exists("face_detector"):
    os.makedirs("face_detector")

# Download face detection model files
deploy_prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

download_file(deploy_prototxt_url, "face_detector/deploy.prototxt")
download_file(caffemodel_url, "face_detector/res10_300x300_ssd_iter_140000.caffemodel")

# Download mask detection model
mask_model_url = "https://github.com/chandrikadeb7/Face-Mask-Detection/raw/master/mask_detector.model"
download_file(mask_model_url, "mask_detector.model")

print("\nAll files downloaded successfully!") 