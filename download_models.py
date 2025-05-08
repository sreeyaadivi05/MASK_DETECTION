import urllib.request
import os
import hashlib

def calculate_file_hash(filename):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url, filename, expected_size=None):
    """Download a file with progress tracking and size verification."""
    print(f"Downloading {filename}...")
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    try:
        # Download with progress tracking
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            file_size = int(response.info().get('Content-Length', 0))
            block_size = 8192
            downloaded = 0
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                
                # Calculate progress
                progress = (downloaded / file_size) * 100 if file_size > 0 else 0
                print(f"\rProgress: {progress:.1f}%", end='')
            
            print("\nDownload completed!")
            
            # Verify file size if expected_size is provided
            if expected_size and os.path.getsize(filename) != expected_size:
                print(f"Warning: File size mismatch. Expected {expected_size} bytes, got {os.path.getsize(filename)} bytes")
                return False
            return True
            
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def main():
    # URLs for the model files
    deploy_prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    mask_model_url = "https://github.com/chandrikadeb7/Face-Mask-Detection/raw/master/mask_detector.model"

    # Expected file sizes (in bytes)
    expected_sizes = {
        "face_detector/deploy.prototxt": 27753,
        "face_detector/res10_300x300_ssd_iter_140000.caffemodel": 19926000,
        "mask_detector.model": 11490448
    }

    # Download files
    success = True
    success &= download_file(deploy_prototxt_url, "face_detector/deploy.prototxt", expected_sizes["face_detector/deploy.prototxt"])
    success &= download_file(caffemodel_url, "face_detector/res10_300x300_ssd_iter_140000.caffemodel", expected_sizes["face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
    success &= download_file(mask_model_url, "mask_detector.model", expected_sizes["mask_detector.model"])

    if success:
        print("\nAll files downloaded successfully!")
    else:
        print("\nSome files failed to download. Please try running the script again.")

if __name__ == "__main__":
    main() 