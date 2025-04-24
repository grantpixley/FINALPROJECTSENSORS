import cv2
import os
import time
import subprocess
from skimage.measure import compare_ssim as ssim

#Function to restart the Jetson's camera daemon
# Prevents "Failed to create CaptureSession" errors by resetting the nvargus-daemon.
def restart_nvargus_daemon():
    try:
        print("Restarting nvargus-daemon...")
        subprocess.run(["sudo", "systemctl", "restart", "nvargus-daemon"], check=True)
        time.sleep(2)  # Give the daemon a moment to fully restart before continuing
    except subprocess.CalledProcessError as e:
        print("Failed to restart nvargus-daemon:", e)
        exit(1)

#GStreamer pipeline for accessing the Pi Camera v2 on Jetson Nano
# This function builds the pipeline string used by OpenCV to interface with the camera.
def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=3280,
    display_height=2464,
    framerate=21,
    flip_method=0
):
    return (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

#Compare a captured image to a reference using SSIM
# This checks for visual differences between the live image and a saved reference image.
def compare_with_reference(current_img, reference_path="reference.png", threshold=0.75):
    if not os.path.exists(reference_path):
        print(f"Reference image not found at {reference_path}")
        return

    # Load the reference image
    reference_img = cv2.imread(reference_path)
    if reference_img is None:
        print("Could not load reference image.")
        return

    # Resize reference to match the current image dimensions (in case they differ)
    reference_img = cv2.resize(reference_img, (current_img.shape[1], current_img.shape[0]))

    # Convert both images to grayscale for SSIM comparison
    gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

    # Apply slight Gaussian blur to reduce noise effects
    gray_current = cv2.GaussianBlur(gray_current, (3, 3), 0)
    gray_ref = cv2.GaussianBlur(gray_ref, (3, 3), 0)

    # Compute SSIM (Structural Similarity Index) between the current and reference images
    score, diff = ssim(gray_current, gray_ref, full=True)
    print(f"SSIM Score: {score:.4f}")

    # If SSIM score is below the threshold, potential damage is detected
    if score < threshold:
        print("Difference detected — possible rope damage.")
    else:
        print("No significant difference from reference.")

    # Visualize the difference map to show areas of change
    diff = (diff * 255).astype("uint8")
    cv2.imshow("Difference Map", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Save the current image as the reference
# Only needed once to establish the baseline image for comparison.
def save_reference_image(image, path="reference.png"):
    cv2.imwrite(path, image)
    print(f"Saved reference image to {path}")

#Main program logic

# Step 1: Restart the camera daemon to ensure a clean start
restart_nvargus_daemon()

# Step 2: Open the camera using the GStreamer pipeline
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Step 3: Capture one frame from the camera
ret, frame = cap.read()
cap.release()  # Always release the camera when done

if not ret:
    print("Error: Could not read frame.")
    exit()

# Step 4: Crop 350 pixels from the right side of the frame (customized for your setup)
cropped_frame = frame[:, :-350]

# Step 5: Resize the cropped frame to 1920x1080 for easier display/comparison
resized_frame = cv2.resize(cropped_frame, (1920, 1080))

# Step 6: Compare the captured frame with the reference image for defect detection
compare_with_reference(resized_frame)

# Step 7: Display the processed image
cv2.imshow("Right-Side Cropped Image (1920x1080)", resized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

