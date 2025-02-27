# Temporal-and-Histogram-difference-calculation-using-graphical-representation
Temporal gradient garph measures the difference between consecutive frames to detect sudden changes. Histogram difference compares the difference of two consecutive frames for changes in brightness distribution
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # AI-based smoothing

# Define video file paths (Updated paths)
input_video_path = r"D:\New downloads\Project Work Data\Input\Video.mp4"
smoothened_video_path = r"D:\New downloads\Project Work Data\Output\OutputSmoothedvideo.avi"
output_graph_path = r"D:\New downloads\Project Work Data\Output\Transition_Analysis.png"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_graph_path), exist_ok=True)

# Open video files
input_video = cv2.VideoCapture(input_video_path)
smoothened_video = cv2.VideoCapture(smoothened_video_path)

# Check if videos opened successfully
if not input_video.isOpened():
    print(f"❌ ERROR: Could not open input video at {input_video_path}")
    exit()
if not smoothened_video.isOpened():
    print(f"❌ ERROR: Could not open smoothened video at {smoothened_video_path}")
    exit()

print("✅ Videos loaded successfully!")

frame_count = 0
temporal_gradients = []
histogram_diffs = []
prev_gray = None
prev_hist = None

# Process frames
while True:
    ret1, frame1 = input_video.read()
    ret2, frame2 = smoothened_video.read()
    
    if not ret1 or not ret2:
        break  # Stop when either video ends
    
    # Resize for consistency
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))
    
    # Convert to grayscale for temporal gradient
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Compute temporal gradient (frame difference)
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray1)
        temporal_gradient = np.mean(diff)  # Average intensity difference
        temporal_gradients.append(temporal_gradient)
    
    prev_gray = gray1.copy()
    
    # Compute histogram difference
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])  # Histogram for brightness
    hist1 = cv2.normalize(hist1, hist1).flatten()
    
    if prev_hist is not None:
        hist_diff = cv2.compareHist(prev_hist, hist1, cv2.HISTCMP_BHATTACHARYYA)  # Bhattacharyya distance
        histogram_diffs.append(hist_diff)
    
    prev_hist = hist1.copy()
    
    frame_count += 1

# Release video files
input_video.release()
smoothened_video.release()

print(f"✅ Processed {frame_count} frames.")

# AI-Based Smoothing Using Gaussian Filter
temporal_gradients_smooth = gaussian_filter1d(temporal_gradients, sigma=2)
histogram_diffs_smooth = gaussian_filter1d(histogram_diffs, sigma=2)

# Plot Temporal Gradient and Histogram Differences
plt.figure(figsize=(12, 5))

# Temporal Gradient Plot
plt.subplot(1, 2, 1)
plt.plot(temporal_gradients, label="Raw Temporal Gradient", color='g', alpha=0.5)
plt.plot(temporal_gradients_smooth, label="Smoothed Temporal Gradient", color='b', linewidth=2)
plt.xlabel("Frame Number")
plt.ylabel("Gradient Intensity")
plt.title("Temporal Gradient: Sudden Transitions")
plt.legend()
plt.grid()

# Histogram Difference Plot
plt.subplot(1, 2, 2)
plt.plot(histogram_diffs, label="Raw Histogram Difference", color='r', alpha=0.5)
plt.plot(histogram_diffs_smooth, label="Smoothed Histogram Difference", color='b', linewidth=2)
plt.xlabel("Frame Number")
plt.ylabel("Histogram Difference")
plt.title("Histogram-Based Transition Detection")
plt.legend()
plt.grid()

# Save the plot as a PNG file
plt.tight_layout()
plt.savefig(output_graph_path, dpi=300)  # Save with high resolution
plt.show()

print(f"✅ Transition analysis graph saved at: {output_graph_path}")
