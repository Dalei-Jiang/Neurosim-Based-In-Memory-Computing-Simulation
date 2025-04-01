import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.ndimage import convolve, zoom
def VideoLoad(videoname):
    cap = cv2.VideoCapture(videoname)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    frames = []
    while True:
        ret, frame = cap.read() 
        if not ret:
            break  
        frames.append(frame) 
    cap.release()
    frame = np.array(frames)
    return frame

def Imageset_visualize(sharpened_image, ind, M, N, x, y, mode = 'rainbow'):
    indices = ind[:M*N]
    plt.figure(figsize=(x, y)) 
    for i, idx in enumerate(indices, 1):
        ax = plt.subplot(M, N, i) 
        ax.imshow(sharpened_image[idx],cmap=mode) 
        ax.axis('off') 
        ax.set_title(f'Image {idx}') 
    plt.tight_layout()
    plt.show()

def resize_image(Size, sharpened_image):
    new_data = np.empty((sharpened_image.shape[0], 1, Size, Size))
    for i in range(sharpened_image.shape[0]):
        image = sharpened_image[i, 0]
        resized_image = cv2.resize(image, (Size, Size), interpolation=cv2.INTER_CUBIC)
        new_data[i, 0] = resized_image
    return new_data

def bound(arr, threshold):
    max_value = max(arr)
    threshold_value = threshold * max_value
    start = 0
    end = -10000
    if arr[-1]==0 or arr[0]==0:
        flag = True
    else:
        flag = False
    while start < len(arr) and arr[start] < threshold_value:
        start += 1    
    end = len(arr) - 1
    while end >= 0 and arr[end] < threshold_value:
        end -= 1
    return (start, end + 1, flag) if start <= end else (0, 0, False)

def extract_subimages(data, M, N, K):
    num_images, channels, height, width = data.shape
    subimages = []
    for img in data:
        for _ in range(K):
            x = np.random.randint(0, width - N + 1)
            y = np.random.randint(0, height - M + 1)
            subimage = img[:, y:y+M, x:x+N]
            subimages.append(subimage)
    subimages = np.array(subimages)
    return subimages

def average_image(framecut, N):
    num_images = framecut.shape[0]
    new_num_images = num_images - N + 1
    new_images = np.zeros((new_num_images, framecut.shape[1], framecut.shape[2]))
    for i in range(new_num_images):
        new_images[i] = np.mean(framecut[i:i+N], axis=0)
    return new_images.astype(np.float32)

def Gray_scaling(framecut):
    gray_img = 0.2989 * framecut[:, :, :, 0] 
    gray_img += 0.5870 * framecut[:, :, :, 1] 
    gray_img += 0.1140 * framecut[:, :, :, 2]
    gray_img /= 255.0 
    return gray_img

# def extract_bright_window(frames, output_size):
#     N, P, Q, _ = frames.shape
#     R, S = output_size
#     extracted_windows = np.zeros((N, R, S, 3), dtype=np.uint8)
#     for i in range(N):
#         frame = frames[i]
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, thresh_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         window = frame[y:y+h, x:x+w]
#         resized_window = cv2.resize(window, (S, R))
#         extracted_windows[i] = resized_window
#     return extracted_windows

def extract_bright_window(frames, output_size):
    N, P, Q, _ = frames.shape
    R, S = output_size
    extracted_windows = np.zeros((N, R, S, 3), dtype=np.uint8)
    for i in range(N):
        frame = frames[i]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:  # 检查是否有轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            window = frame[y:y+h, x:x+w]
            resized_window = cv2.resize(window, (S, R))
            extracted_windows[i] = resized_window
        else:
#             print(f"No contours found for frame {i}. Using default (black) window.")
            pass
    
    return extracted_windows


def transform_image(image, threshold=0.01):
    flattened_image = image.flatten()
    threshold = np.percentile(flattened_image, 100-threshold)
    new_image = np.where(image >= threshold, 1, 0)
    return new_image

def find_clusters(arr):
    clusters = []
    start = None
    for i, value in enumerate(arr):
        if value != 0 and start is None:
            start = i
        elif value == 0 and start is not None:
            end = i - 1
            cluster_sum = np.sum(arr[start:end + 1])
            clusters.append((start, end, cluster_sum))
            start = None
    if start is not None:
        end = len(arr) - 1
        cluster_sum = np.sum(arr[start:end + 1])
        clusters.append((start, end, cluster_sum))
    num_clusters = len(clusters)
    cluster_sums = [cluster[2] for cluster in clusters]
    cluster_indices = [(cluster[0], cluster[1]) for cluster in clusters]
    return num_clusters, cluster_sums, cluster_indices