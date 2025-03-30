# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:05:44 2024

@author: dalei
"""
import time
import torch
import cv2
import smtplib
import numpy as np
from email.mime.multipart import MIMEMultipart
from torch.utils.data import DataLoader, Dataset
from email.mime.text import MIMEText
from scipy.ndimage import zoom

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=int)

def email(body, subject):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    # TODO input the sender email
    sender_email = ""
    # TODO input dual-certificated email password
    sender_password = ""  
    # TODO input receiver email
    recipient_email = ""

    subject = subject
    body = body
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        # 连接到SMTP服务器并发送邮件
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # 开始TLS加密
        server.login(sender_email, sender_password)  # 登录
        server.sendmail(sender_email, recipient_email, msg.as_string())  # 发送邮件
        server.quit()  # 断开连接
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
    print(body)
    
def softmax(x):
    exp_x = np.exp(4*(x - np.max(x, axis=1, keepdims=True)))  # 稳定性调整
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def calculate_final_scores(logits, scores = []):
    probabilities = softmax(logits)
    final_scores = np.sum(probabilities * np.array(scores), axis=1)
    
    return final_scores

def extract_bright_window(frames, output_size):
    N, P, Q = frames.shape
    R, S = output_size
    extracted_windows = np.zeros((N, 1, R, S), dtype=np.uint8)

    for i in range(N):
        frame = frames[i]
        
        # 使用 Otsu’s 阈值来将图片转换为二值图像
        _, thresh_frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 查找二值图像中的外部轮廓
        contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 获取最大轮廓并找到其边界矩形
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 提取并调整窗口大小以适应 (R, S)
            window = frame[y:y+h, x:x+w]
            resized_window = cv2.resize(window, (S, R))
            extracted_windows[i, 0] = resized_window
        else:
            # 无轮廓时的默认处理
            extracted_windows[i, 0] = np.zeros((R, S), dtype=np.uint8)  # 使用黑色填充

    return extracted_windows

def resize_frames(frames, output_size):
    N, P, Q = frames.shape
    R, S = output_size
    resized_frames = np.zeros((N, 1, R, S), dtype=np.uint8)

    for i in range(N):
        frame = frames[i]
        
        # 确保frame是8位无符号整数
        frame = frame.astype(np.uint8)
        
        # 调整帧大小
        resized_frame = cv2.resize(frame, (S, R))
        resized_frames[i, 0] = resized_frame

    return resized_frames
