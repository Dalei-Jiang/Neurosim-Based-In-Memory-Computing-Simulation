# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 06:48:23 2025

@author: dalei
"""

import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(input_video_path, output_dir, chunk_length):
    if not os.path.exists(input_video_path):
        print(f"Input video not exists: {input_video_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video = VideoFileClip(input_video_path)
    duration = video.duration
    base_name = os.path.basename(input_video_path).rsplit('.', 1)[0][:-2]
    for i in range(0, int(duration), chunk_length):
        start_time = i
        end_time = min(i + chunk_length, duration)
        new_video = video.subclip(start_time, end_time)
        output_path = os.path.join(output_dir, f"{base_name}_{i // chunk_length}.mp4")
        new_video.write_videofile(output_path, codec='libx264')
    print(f"Splitting completed! The results are stored in: {output_dir}")

if __name__ == "__main__":
    input_video_path = "../../video/GaN/GaN_0_0.mp4"
    output_dir = "../../video/GaN/"+input_video_path[16:-6]+'/'+input_video_path[16:-4]  # 输出目录路径
    chunk_length = 15
    split_video(input_video_path, output_dir, chunk_length)
    print(output_dir)