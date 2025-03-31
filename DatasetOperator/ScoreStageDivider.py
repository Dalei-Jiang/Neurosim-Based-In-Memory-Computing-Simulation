# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:26:04 2024

@author: dalei
"""

from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(input_file, breakpoints):
    """
    Splits the input video into multiple segments based on the breakpoints.

    :param input_file: Path to the input MP4 video file.
    :param breakpoints: List of breakpoints in seconds where the video should be split.
    """
    # Load the video file
    video = VideoFileClip(input_file)
    
    # Append the duration of the video to the breakpoints to handle the last segment
    breakpoints.append(video.duration)
    
    # Initialize start time
    start_time = 0
    
    # Split the video and save segments
    for i, end_time in enumerate(breakpoints):
        # Clip the video from start_time to end_time
        subclip = video.subclip(start_time, end_time)
        # Save the subclip to a new file
        output_file = f"{i}.mp4"
        subclip.write_videofile(output_file, codec="libx264")
        # Update start_time for the next segment
        start_time = end_time

# Example usage
input_file = ""
breakpoints = [12,131,148,175,195,211,460]
split_video(input_file, breakpoints)