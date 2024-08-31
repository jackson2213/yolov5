from PIL import Image, ImageDraw
import numpy as np

import os



files = os.listdir("E:\\download\\videos")
files.sort()

with open("E:\\download\\videos\\labels.txt", "r") as f:
    result=f.readlines()

map_list={}
for line in result:
    line = line.strip('\n')
    line = line.split(" ")
    map_list[line[0].strip('.mp4')]="ADD" if line[1]=="1" else "DROP"

base_path = "E:\\download\\"

for th in ["4","5","6","6.5","7","8","9"]:
    for arg in ["iou","xyl"]:
        accurate = 0
        files = os.listdir(f"{base_path}{th}\\result_{arg}")
        video_file = [file for file in files if file.endswith(".avi")]
        for file in video_file:
            key,_,label =file.strip(".avi").rpartition('_')
            if map_list[key] == label:
                accurate += 1
        print(f"{arg}\t 0.{th}\t{accurate}\t{len(video_file)} \t{accurate/len(video_file)}")





iou_file_path = "result_iou"

