#-*- coding:utf-8 -*-

import os
import json

from process import compress_video

json_path = 'data/라벨링데이터/2~5분/video_summary_validation_data(2~5분).json'
video_path = 'data/원천데이터/2~5분/SUMVID_SHORT_VALID.zip/'
output_json_path = 'processed/label/processed_video_data.json'
output_video_dir = 'processed/video/'

with open(json_path, 'r', encoding='utf-8') as f:
    label_data = json.load(f)

new_video_data = []
video_idx = 1

for item in label_data:
    input_video_name = item['filename'] + '.mp4'
    input_video_path = os.path.join(video_path, input_video_name)
    output_video_name = f"processed_video_{video_idx}.mp4"
    output_video_path = os.path.join(output_video_dir, output_video_name)

    if not os.path.exists(input_video_path):
        print(f"Not Found : {input_video_path}")
        continue
    
    compress_video(input_video_path, output_video_path, 256, 256, 3) # 영상 전처리 진행

    category = item["category"]

    item['filename'] = output_video_name
    item['category'] = category.encode('utf-8').decode()
    item['path'] = output_video_path
    item['quality'] = '256 256' # 추 후에 데이터 사용할 때, split으로 사용할 수 있게 띄워쓰기로 구분

    video_idx += 1
    new_video_data.append(item)

# 전처리된 데이터에 대해 라벨을 새로 저장해줌
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(new_video_data, f, ensure_ascii=False, indent=2)

print("Process Finish")