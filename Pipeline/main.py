import os
import json
import numpy as np
from tqdm import tqdm

from process_audio import extract_audio, preprocess_audio
from process_video import preprocess_video_every_3_seconds

video_length = ['2~5분']

new_video_data = []
output_json_path = f'processed/label/processed_video_data.json'

for i, leng in enumerate(video_length):

    output_video_dir = 'processed/video/'
    output_wav_dir = 'processed/wav/'
    output_audio_dir = 'processed/audio/'

    json_path = f'data/라벨링데이터/video_summary_validation_data({leng}).json'
    video_path = f'data/원천데이터/{leng}/'

    with open(json_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    if i == 0:
        video_idx = 1

    for item in tqdm(label_data):
        input_video_name = item['filename'] + '.mp4'
        input_video_path = os.path.join(video_path, input_video_name)

        output_video_name = f"processed_video_{video_idx}.npy"
        output_video_path = os.path.join(output_video_dir, output_video_name)
        
        output_wav_name = f"processed_video_{video_idx}.wav"
        output_wav_path =  os.path.join(output_wav_dir, output_wav_name)

        output_audio_name = f"processed_audio_{video_idx}.npy"
        output_audio_path =  os.path.join(output_audio_dir, output_audio_name)

        if not os.path.exists(input_video_path):
            print(f"Not Found : {input_video_path}")
            continue


        ######################################
        # 오디오 데이터 분리 후 저장
        extract_audio(input_video_path, output_wav_path) # mp4에서 오디오(wav) 추출 및 저장

        ######################################
        # 영상 전처리 진행 및 저장
        blocks_num = item["three_secs"][-1] + 1
        # print(item)
        annotations = item['annots']

        output = preprocess_video_every_3_seconds(input_video_path, (256, 256), blocks_num)
        np.save(output_video_path, output)

        ######################################
        # 오디오 전처리 및 학습 가능 파일로 저장
        # 저장 및 동기화 시간을 고려하여 영상 전처리 후 마지막 순서에 배치
        mel_spectrogram_segments = preprocess_audio(output_wav_path)
        np.save(output_audio_path, mel_spectrogram_segments)


        category = item["category"]

        item['filename'] = output_video_name
        item['category'] = category.encode('utf-8').decode()
        item['video_path'] = output_video_path
        item['audio_path'] = output_audio_path
        item['quality'] = '256 256' # 추 후에 데이터 사용할 때, split으로 사용할 수 있게 띄워쓰기로 구분

        video_idx += 1
        new_video_data.append(item)

# 전처리된 데이터에 대해 라벨을 새로 저장해줌
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(new_video_data, f, ensure_ascii=False, indent=2)

print(f"Process Finish :: {leng}")