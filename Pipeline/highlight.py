import os
import numpy as np

from process_audio import extract_audio, preprocess_audio
from process_video import preprocess_video_every_3_seconds

from tensorflow.keras.models import load_model


video_path = "../data/원천데이터/2~5분/SUMVID_SHORT_TRAIN_01/test.mp4"


def pipeline_video(video_path:str):
    ####### 1. 사용자가 입력한 비디오 불러오기
    if not os.path.exists(video_path):
        print(f"Video Not Found : {video_path}")
        return
    
    ####### 2. 사용자가 입력한 비디오 전처리하기
    # 1) 오디오와 비디오 분리
    # 2) 분리된 비디오와 오디오 각각 전처리
    
    audio = extract_audio(audio)
    audio = preprocess_audio(video_path)
    

    video = preprocess_video_every_3_seconds(video_path, (256, 256), 3)

    ####### 3. 모델 불러오기
    # 1) 비디오 모델 load
    # 2) 오디오 모델 load

    video_model = load_model("video_model.h5")
    audio_model = load_model("audio_model_resnet.h5")

    ####### 4. 모델 추론
    # 1) 비디오 모델 Inference
    # 2) 오디오 모델 Inference

    video_output = video_model.predict(video)
    audio_output = audio_model.predict(audio)

    ####### 5. Ensemble
    ##
    ensemble_output = np.mean([video_output, audio_output], axis=0)
    final_predictions = np.argmax(ensemble_output, axis=1)
    
    ####### 6. Return
    return final_predictions

