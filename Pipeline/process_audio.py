import numpy as np

import librosa 
import librosa.display as dsp
from IPython.display import Audio
from moviepy.editor import VideoFileClip


def extract_audio(video_path, audio_path):
    '''
    1. 무비 파일 경로\n
    video_path = 'data/원천데이터/2~5분/test.mp4'
    audio_path = 'audio.wav'

    2. 오디오 추출\n
    extract_audio(video_path, audio_path)
    '''
    # 비디오 파일 열기
    video_clip = VideoFileClip(video_path)
    
    # 오디오 추출 및 저장
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path, verbose=False, logger=None)

    # 파일 닫기
    video_clip.close()
    
    
def preprocess_audio(audio_path, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=130, segment_duration=3):
    # 오디오 파일 로드
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    # 세그먼트 길이 계산 (샘플 단위)
    segment_length = int(sr * segment_duration)
    
    # 오디오를 3초 단위 세그먼트로 나누기
    segments = []
    num_segments = len(audio) // segment_length
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = audio[start_idx:end_idx]
        
        # 멜 스펙트로그램 추출 및 디시벨 변환
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        segments.append(mel_spectrogram_db)
    
    return np.array(segments)

