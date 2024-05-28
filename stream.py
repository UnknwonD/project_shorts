import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import tempfile
import base64
import time
import cv2
import os

from process_model import pipeline_video

def get_values_and_indices(arr):
    result = []
    for i, row in enumerate(arr):
        for j, val in enumerate(row):
            result.append((i, j, val))
    return result

@st.cache_data
def process_video_data(video_path):
    video_data, audio_data = pipeline_video(video_path)
    
    # Prepare audio data
    new_audio_data = np.zeros((audio_data.shape[0], audio_data.shape[1] + 1))
    for i, audio_row in enumerate(audio_data):
        half_value = audio_row[1] / 2
        new_audio_data[i][0] = round(audio_row[0], 5)
        new_audio_data[i][1] = round(half_value, 5)
        new_audio_data[i][2] = round(half_value, 5)
    
    # Prepare video data
    new_video_data = np.round(video_data, 5)

    return new_video_data, new_audio_data

def compute_ensemble(video_data, audio_data, video_weight, audio_weight, threshold):
    # Ensure both arrays have the same length by truncating to the shortest length
    min_length = min(video_data.shape[0], audio_data.shape[0])
    video_data = video_data[:min_length]
    audio_data = audio_data[:min_length]
    
    # Compute ensemble scores
    ensemble_scores = (video_data * video_weight + audio_data * audio_weight) / (video_weight + audio_weight)
    ensemble_labels = ensemble_scores.argmax(axis=1)

    # Apply threshold to label "2"
    high_confidence_twos = ensemble_scores[:, 2] >= threshold
    ensemble_labels[high_confidence_twos] = 2
    
    
    return ensemble_labels, ensemble_scores


def get_max_values_and_indices(video_data, audio_data, video_weight, audio_weight, threshold):
    # Ensure both arrays have the same length by truncating to the shortest length
    min_length = min(video_data.shape[0], audio_data.shape[0])
    video_data = video_data[:min_length]
    audio_data = audio_data[:min_length]
    
    # Compute ensemble scores
    ensemble_scores = (video_data * video_weight + audio_data * audio_weight) / (video_weight + audio_weight)
    ensemble_labels = ensemble_scores.argmax(axis=1)

    # Apply threshold to label "2"
    high_confidence_twos = ensemble_scores[:, 2] >= threshold
    ensemble_labels[high_confidence_twos] = 2
    
    # Format output as (i, label, score)
    output = [(i, ensemble_labels[i], max(ensemble_scores[i])) for i in range(min_length)]
    
    sorted_data = sorted(output, key=lambda x: (x[1], x[2]), reverse=True)
    sorted_data = sorted(sorted_data[:5], key=lambda x: x[0])
    
    return sorted_data


def preprocess_shorts(video_path: str, label: list, output_path: str):
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not retrieve FPS from video.")
        vidcap.release()
        return
    
    interval = int(fps * 3)  # 3초 단위로 분리
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    sequences = []

    for lbl in label:
        index = lbl[0]
        start_frame = float(index * interval)
        
        print(f"Processing index {index}, start_frame {start_frame}")

        if start_frame >= total_frames:
            print(f"Warning: start_frame {start_frame} is out of bounds for video with {total_frames} frames.")
            continue
        
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_pos = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        if current_pos != start_frame:
            print(f"Error: Could not set video to start frame {start_frame}. Current position is {current_pos}.")
            continue

        frames = []
        for _ in range(interval):
            success, frame = vidcap.read()
            if not success:
                print("Warning: Could not read frame. Ending segment early.")
                break
            frames.append(frame)
        
        if len(frames) == interval:
            sequences.extend(frames)
    
    if sequences:
        height, width, layers = sequences[0].shape
        size = (width, height)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        for frame in sequences:
            out.write(frame)
        
        out.release()
    
    vidcap.release()





# Streamlit app
st.set_page_config(page_title='Video Highlight Trends', layout='wide')
st.title('Video Highlight Trends')

# Initialize session state for storing processed data
if 'new_video_data' not in st.session_state:
    st.session_state.new_video_data = None
if 'new_audio_data' not in st.session_state:
    st.session_state.new_audio_data = None
if 'page' not in st.session_state:
    st.session_state.page = 'Main'
if 'model_selection' not in st.session_state:
    st.session_state.model_selection = 'Video'

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Go to', ['Main', 'Confirmed Labels', 'Ensemble', 'Output'])

st.session_state.page = page

# File uploader for video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Create a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        video_path = temp_file.name

    # Process the video and audio data if not already processed
    if st.session_state.new_video_data is None and st.session_state.new_audio_data is None:
        new_video_data, new_audio_data = process_video_data(video_path)
        
        st.session_state.new_video_data = new_video_data
        st.session_state.new_audio_data = new_audio_data
    else:
        new_video_data = st.session_state.new_video_data
        new_audio_data = st.session_state.new_audio_data

    # Extract importance scores from the video data (assuming the importance is the max score in each 3-second block)
    importance_scores = new_video_data.max(axis=1).astype(int).tolist()

    if st.session_state.page == 'Main':
        # Main Page
        # Process both datasets
        values_and_indices1 = get_values_and_indices(new_video_data)
        values_and_indices2 = get_values_and_indices(new_audio_data)

        # Convert to DataFrame for easier plotting
        data1 = pd.DataFrame(values_and_indices1, columns=['Block Num', 'Highlight Label', 'Score'])
        data2 = pd.DataFrame(values_and_indices2, columns=['Block Num', 'Highlight Label', 'Score'])

        # Layout with columns for better organization
        col1, col2 = st.columns([2, 3])

        with col1:
            st.header('Dataset Selection')
            
            # Sidebar for dataset selection
            dataset = st.selectbox('Select Dataset', ['Video', 'Audio'])

            # Select the appropriate dataset
            if dataset == 'Video':
                data = data1
            else:
                data = data2

            # Filter data controls below the graph
            unique_column_indices = data['Highlight Label'].unique()
            selected_column_indices = st.multiselect('Select Column Index', unique_column_indices, default=unique_column_indices)

        with col2:
            st.header('Highlight Trends')

            # Filter data based on selection
            filtered_data = data[(data['Highlight Label'].isin(selected_column_indices))]

            # Create Altair chart
            chart = alt.Chart(filtered_data).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
                x='Block Num:Q',
                y='Score:Q',
                tooltip=['Block Num', 'Highlight Label', 'Score'],
                color='Highlight Label:N'
            ).properties(
                title='Highlight Trends'
            ).interactive()

            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)

    elif st.session_state.page == 'Confirmed Labels':
        # Confirmed Labels Page
        st.header('Confirmed Labels Visualization')

        # Layout with columns for better organization
        col1, col2 = st.columns([2, 3])

        with col1:
            st.header('Model Selection')
            
            # Select model results
            model_selection = st.radio('Select Model', ['Video', 'Audio'])
            st.session_state.model_selection = model_selection

        with col2:
            st.header('Confirmed Labels Trends')

            # Select model results
            model_selection = st.session_state.model_selection

            if model_selection == 'Video':
                confirmed_labels = new_video_data.argmax(axis=1)
            else:
                confirmed_labels = st.session_state.new_audio_data.argmax(axis=1)

            # Create a DataFrame for confirmed labels visualization
            confirmed_labels_df = pd.DataFrame({
                'Time Block': np.arange(len(confirmed_labels)),
                'Confirmed Label': confirmed_labels
            })

            # Line chart for confirmed labels
            confirmed_labels_chart = alt.Chart(confirmed_labels_df).mark_line(point=True).encode(
                x='Time Block:Q',
                y='Confirmed Label:Q',
                tooltip=['Time Block', 'Confirmed Label'],
                color='Confirmed Label:N'
            ).properties(
                title='Confirmed Labels Over Time'
            ).interactive()

            # Display the confirmed labels chart
            st.altair_chart(confirmed_labels_chart, use_container_width=True)
    
    elif st.session_state.page == 'Output':
        # Ensemble Page
        st.header('Model Output Visualization')

        # Sidebar for ensemble parameters
        st.sidebar.subheader('Ensemble Parameters')
        video_weight = st.sidebar.slider('Video Model Weight', 0.0, 1.0, 0.5)
        audio_weight = st.sidebar.slider('Audio Model Weight', 0.0, 1.0, 0.5)
        threshold = st.sidebar.slider('Threshold for Label 2', 0.0, 1.0, 0.5)

        # Compute ensemble results
        ensemble_labels, ensemble_scores = compute_ensemble(new_video_data, new_audio_data, video_weight, audio_weight, threshold)

        # Create a DataFrame for ensemble labels visualization
        ensemble_labels_df = pd.DataFrame({
            'Time Block': np.arange(len(ensemble_labels)),
            'Ensemble Label': ensemble_labels
        })

        # Line chart for ensemble labels
        ensemble_labels_chart = alt.Chart(ensemble_labels_df).mark_line(point=True).encode(
            x='Time Block:Q',
            y='Ensemble Label:Q',
            tooltip=['Time Block', 'Ensemble Label'],
            color='Ensemble Label:N'
        ).properties(
            title='Ensemble Labels Over Time'
        ).interactive()

        # Display the ensemble labels chart
        st.altair_chart(ensemble_labels_chart, use_container_width=True)
        
    elif st.session_state.page == 'Ensemble':
        # Ensemble Page
        st.header('Ensemble Model Visualization')

        # Sidebar for ensemble parameters
        st.sidebar.subheader('Ensemble Parameters')
        video_weight = st.sidebar.slider('Video Model Weight', 0.0, 1.0, 0.5)
        audio_weight = st.sidebar.slider('Audio Model Weight', 0.0, 1.0, 0.5)
        threshold = st.sidebar.slider('Threshold for Label 2', 0.0, 1.0, 0.5)
        compute_button = st.sidebar.button('Submit')

        if compute_button:
            # Assuming `new_video_data` and `new_audio_data` are available
            sorted_data = get_max_values_and_indices(new_video_data, new_audio_data, video_weight, audio_weight, threshold)
            output_path = "/Users/idaeho/Documents/GitHub/project_shorts/shorts.mp4"
            preprocess_shorts(video_path, sorted_data, "/Users/idaeho/Documents/GitHub/project_shorts/shorts.mp4")
            
            # Display the results
            st.subheader('Ensemble Results')
            for data in sorted_data:
                st.write(f"Index: {data[0]}, Label: {data[1]}, Scores: {data[2]}")

            # Ensure the video file is fully written before trying to display it
            time.sleep(4)  # Adding a short delay to ensure the file is written
            
            # Check if the file exists
            if os.path.exists(output_path):
                st.subheader('Ensemble Results')
                for data in sorted_data:
                    st.write(f"Index: {data[0]}, Label: {data[1]}, Scores: {data[2]}")
                
                # Display the video
                st.video(output_path)
            else:
                st.error("Error: Video file was not created successfully.")

        
