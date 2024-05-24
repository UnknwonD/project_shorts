import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import tempfile
import base64

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

def video_with_progress(video_path, importance_scores):
    # Display video
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    video_url = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}"

    # Custom CSS for the video container
    css_code = f"""
    <style>
    .video-container {{
        position: relative;
        width: 700px;
        margin: 0 auto;
    }}
    .video-slider {{
        position: absolute;
        top: -20px;
        width: 100%;
        z-index: 1;
        height: 10px;
        opacity: 0.5;
    }}
    </style>
    """

    # JavaScript to sync slider and video
    js_code = f"""
    <script>
    document.addEventListener("DOMContentLoaded", function() {{
        const video = document.getElementById("video");
        const slider = document.getElementById("slider");

        slider.addEventListener("input", function() {{
            video.currentTime = (this.value / 100) * video.duration;
        }});

        video.addEventListener("timeupdate", function() {{
            slider.value = (video.currentTime / video.duration) * 100;
        }});
    }});
    </script>
    """

    st.markdown(css_code, unsafe_allow_html=True)
    st.markdown(js_code, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="video-container">
            <video id="video" width="700" controls>
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <input type="range" id="slider" class="video-slider" min="0" max="100" step="0.1">
        </div>
    """, unsafe_allow_html=True)

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
page = st.sidebar.radio('Go to', ['Main', 'Confirmed Labels', 'Ensemble'])

st.session_state.page = page

# File uploader for video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Create a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        video_path = temp_file.name

    st.write("Processing video, please wait...")

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

        # Display video with progress
        video_with_progress(video_path, importance_scores)

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
    
    elif st.session_state.page == 'Ensemble':
        # Ensemble Page
        st.header('Ensemble Model Visualization')

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
