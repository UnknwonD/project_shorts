import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from process_model import pipeline_video

def get_values_and_indices(arr):
    result = []
    for i, row in enumerate(arr):
        for j, val in enumerate(row):
            result.append((i, j, val))
    return result


video_path = "/Users/idaeho/Desktop/ai_cap/pang.mp4"
print("Start")
video_data, audio_data = pipeline_video(video_path)
print("end")

new_audio_data = np.zeros((audio_data.shape[0], audio_data.shape[1] + 1))

for i, audio_data in enumerate(audio_data):
    half_value = audio_data[1] / 2
    new_audio_data[i][0] = round(audio_data[0], 5)
    new_audio_data[i][1] = round(half_value, 5)
    new_audio_data[i][2] = round(half_value, 5)
    
new_video_data = np.round(video_data, 5)

# Process both datasets
values_and_indices1 = get_values_and_indices(new_video_data)
values_and_indices2 = get_values_and_indices(new_audio_data)

# Convert to DataFrame for easier plotting
data1 = pd.DataFrame(values_and_indices1, columns=['Block Num', 'Highlight Label', 'Score'])
data2 = pd.DataFrame(values_and_indices2, columns=['Block Num', 'Highlight Label', 'Score'])

# Streamlit app
st.title('Video Highlight Trends')

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

# Filter data based on selection
filtered_data = data[(data['Highlight Label'].isin(selected_column_indices))]

# Create Altair chart
chart = alt.Chart(filtered_data).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
    x='Row Index:Q',
    y='Value:Q',
    tooltip=['Block Num', 'Highlight Label', 'Score'],
    color='Column Index:N'
).properties(
    title='Highlight Trends'
).interactive()

# Display the chart in Streamlit
st.altair_chart(chart, use_container_width=True)
