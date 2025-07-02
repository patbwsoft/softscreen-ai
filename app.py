import streamlit as st
import os
import cv2
import numpy as np
import tempfile
import yaml
from datetime import datetime
from mello_core.memory import TemporalMemory
from mello_core.detector import MelloDetector

# === Streamlit setup ===
os.environ["TORCH_ALLOW_TF32_CUDA"] = "1"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="SoftScreen", layout="centered")

# Inject CSS for mobile-friendly layout, static height, and purple branding
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Poppins:wght@700&display=swap');

        html, body, [class*="css"] {
            background-color: #5E4B8B;
            font-family: 'Inter', sans-serif;
            height: 100vh;
            overflow: hidden;
        }

        .title-container {
            text-align: center;
            padding-top: 2rem;
            padding-bottom: 1rem;
        }

        .softscreen-title {
            font-family: 'Poppins', sans-serif;
            font-size: 3rem;
            color: #D6B4FC;
        }

        .softscreen-subtitle {
            font-size: 1.2rem;
            color: #ffffff;
            font-weight: 600;
        }

        .stButton>button {
            background-color: #ffffff;
            color: #5E4B8B;
            border-radius: 30px;
            font-weight: bold;
            width: 100%;
            transition: 0.3s ease;
        }

        .stButton>button:hover {
            box-shadow: 0 0 8px #ffffff;
        }

        .card {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .stMultiSelect [data-baseweb="tag"] {
            background-color: #8E79C7 !important;
            color: #fff !important;
        }

        @keyframes blurLoop {
            0% { filter: blur(0px); opacity: 1; }
            50% { filter: blur(5px); opacity: 0.7; }
            100% { filter: blur(0px); opacity: 1; }
        }

        .upload-animation {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .upload-animation img {
            width: 60px;
            animation: blurLoop 2.5s infinite;
        }

        section[data-testid="stFileUploader"] div[role="button"] {
            background-color: #8E79C7 !important;
            color: white !important;
            border: 2px dashed white !important;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            font-weight: bold;
        }

        section[data-testid="stFileUploader"] div[role="button"]:hover {
            background-color: #A28BE5 !important;
            border-color: #fff;
        }

        @media screen and (max-width: 768px) {
            .softscreen-title {
                font-size: 2.2rem;
            }
            .softscreen-subtitle {
                font-size: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("""
<div class="title-container">
    <div class="softscreen-title">SoftScreen</div>
    <div class="softscreen-subtitle">Blur objects in videos using adaptive AI</div>
</div>
""", unsafe_allow_html=True)

# === Label selection ===
st.markdown('<div class="card">', unsafe_allow_html=True)
available_classes = ["dog", "person", "cat", "horse", "cow", "bottle"]
selected_labels = st.multiselect("Select what you want to blur:", available_classes, default=["dog"])
st.markdown('</div>', unsafe_allow_html=True)

# === Config saving ===
config_path = os.path.join("mello_core", "settings.yaml")
os.makedirs(os.path.dirname(config_path), exist_ok=True)

if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}

config["blur_labels"] = selected_labels
with open(config_path, "w") as f:
    yaml.dump(config, f, sort_keys=False)

# === Upload video ===
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your video...", type=["mp4", "mov"])
latest_file = None

if uploaded_file is not None:
    video_save_path = os.path.join("uploaded_videos", uploaded_file.name)
    with open(video_save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    latest_file = video_save_path
    st.success(f"Uploaded: {uploaded_file.name}")
st.markdown('</div>', unsafe_allow_html=True)

# === Process video ===
processing = False

if latest_file:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.info("Extracting frames...")
    session_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    extract_cmd = f"ffmpeg -i '{latest_file}' -vf fps=30 -q:v 2 '{frames_dir}/frame_%05d.jpg'"
    if os.system(extract_cmd) != 0:
        st.error("FFmpeg failed to extract frames.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    st.info("Blurring selected objects...")
    detector = MelloDetector()
    mello = TemporalMemory()

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    processed_dir = os.path.join(session_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    progress_bar = st.progress(0)

    with st.spinner("Mello is analyzing frames..."):
        processing = True
        st.markdown("""
        <div class="upload-animation">
            <img src="https://cdn-icons-png.flaticon.com/512/616/616408.png" alt="Dog Icon" />
        </div>
        """, unsafe_allow_html=True)

        for idx, frame_name in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_name)
            frame = cv2.imread(frame_path)

            detections = detector.detect(frame)
            output_frame = mello.process_frame(frame, detections)

            out_path = os.path.join(processed_dir, frame_name)
            cv2.imwrite(out_path, output_frame)
            progress_bar.progress((idx + 1) / len(frame_files))

    # === Rebuild video ===
    st.info("Rebuilding video with blur applied...")
    raw_output = os.path.join(session_dir, "output_raw.mp4")
    final_output = os.path.join(session_dir, "output_final.mp4")

    build_cmd = f"ffmpeg -y -framerate 30 -i '{processed_dir}/frame_%05d.jpg' -c:v libx264 -preset veryfast -pix_fmt yuv420p '{raw_output}'"
    if os.system(build_cmd) != 0:
        st.error("Failed to build video.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # === Add audio ===
    st.info("Reattaching audio (if available)...")
    audio_cmd = f"ffmpeg -y -i '{raw_output}' -i '{latest_file}' -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest '{final_output}'"
    if os.system(audio_cmd) != 0:
        st.warning("Audio could not be attached. Final video will be silent.")
        final_output = raw_output

    st.success("Here's your final video:")
    st.video(final_output)
    st.markdown('</div>', unsafe_allow_html=True)

    # === Mello's Reflection ===
    st.markdown('<div class="card">', unsafe_allow_html=True)
    journal_dir = os.path.join("outputs", "mello_journal")
    if os.path.exists(journal_dir):
        journal_files = sorted([f for f in os.listdir(journal_dir) if f.endswith(".txt")], reverse=True)
        if journal_files:
            st.subheader("Mello's Reflection")
            with open(os.path.join(journal_dir, journal_files[0]), "r", encoding="utf-8") as jf:
                st.code(jf.read(), language=None)
    st.markdown('</div>', unsafe_allow_html=True)

    # === Download ===
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with open(final_output, "rb") as f:
        st.download_button("Download Final Video", f, file_name="blurred_output.mp4")
    st.markdown('</div>', unsafe_allow_html=True)