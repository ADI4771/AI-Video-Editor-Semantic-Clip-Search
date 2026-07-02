import streamlit as st
import os
import sys
import shutil
import tempfile
from pathlib import Path

#  Path setup 
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pipeline.extract_frames import extract_frames
from pipeline.embeddings import compute_frame_embeddings
from pipeline.search import search_frames, group_into_segments
from pipeline.clip_generator import generate_clips_from_segments, merge_clips
from utils.video_utils import get_video_info
from utils.query_simplifier import simplify_query

# Page config
st.set_page_config(
    page_title="AI Video Editor",
    page_icon="A",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0e0e0e; }
    .stApp { background-color: #0e0e0e; color: #f0f0f0; }
    .card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .score-badge {
        background: #16213e;
        color: #00d4ff;
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.85em;
        font-weight: bold;
    }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

#  Session state 
for key in ["frame_data", "video_path", "video_info", "work_dir", "segments", "clip_paths", "merged_path"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Header 
st.title(" AI Video Editor")
st.caption("Upload a video · Ask in plain English · Get your clips")
st.divider()


# STEP 1 — UPLOAD VIDEO

st.subheader("① Upload Video")
uploaded_file = st.file_uploader(
    "Drop your video here",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Supported: MP4, MOV, AVI, MKV, WebM",
)

fps_option = st.slider("Frame extraction rate (FPS)", min_value=0.5, max_value=3.0, value=1.0, step=0.5,
                        help="Higher = more accurate search, slower processing")

if uploaded_file and st.button(" Process Video", type="primary"):
    # Clean up previous session
    if st.session_state.work_dir and Path(st.session_state.work_dir).exists():
        shutil.rmtree(st.session_state.work_dir)

    work_dir = tempfile.mkdtemp(prefix="aivideo_")
    st.session_state.work_dir = work_dir

    # Save uploaded file
    video_path = os.path.join(work_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.video_path = video_path

    with st.status("Processing video...", expanded=True) as status:
        st.write(" Reading video metadata...")
        info = get_video_info(video_path)
        st.session_state.video_info = info
        st.write(f"  → {info['width']}×{info['height']} | {info['fps']:.1f} fps | {info['duration_str']}")

        st.write(f" Extracting frames at {fps_option} fps...")
        frames_dir = os.path.join(work_dir, "frames")
        frames = extract_frames(video_path, frames_dir, fps=fps_option)
        st.write(f"  → {len(frames)} frames extracted")

        st.write(" Computing CLIP embeddings...")
        embeddings_dir = os.path.join(work_dir, "embeddings")
        frame_data = compute_frame_embeddings(frames, embeddings_dir)
        st.session_state.frame_data = frame_data
        st.write(f"  → Embeddings ready for {len(frame_data['embeddings'])} frames")

        status.update(label=" Video ready for search!", state="complete")

# Show video info if already processed
if st.session_state.video_info:
    info = st.session_state.video_info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", info["duration_str"])
    c2.metric("Resolution", f"{info['width']}×{info['height']}")
    c3.metric("FPS", f"{info['fps']:.1f}")
    c4.metric("Frames indexed", len(st.session_state.frame_data["embeddings"]))

# 
# STEP 2  SEARCH
#
if st.session_state.frame_data:
    st.divider()
    st.subheader("Describe What You're Looking For")

    col_q, col_k = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "Search query",
            placeholder='e.g. "snowy mountains at sunset", "person running on beach"',
            label_visibility="collapsed",
        )
    with col_k:
        top_k = st.number_input("Top K clips", min_value=1, max_value=10, value=3, step=1)

    use_llm = st.checkbox(" Use AI to optimize my query (LLM layer)", value=True,
                           help="Simplifies your query for better CLIP matching")

    if query and st.button(" Search", type="primary"):
        with st.spinner("Searching..."):

            # LLM query simplification
            if use_llm:
                with st.expander(" Query Optimization", expanded=True):
                    simplified = simplify_query(query)
                    st.write(f"**Original:** {simplified['original']}")
                    st.write(f"**Optimized:** `{simplified['primary']}`")
                    if simplified["alternatives"]:
                        st.write(f"**Alternatives:** {' · '.join(simplified['alternatives'])}")
                search_query = simplified["primary"]
            else:
                search_query = query

            # Search with primary + alternatives
            all_results = search_frames(
                search_query,
                st.session_state.frame_data,
                top_k=top_k * 3,  # over-fetch for grouping
            )

            if use_llm and simplified.get("alternatives"):
                for alt in simplified["alternatives"]:
                    alt_results = search_frames(alt, st.session_state.frame_data, top_k=top_k)
                    # Merge: keep best score per timestamp
                    existing_ts = {r["timestamp"] for r in all_results}
                    for r in alt_results:
                        if r["timestamp"] not in existing_ts:
                            all_results.append(r)

            # Smart segment grouping
            segments = group_into_segments(all_results, gap_threshold=8.0, clip_padding=5.0)
            segments = segments[:top_k]  # Keep top-K segments
            st.session_state.segments = segments

        if segments:
            st.success(f"Found {len(segments)} matching segment(s)!")
        else:
            st.warning("No matching segments found. Try a different query or lower the threshold.")


# STEP 3 — PREVIEW THUMBNAILS

if st.session_state.segments:
    st.divider()
    st.subheader("③ Preview Results")

    segments = st.session_state.segments
    cols = st.columns(min(len(segments), 3))

    for i, seg in enumerate(segments):
        col = cols[i % 3]
        with col:
            # Thumbnail
            if seg.get("frame_path") and Path(seg["frame_path"]).exists():
                col.image(seg["frame_path"], use_column_width=True)

            duration = seg["end"] - seg["start"]
            start_str = f"{int(seg['start']//60)}:{int(seg['start']%60):02d}"
            end_str = f"{int(seg['end']//60)}:{int(seg['end']%60):02d}"

            col.markdown(f"""
<div class="card">
    <b>Clip {i+1}</b> &nbsp;
    <span class="score-badge">Score: {seg['peak_score']:.2f}</span><br/>
     {start_str} → {end_str} ({duration:.1f}s)<br/>
     {len(seg['matched_timestamps'])} matched frame(s)
</div>
""", unsafe_allow_html=True)


# STEP 4 — GENERATE & DOWNLOAD

if st.session_state.segments:
    st.divider()
    st.subheader(" Generate & Download")

    col_a, col_b = st.columns(2)
    export_individual = col_a.checkbox("Export individual clips", value=False)
    export_merged = col_b.checkbox("Export merged highlight reel", value=True)

    if st.button(" Generate Clips", type="primary"):
        work_dir = st.session_state.work_dir
        clips_out_dir = os.path.join(work_dir, "output_clips")

        with st.status("Generating clips...", expanded=True) as status:
            st.write(" Cutting segments from video...")
            clip_paths = generate_clips_from_segments(
                st.session_state.video_path,
                st.session_state.segments,
                clips_out_dir,
            )
            st.session_state.clip_paths = clip_paths
            st.write(f"  → {len(clip_paths)} clip(s) created")

            if export_merged and len(clip_paths) > 0:
                st.write(" Merging into highlight reel...")
                merged_path = os.path.join(work_dir, "highlight_reel.mp4")
                merge_clips(clip_paths, merged_path)
                st.session_state.merged_path = merged_path

            status.update(label=" Done!", state="complete")

    # Download buttons
    if st.session_state.merged_path and Path(st.session_state.merged_path).exists():
        with open(st.session_state.merged_path, "rb") as f:
            st.download_button(
                label="⬇ Download Highlight Reel",
                data=f,
                file_name="highlight_reel.mp4",
                mime="video/mp4",
                type="primary",
            )
        st.video(st.session_state.merged_path)

    if export_individual and st.session_state.clip_paths:
        st.write("**Individual clips:**")
        for i, cp in enumerate(st.session_state.clip_paths):
            if Path(cp).exists():
                with open(cp, "rb") as f:
                    st.download_button(
                        label=f"⬇ Clip {i+1}",
                        data=f,
                        file_name=Path(cp).name,
                        mime="video/mp4",
                        key=f"dl_clip_{i}",
                    )

#  Footer 
st.divider()
st.caption(" Powered by CLIP · ffmpeg · Claude API")
