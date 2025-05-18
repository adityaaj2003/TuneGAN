import os
import base64
from pathlib import Path

import torch
import torchaudio
import streamlit as st
from audiocraft.models import MusicGen


###############################################################################
# ------------------------------  PAGE CONFIG  ------------------------------ #
###############################################################################
st.set_page_config(
    page_title="TuneGAN · Text‑to‑Music",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS (Google font, card shadows, primary color)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Streamlit primary color override */
    :root {
        --primary-base: #6366f1;   /* indigo‑500 */
        --secondary-background-color: rgba(99, 102, 241, 0.08);
    }

    /* Fancy card */
    .st-audio-card {
        background: var(--secondary-background-color);
        padding: 1.2rem 1rem 1.4rem;
        border-radius: 1rem;
        box-shadow: 0 10px 18px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

###############################################################################
# ------------------------------  HELPERS  ---------------------------------- #
###############################################################################
AUDIO_DIR = Path("audio_output")
AUDIO_DIR.mkdir(exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_model():
    return MusicGen.get_pretrained("facebook/musicgen-small")

def generate_music(description: str, duration: int) -> torch.Tensor:
    model = load_model()
    model.set_generation_params(use_sampling=True, top_k=250, duration=duration)
    # single‑element list -> tensor [C, T]
    return model.generate([description], progress=True)[0]

def save_waveform(wave: torch.Tensor, idx: int = 0, sr: int = 32_000) -> Path:
    path = AUDIO_DIR / f"audio_{idx}.wav"
    torchaudio.save(str(path), wave.detach().cpu(), sr)
    return path

def download_link(file_path: Path, label: str = "Download") -> str:
    data = file_path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"""
        <a href="data:audio/wav;base64,{b64}" download="{file_path.name}">
            {label} ⬇
        </a>
    """

###############################################################################
# ------------------------------  SIDEBAR  ---------------------------------- #
###############################################################################
st.sidebar.header("🛠️  Controls")

description = st.sidebar.text_area(
    "Enter a musical prompt",
    placeholder="e.g. 'Lo‑fi chill beats with vinyl crackle and soft piano'",
    height=120,
)

duration = st.sidebar.slider("Duration (seconds)", 1, 30, 10, step=1)
generate_btn = st.sidebar.button("🎼  Generate Music", type="primary", use_container_width=True)

st.sidebar.markdown("---")


###############################################################################
# ------------------------------  MAIN AREA  -------------------------------- #
###############################################################################
st.title("TuneGAN 🎵")
st.caption("Turn a short text prompt into royalty‑free music in seconds.")

if generate_btn:
    if not description.strip():
        st.warning("Please enter a prompt first. 💬")
        st.stop()

    with st.spinner("🎙️  Composing… this may take ~10 s"):
        waveform = generate_music(description, duration)
        file_path = save_waveform(waveform)

    # ---- Result card ------------------------------------------------------ #
    with st.container():
        st.markdown('<div class="st-audio-card">', unsafe_allow_html=True)
        st.subheader("Your Track")
        with st.expander("▶️  Listen / Download", expanded=True):
            st.audio(file_path.read_bytes(), format="audio/wav")
            st.markdown(download_link(file_path, "Save .wav"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.success("Done! Enjoy your music. 🎧")

else:
    st.info("Enter a prompt in the sidebar and hit **Generate Music** to begin.")
