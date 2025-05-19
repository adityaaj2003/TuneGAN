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

    /* Developer badge */
    .dev-badge {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--primary-base);
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-top: 1.5rem;
    }
    .dev-badge svg {
        width: 18px;
        height: 18px;
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
        <a href=\"data:audio/wav;base64,{b64}\" download=\"{file_path.name}\">{label} ⬇</a>
    """

###############################################################################
# ------------------------------  SIDEBAR  ---------------------------------- #
###############################################################################
page = st.sidebar.radio("Navigate", ["🎹 Generate", "ℹ️ About"], label_visibility="collapsed")

# -- Developer badge in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="dev-badge">
        <svg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='currentColor'><path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 16h-1v-4h-1m-1 4H9m4-2h1m1 0h1m1 0h1m-5 5H7a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h6a2 2 0 012 2v4' /></svg>
        Developed by <span>Aditya Anand Joshi</span>
    </div>
    """,
    unsafe_allow_html=True,
)

###############################################################################
# ------------------------------  MAIN AREA  -------------------------------- #
###############################################################################
if page == "🎹 Generate":
    st.title("TuneGAN 🎵")
    st.caption("Turn a short text prompt into royalty‑free music in seconds.")

    description = st.text_area(
        "Enter a musical prompt",
        placeholder="e.g. 'Lo‑fi chill beats with vinyl crackle and soft piano'",
        height=120,
    )
    col1, col2 = st.columns([1,3])
    with col1:
        duration = st.number_input("Duration (s)", 1, 30, 10)
    with col2:
        generate_btn = st.button("🎼  Generate Music", type="primary")

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
        st.info("Enter a prompt and hit **Generate Music** to begin.")

# ----------------------------- ABOUT PAGE ---------------------------------- #
else:
    st.title("About TuneGAN")
    st.write(
        """
        **TuneGAN** is a lightweight text‑to‑music generator.  

        **How it works:**  

        1. You provide a short natural‑language prompt describing the style, instruments, or mood you want.  
        2. The prompt is tokenised and passed to the pretrained MusicGen model with GAN.  
        3. The model autoregressively generates a multitrack audio waveform at 32 kHz.  
        4. The waveform is returned to Streamlit, saved as **.wav**, and streamed back to you.  

        Generation runs entirely on‑device/server—no external API calls—so your prompts never leave the host machine.
        """
    )

    st.markdown("---")
    st.subheader("📋 Coordinator")
    colC1, colC2 = st.columns([1,3])
    with colC1:
        st.image("https://avatars.githubusercontent.com/u/11537016?v=4", width=110)
    with colC2:
        st.markdown("**Prof. Anand Vaidya**  \n" \
        "Project coordinator and academic mentor.")
    st.markdown("---")
    st.subheader("👥  Developers")

    devs = [
        {"name": "Aditya Anand Joshi", "img": "https://avatars.githubusercontent.com/u/11537016?v=4", "bio": "Final‑year B.E. Computer Science, SDM College of Engineering & Technology, Dharwad"},
        {"name": "Kirankumar Devarakondi", "img": "https://avatars.githubusercontent.com/u/11537016?v=4", "bio": "Final‑year B.E. Computer Science, SDM College of Engineering & Technology, Dharwad"},
        {"name": "Vallabh Nayamgoud", "img": "https://avatars.githubusercontent.com/u/11537016?v=4", "bio": "Final‑year B.E. Computer Science, SDM College of Engineering & Technology, Dharwad"},
        {"name": "Varun Sultanpur", "img": "https://avatars.githubusercontent.com/u/11537016?v=4", "bio": "Final‑year B.E. Computer Science, SDM College of Engineering & Technology, Dharwad"},
    ]

    for d in devs:
        colA, colB = st.columns([1,3])
        with colA:
            st.image(d["img"], width=110)
        with colB:
            st.markdown(f"**{d['name']}**  \n{d['bio']}")
        st.markdown("---")

    st.caption("© 2025 TuneGAN Team – All rights reserved.")
