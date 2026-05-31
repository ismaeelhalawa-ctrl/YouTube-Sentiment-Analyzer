# ============================================================
# YouTube Comment Sentiment Analyzer PRO
# ============================================================
import re
import io
import warnings
from collections import Counter
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import (AutoModelForSequenceClassification , AutoTokenizer , pipeline)
import os
from dotenv import load_dotenv
from database import save_analysis
from database import get_all_analyses
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
AR_MODEL = "Ammar-alhaj-ali/arabic-MARBERT-sentiment"
EN_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

POSITIVE_EMOJIS = {
    "😍", "❤️", "❤", "🔥", "👏", "😊",
    "👍", "🥰", "💖", "😎", "🙏", "💪", "✨", "🎉",
    "😁", "😄", "😃", "🙂", "🤩", "😘", "💕", "💯",
    "🌹", "🌟", "🎊", "🎈", "🤗", "😇", "👌", "🫶",
    "💙", "💚", "💛", "🧡", "💜", "🤝", "🏆", "🥳"}

NEGATIVE_EMOJIS = {
    "😡", "🤬", "👎", "💔", "😢", "😭", "🙄", "😒",
    "🤮", "😤", "😠", "🗑️", "🤡", "😑", "😞", "😕",
    "☠️", "💀", "😫", "😩", "😰", "😱", "😤", "🤢",
    "👺", "🚫", "❌", "⚠️"}

SARCASM_EMOJIS = {
    "🙄", "😒", "😏", "🤡", "😂", "🤣", "😑",
    "👌", "👏", "🤦", "🤦‍♂️", "🤦‍♀️"}



SARCASM_PHRASES = {"يا سلام","واو","عبقري","أكيد","طبعا","ما شاء الله عليك","برافو عليك","روعة يعني",
                   "ما أحلى","ولا أروع","أكيد يعني","أقوى شيء شفته","مبدع جدًا","يا للعظمة",
                   "تحفة فنية","best thing ever","wow genius","sure bro","great job","amazing quality",
                   "perfect as always"}


ARABIC_STOP = {
    "في", "من", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه", "هو",
    "هي", "أن", "ان", "كان", "لي", "لك", "له", "لها", "نحن", "أنت",
    "انت", "أنا", "انا", "هم", "عند", "كل", "قد", "لم", "لا", "ما",
    "يا", "اي", "اللي", "الي", "وان", "وفي", "وهو", "وهي", "انه",
    "بس", "زي", "عشان", "علشان", "دا", "ده", "دي", "مش", "مو", "مب",
    "وش", "يلي", "اللي", "حتى", "حتا", "لو", "لان", "لأن", "بعد",
    "قبل", "فيه", "فيها", "منه", "منها", "عليه", "عليها",
}
ENGLISH_STOP = {
    "the", "a", "an", "is", "it", "this", "that", "to", "and", "of",
    "in", "for", "on", "with", "i", "you", "we", "are", "was", "be",
    "have", "not", "but", "your", "they", "he", "she", "at", "by",
    "as", "do", "so", "if", "or", "its", "me", "my", "just", "very",
}
STOP_WORDS = ARABIC_STOP | ENGLISH_STOP

TOPIC_KEYWORDS = {
    "التمثيل / Acting":     ["تمثيل", "ممثل", "ممثلة", "اداء", "actor", "acting", "performance", "cast"],
    "الموسيقى / Music":     ["موسيقى", "اغنية", "أغنية", "صوت", "music", "song", "soundtrack", "beat"],
    "الإيقاع / Pacing":     ["طويل", "قصير", "ممل", "مملة", "بطيء", "slow", "fast", "boring", "pacing"],
    "التصوير / Visuals":    ["كاميرا", "تصوير", "لقطة", "camera", "cinematography", "shot", "visual"],
    "الإضاءة / Lighting":   ["اضاءة", "إضاءة", "lighting", "light", "dark", "bright"],
    "الذكاء الاصطناعي / AI":["ai", "chatgpt", "gpt", "llm", "ذكاء اصطناعي", "ذكاء"],
    "الألعاب / Gaming":     ["game", "gaming", "لعبة", "العاب", "gamer", "gameplay"],
    "التعليم / Education":  ["تعليم", "شرح", "درس", "course", "lesson", "tutorial", "learn"],
    "القصة / Story":        ["قصة", "حبكة", "story", "plot", "script", "ending", "نهاية"],
    "الإخراج / Direction":  ["مخرج", "إخراج", "director", "direction", "scene"],}

COLOR_MAP = {
    "positive": "#22C55E",   # canonical green
    "negative": "#EF4444",   # canonical red
    "neutral":  "#9CA3AF",   # canonical gray
    "spam":     "#f97316",}

st.set_page_config(
    page_title='YouTube Sentiment Analyzer PRO',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded',)

# ================
# Language Toggle
# ================
if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = "en"

def t(en_text: str, ar_text: str) -> str:
    """Return English or Arabic text based on current UI language."""
    return ar_text if st.session_state["ui_lang"] == "ar" else en_text

def _rtl_class() -> str:
    return ' dir="rtl" style="text-align:right;"' if st.session_state["ui_lang"] == "ar" else ''


# Glassmorphism + Floating Shapes + RTL
st.markdown("""
<style>
/* ========== BASE & RTL ========== */
* { margin: 0; padding: 0; box-sizing: border-box; }

.stApp {
    background: linear-gradient(135deg, #0A0B1A 0%, #111225 50%, #1a1a2e 100%) !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
    direction: rtl;
}

/* ========== FLOATING BACKGROUND SHAPES ========== */
.stApp::before {
    content: '';
    position: fixed;
    top: -400px;
    right: -400px;
    width: 800px;
    height: 800px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.25) 0%, rgba(139, 92, 246, 0.12) 40%, rgba(139, 92, 246, 0.05) 70%, transparent 85%);
    border-radius: 50%;
    z-index: 0;
    animation: float-massive-1 25s ease-in-out infinite;
    pointer-events: none;
}

.stApp::after {
    content: '';
    position: fixed;
    bottom: -500px;
    left: -400px;
    width: 900px;
    height: 900px;
    background: radial-gradient(circle, rgba(236, 72, 153, 0.22) 0%, rgba(236, 72, 153, 0.1) 50%, rgba(236, 72, 153, 0.04) 75%, transparent 90%);
    border-radius: 50%;
    z-index: 0;
    animation: float-massive-2 30s ease-in-out infinite;
    pointer-events: none;
}

body::before {
    content: '';
    position: fixed;
    top: 20%;
    left: -300px;
    width: 700px;
    height: 700px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.18) 0%, rgba(59, 130, 246, 0.08) 60%, transparent 80%);
    border-radius: 50%;
    z-index: 0;
    animation: float-massive-3 35s ease-in-out infinite;
    pointer-events: none;
}

@keyframes float-massive-1 {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.9; }
    25% { transform: translate(-100px, 80px) scale(1.1); opacity: 0.7; }
    50% { transform: translate(-50px, -60px) scale(0.95); opacity: 1; }
    75% { transform: translate(80px, 40px) scale(1.05); opacity: 0.8; }
}

@keyframes float-massive-2 {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.8; }
    33% { transform: translate(120px, -80px) scale(1.15); opacity: 0.9; }
    66% { transform: translate(-80px, 60px) scale(0.9); opacity: 0.7; }
}

@keyframes float-massive-3 {
    0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); opacity: 0.7; }
    50% { transform: translate(100px, 50px) rotate(180deg) scale(1.2); opacity: 0.9; }
}

/* ========== MAIN CONTAINER ========== */
.main .block-container {
    padding: 2rem !important;
    max-width: none !important;
    position: relative !important;
    z-index: 100 !important;
}

/* ========== HERO CARD ========== */
.hero-card {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 28px !important;
    padding: 32px 36px !important;
    margin-bottom: 28px !important;
    box-shadow: 0 25px 45px -12px rgba(0, 0, 0, 0.5) !important;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

.hero-card h1 {
    margin: 0 0 12px 0;
    font-size: 50px;
    font-weight: 700;
    background: linear-gradient(135deg, #f8fafc 0%, #8B5CF6 50%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-card p {
    margin: 0;
    color: #94a3b8;
    font-size: 17.5px;
    line-height: 1.5;
}

/* ========== BADGES ========== */
.badge-container {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
    margin-top: 24px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    border-radius: 40px;
    font-size: 13px;
    font-weight: 600;
    backdrop-filter: blur(4px);
    transition: all 0.2s ease;
}

.badge:hover {
    transform: translateY(-2px);
}

.badge.green { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.badge.blue { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.badge.orange { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
.badge.purple { background: rgba(139,92,246,0.15); color: #a78bfa; border: 1px solid rgba(139,92,246,0.3); }
.badge.red { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

/* ========== SEARCH CONTAINER ========== */
.search-container {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    padding: 24px !important;
    margin-bottom: 28px !important;
    position: relative;
    z-index: 100;
}

.search-title {
    font-size: 20px;
    font-weight: 600;
    color: white;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%) !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: white !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    height: 48px !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(139, 92, 246, 0.5) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ========== TEXT INPUT ========== */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 0.75rem 1rem !important;
    color: white !important;
    height: 48px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
}

.stTextInput > div > div > input::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}

/* ========== SELECTBOX ========== */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    color: white !important;
    height: 48px !important;
}

/* ========== METRIC CARDS ========== */
.metric-card {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}

.metric-card:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    transform: translateY(-5px) !important;
    border-color: rgba(139, 92, 246, 0.3) !important;
}

.metric-card h3 {
    margin: 0;
    font-size: 13px;
    color: #94a3b8;
    font-weight: 500;
}

.metric-card h2 {
    margin: 8px 0 0;
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
}

/* ========== COMMENT CARDS ========== */
.comment-card {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin: 10px 0 !important;
    border-right: 4px solid #3b82f6;
    transition: all 0.2s ease;
}

.comment-card:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    transform: translateX(-3px);
}

.comment-card.positive { border-right-color: #22c55e; }
.comment-card.negative { border-right-color: #ef4444; }
.comment-card.neutral  { border-right-color: #9CA3AF; }

.comment-card .author {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 6px;
    font-weight: 500;
}

.comment-card .text {
    font-size: 14px;
    color: #e2e8f0;
    line-height: 1.5;
}

.comment-card .meta {
    font-size: 11px;
    color: #64748b;
    margin-top: 8px;
}

/* ========== VIDEO CARD ========== */
.video-card {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 24px !important;
    padding: 24px !important;
    margin-bottom: 24px !important;
}

/* ========== SECTION TITLE ========== */
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin: 28px 0 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #334155;
    display: inline-block;
}

/* ========== TABS CUSTOM ========== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.05);
    padding: 8px;
    border-radius: 60px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 40px !important;
    padding: 8px 24px !important;
    font-weight: 500 !important;
}

.stTabs [aria-selected="true"] {
    background: #8B5CF6 !important;
    color: white !important;
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0a0f1a 100%) !important;
    border-left: 1px solid rgba(255, 255, 255, 0.05);
}

/* ========== DIVIDER ========== */
hr {
    border-color: #334155 !important;
    margin: 20px 0 !important;
}

/* ========== FOOTER ========== */
footer {
    visibility: hidden;
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: #1e293b;
}

::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #64748b;
}

/* ========== RESPONSIVE ========== */
@media (max-width: 768px) {
    .hero-card h1 {
        font-size: 28px;
    }
    
    .hero-card p {
        font-size: 14px;
    }
    
    .badge {
        padding: 6px 12px;
        font-size: 11px;
    }
    
    .metric-card h2 {
        font-size: 22px;
    }
    
    .section-title {
        font-size: 18px;
    }
}

/* ========== PLOTLY CHARTS FIX ========== */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ========== DATA FRAME ========== */
.stDataFrame {
    background: transparent !important;
}

.dataframe {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# Dynamic RTL / LTR injection based on language toggle
if st.session_state["ui_lang"] == "ar":
    st.markdown("""
<style>
/* RTL mode for main content */
.stApp, .main .block-container, .stMarkdown, p, div, span, h1, h2, h3 {
    direction: rtl !important;
}
/* keep sidebar always LTR */
[data-testid="stSidebar"], [data-testid="stSidebar"] * {
    direction: ltr !important;
}
/* inputs and selects stay LTR internally */
input, textarea, select, [data-baseweb="select"] * {
    direction: ltr !important;
}
</style>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<style>
.stApp, .main .block-container {
    direction: ltr !important;
}
</style>
""", unsafe_allow_html=True)

# ============
# HERO SECTION
# ============

_hero_title    = t(" YouTube Comment Sentiment Analyzer PRO 🎯",
                   "🎯 محلل مشاعر تعليقات يوتيوب")
_hero_subtitle = t(
    "Smart sentiment analysis for YouTube comments using advanced AI models: "
    "MARBERT for Arabic and RoBERTa for English, with automatic spam and sarcasm detection.",
    "تحليل ذكي لمشاعر تعليقات يوتيوب باستخدام نماذج الذكاء الاصطناعي المتقدمة: "
    "MARBERT للعربية و RoBERTa للإنجليزية، مع كشف تلقائي للسبام والسخرية."
)
_dir_attr = 'dir="rtl"' if st.session_state["ui_lang"] == "ar" else ''

st.markdown(f"""
<div class="hero-card" {_dir_attr}>
    <h1>{_hero_title}</h1>
    <p>{_hero_subtitle}</p>
    <div class="badge-container">
        <span class="badge green">🤖 MARBERT Active</span>
        <span class="badge blue">📊 RoBERTa Active</span>
        <span class="badge orange">🚨 {t("Spam Detection","كشف السبام")}</span>
        <span class="badge purple">😏 {t("Sarcasm Detection","كشف السخرية")}</span>
        <span class="badge red">🎨 Emoji Boost</span>
    </div>
</div>
""", unsafe_allow_html=True)
#================
#  Loading models
#================
@st.cache_resource(show_spinner=False)
def load_arabic_model():
    tokenizer = AutoTokenizer.from_pretrained(AR_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(AR_MODEL)
    model.eval()
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True,
        max_length=256,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=32)

@st.cache_resource(show_spinner=False)
def load_english_model():
    return pipeline(
        "sentiment-analysis",
        model=EN_MODEL,
        truncation=True,
        max_length=256,
        device=0 if torch.cuda.is_available() else -1,
        batch_size=32)

# ====================
# Extracting Video ID
# ====================

from urllib.parse import urlparse, parse_qs

def extract_video_id(url: str) -> str:

    url = str(url).strip()    
    if url.startswith("AIza"):
        return "ERROR:apikey"

    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url

    try:

        parsed = urlparse(url)

        if "youtube.com" in parsed.netloc:
            query = parse_qs(parsed.query)

            if "v" in query:
                return query["v"][0]

            path_parts = parsed.path.split("/")

            for i, part in enumerate(path_parts):
                if part in {"shorts", "embed", "live"}:
                    if i + 1 < len(path_parts):
                        return path_parts[i + 1]

        if "youtu.be" in parsed.netloc:
            return parsed.path.strip("/")

    except Exception:
        pass

    return "ERROR:invalid"


# ======================
# Get video information
# ======================
def get_video_info(video_id: str) -> tuple[dict, str | None]:
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        r = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()

        if not r.get("items"):
            return {}, "الفيديو غير موجود أو الرابط خاطئ."

        item    = r["items"][0]
        snippet = item["snippet"]
        stats   = item["statistics"]

        thumbnail = (
            snippet.get("thumbnails", {})
            .get("maxres", snippet.get("thumbnails", {}).get("high", {}))
            .get("url", "")
        )

        return {
            "title":        snippet.get("title", ""),
            "channel":      snippet.get("channelTitle", ""),
            "published":    snippet.get("publishedAt", ""),
            "thumbnail":    thumbnail,
            "views":        int(stats.get("viewCount",   0)),
            "likes":        int(stats.get("likeCount",   0)),
            "comments":     int(stats.get("commentCount", 0)),
        }, None

    except HttpError as e:
        msg = str(e)
        if "keyInvalid" in msg or "API key" in msg:
            return {}, "Invalid YouTube API Key ."
        if "videoNotFound" in msg or "404" in msg:
            return {}, "The video is not available"
        if "quotaExceeded" in msg:
            return {}, "The daily limit for the API has been exceeded (quota exceeded)"
        return {}, f"Error from YouTube API: {e.reason}"
    except Exception as e:
        return {}, f"Unexpected Error: {e}"

# ===============
# Get Comments
# ===============
@st.cache_data(ttl=3600)
def get_comments(video_id: str,max_comments: int | None,order: str,include_replies: bool = False,) -> tuple[pd.DataFrame, str | None]:
    try:
        youtube         = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        comments: list[dict] = []
        next_page_token: str | None = None
        page_size = 100

        while True:
            req_kwargs = dict(
                part="snippet,replies" if include_replies else "snippet",videoId=video_id,maxResults=page_size,textFormat="plainText",order=order,)
            if next_page_token:
                req_kwargs["pageToken"] = next_page_token

            response = youtube.commentThreads().list(**req_kwargs).execute()
            for item in response.get("items", []):
                top_sn = item["snippet"]["topLevelComment"]["snippet"]
                reply_count = int(item["snippet"].get("totalReplyCount", 0))
                comments.append({
                    "comment_id":  item["id"],
                    "author":      top_sn.get("authorDisplayName", ""),
                    "text":        top_sn.get("textDisplay", ""),
                    "likes":       int(top_sn.get("likeCount", 0)),
                    "publishedAt": top_sn.get("publishedAt", ""),
                    "replies":     reply_count,
                    "text_length": len(top_sn.get("textDisplay", "")),
                    "is_reply":    False,
                })
                if include_replies and reply_count > 0:
                    reply_items = (
                        item.get("replies", {}).get("comments", [])
                    )
                    for rep in reply_items:
                        rsn = rep.get("snippet", {})
                        comments.append({
                            "comment_id":  rep.get("id", ""),
                            "author":      rsn.get("authorDisplayName", ""),
                            "text":        rsn.get("textDisplay", ""),
                            "likes":       int(rsn.get("likeCount", 0)),
                            "publishedAt": rsn.get("publishedAt", ""),
                            "replies":     0,
                            "text_length": len(rsn.get("textDisplay", "")),
                            "is_reply":    True,
                        })
                if max_comments and len(comments) >= max_comments:
                    break

            if max_comments and len(comments) >= max_comments:
                break
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        if not comments:
            return pd.DataFrame(), "There are no comments or the comments are closed"

        df = pd.DataFrame(comments)
        if max_comments:
            df = df.head(max_comments)
        return df, None

    except HttpError as e:
        msg = str(e)
        if "commentsDisabled" in msg:
            return pd.DataFrame(), "Comments are closed on this video"
        if "quotaExceeded" in msg or "rateLimitExceeded" in msg:
            return pd.DataFrame(), "YouTube API quota exceeded. Please try again tomorrow or reduce the number of comments requested."
        if "keyInvalid" in msg or "API key" in msg:
            return pd.DataFrame(), "Invalid YouTube API key. Please check your .env file."
        if "videoNotFound" in msg or "404" in msg:
            return pd.DataFrame(), "Video not found or unavailable."
        if "forbidden" in msg.lower() or "403" in str(e.status_code if hasattr(e, 'status_code') else ""):
            return pd.DataFrame(), "Access denied. The video may be age-restricted or private."
        return pd.DataFrame(), f"YouTube API error: {getattr(e, 'reason', str(e))}"
    except Exception as e:
        return pd.DataFrame(), f"Unexpected error while fetching comments: {e}"

# =====================================
# Text cleaning (while preserving emojis)
# =====================================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[أإآأ]", "ا", text)
    text = re.sub(r"ة",       "ه", text)
    text = re.sub(r"[يى]",   "ي", text)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(
        r"[^\u0600-\u06FFa-zA-Z0-9\s"
        r"\U0001F300-\U0001FAFF"
        r"\U00002700-\U000027BF"
        r"\U00002600-\U000026FF"
        r"\u2600-\u26FF]",
        " ",text,)

    return re.sub(r"\s+", " ", text).strip()

# ==================
# Language Detection
# ==================
def detect_language(text: str) -> str:

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\S+", "", text)
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    english = len(re.findall(r"[a-zA-Z]", text))
    total = arabic + english

    if total == 0:
        return "other"
    ratio = arabic / total
    if ratio >= 0.7:
        return "arabic"
    if ratio <= 0.25:
        return "english"
    
    return "mixed"

# ==============
# Spam detection
# ==============
_LINK_RE        = re.compile(r"https?://\S+|www\.\S+", re.I)
_PHONE_RE       = re.compile(r"(?<!\d)\d{10,}(?!\d)")
_WHATSAPP_RE    = re.compile(r"(?:whatsapp|واتس\s*اب|واتساب)[^\d]*(\+?\d[\d\s\-]{7,})", re.I)
_ALL_CAPS_RE    = re.compile(r"\b[A-Z]{5,}\b")
_PROMO_RE       = re.compile(
    r"\b(check\s+(?:my|out\s+my)|visit\s+my\s+channel|sub(?:scribe)?\s+(?:to\s+)?my|"
    r"link\s+in\s+(?:bio|description)|زور\s+قناتي|تابع\s+قناتي|اشترك\s+في\s+قناتي)\b",re.I)

SPAM_EN_STRONG = {
    "buy now", "click here", "earn money online", "dm me for", "check my channel",
    "visit my channel", "link in bio", "link in description",
    "make money", "earn $", "100% free", "limited offer", "act now",
    "whatsapp me", "join telegram", "forex signals", "crypto signals",
    "binary options", "investment opportunity", "giveaway winner",
    "you have been selected", "congratulations you won"}

SPAM_EN_WEAK = {
    "subscribe", "follow me", "giveaway", "free", "crypto", "bitcoin",
    "forex", "telegram", "promotion", "invest"}

SPAM_AR_STRONG = {
    "اكسب المال", "ربح المال", "زور قناتي", "تابع قناتي",
    "اشترك في قناتي", "رابط في البايو", "رابط في الوصف",
    "استثمار مضمون", "ربح مضمون", "فرصة استثمار",
    "رابح 100%", "فرصة ذهبية", "تواصل معي واتساب"}

SPAM_AR_WEAK = {
    "اشترك", "تابعني", "جائزة", "واتساب", "تيليجرام",
    "تداول", "استثمار", "كريبتو"}

def detect_spam(text: str) -> tuple[bool, str]:
    raw = str(text).strip()
    if not raw:
        return False, ""

    t_lower = raw.lower()
    word_count = len(t_lower.split())

    # Suspicious links
    links = _LINK_RE.findall(t_lower)
    if len(links) >= 2:
        return True, "Multiple links"
    if len(links) == 1 and word_count <= 6:
        # Short comment that is mostly a link is promotional
        return True, "Suspicious link"

    # WhatsApp / phone numbers 
    if _WHATSAPP_RE.search(raw):
        return True, "WhatsApp number"
    if _PHONE_RE.search(t_lower):
        return True, "Phone number"

    # Strong-signal promotional pattern (regex)
    if _PROMO_RE.search(t_lower):
        return True, "Promotional comment"

    # Strong keyword hits (one hit is enough)
    for phrase in SPAM_EN_STRONG:
        if phrase in t_lower:
            return True, "Spam keywords"
    for phrase in SPAM_AR_STRONG:
        if phrase in t_lower:
            return True, "Spam keywords"

    # Weak keywords — need 2+ to fire
    weak_hits = 0
    for word in SPAM_EN_WEAK:
        if re.search(rf"\b{re.escape(word)}\b", t_lower):
            weak_hits += 1
    for word in SPAM_AR_WEAK:
        if word in t_lower:
            weak_hits += 1
    if weak_hits >= 3:
        return True, "Multiple spam keywords"

    # Excessive emojis
    emoji_count = len(re.findall(r"[\U0001F300-\U0001FAFF]", raw))
    if emoji_count >= 12:
        return True, "Excessive emojis"

    # Meaningless character repetition (not normal laughter)
    if re.search(r"(?!ه{3,}|ha{3,}|هه{3,})(.)\1{12,}", t_lower):
        return True, "Excessive repeated characters"

    # Very low word-diversity in longer texts
    words = t_lower.split()
    if len(words) >= 15:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.30:
            return True, "Repeated content"

    # ALL-CAPS shouting (≥ 4 all-caps words in a short text)
    caps_words = _ALL_CAPS_RE.findall(raw)
    if len(caps_words) >= 4 and word_count <= 12:
        return True, "Excessive caps"

    return False, ""

# ==============================
# Sarcasm detection (rule-based)
# ==============================
def detect_sarcasm(text: str) -> bool:
    text = str(text)
    t = text.lower()
    sarcasm_score = 0

    for emoji in SARCASM_EMOJIS:
        if emoji in text:
            sarcasm_score += 1

    for phrase in SARCASM_PHRASES:
        if phrase.lower() in t:
            sarcasm_score += 2

    if "!!!" in text:
        sarcasm_score += 1

    if "???" in text:
        sarcasm_score += 1

    positive_words = {"amazing","great","perfect","رائع","ممتاز","مبدع"}
    negative_words = {"سيء","فاشل","زبالة","bad","terrible","awful"}
    has_positive = any(w in t for w in positive_words)
    has_negative = any(w in t for w in negative_words)

    if has_positive and has_negative:
        sarcasm_score += 2

    if re.search(r"(هه){3,}", t):
        sarcasm_score += 1

    if re.search(r"(ha){3,}", t):
        sarcasm_score += 1

    return sarcasm_score >= 3

# ====================
# Label normalization
# ====================
LABEL_MAP = {
    "positive": "positive", "negative": "negative", "neutral": "neutral",
    "label_0":  "negative", "label_1":  "neutral",  "label_2": "positive",
    "pos":      "positive", "neg":      "negative",  "neu":     "neutral",}

def normalize_label(raw: str) -> str:
    return LABEL_MAP.get(str(raw).lower().strip(), "neutral")

# ============
# Emoji Boost
# ============
def emoji_boost(text: str, sentiment: str, score: float, is_engagement: bool = False) -> tuple[str, float]:

    if is_engagement:
        return sentiment, score

    pos_count = sum(text.count(e) for e in POSITIVE_EMOJIS)
    neg_count = sum(text.count(e) for e in NEGATIVE_EMOJIS)

    if sentiment == "positive":
        if pos_count >= 3:
            score = min(score + 0.06, 0.98)
        elif pos_count > 0:
            score = min(score + 0.03, 0.95)
        if neg_count >= 2:
            score = max(score - 0.05, 0.0)

    elif sentiment == "negative":
        if neg_count >= 3:
            score = min(score + 0.06, 0.98)
        elif neg_count > 0:
            score = min(score + 0.03, 0.95)

    elif sentiment == "neutral":
        if pos_count >= 4 and neg_count == 0:
            sentiment = "positive"
            score = max(score, 0.60)
        elif neg_count >= 4 and pos_count == 0:
            sentiment = "negative"
            score = max(score, 0.60)

    return sentiment, round(min(score, 1.0), 3)

def is_engagement_comment(text: str) -> bool:
    text = text.lower().strip()
    patterns = ["2024","2025","2026","2027","من يتابع","مين يتابع","من يشاهد","مين يشاهد","من موجود","مين موجود","من هنا",
                "مين هنا","لسه يتابع","لسا يتابع","لسا موجود","لسه موجود","مين جاي من","من جاي من","مين جاء من","من جاء من",
                "مين يسمع","من يسمع","مين يتذكر","من يتذكر","مين رجع","من رجع","مين حضر","من حضر","مين لسه","من لسه",
                "حد هون","حد هنا","في حد","في احد","مين متابع","من متابع","مين صاحي","من صاحي","مين يشوف","من يشوف",
                "مين يشاهد في","من يشاهد في","اللي جاى من","الي جاي من","الجاى","الجاي","who is watching","anyone watching",
                "still watching","watching in","who is here","anyone here","still here","who came from","who remembers","anyone remember",
                "who's listening","who is listening","who is still listening","still listening","who's still here","watching this in",
                "listening in","anyone in","who else","am i the only one","is anyone else","who is watching this in"]
    return any(p in text for p in patterns)

# =================
# Emotion Analysis
# =================
def analyze_sentiment(df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
    if df.empty:
        return df

    ar_model = load_arabic_model()
    en_model = load_english_model()

    df = df.copy()
    df["clean_text"]  = df["text"].apply(clean_text)
    df["language"]    = df["clean_text"].apply(detect_language)

    spam_results      = df["text"].apply(detect_spam)
    df["is_spam"]     = spam_results.apply(lambda x: x[0])
    df["spam_reason"] = spam_results.apply(lambda x: x[1])
    df["is_sarcasm"]  = df["text"].apply(detect_sarcasm)
    df["_is_engage"]  = df["clean_text"].apply(is_engagement_comment)

    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    results_map: dict[int, tuple[str, float]] = {}
    total = len(df)
    bar   = st.progress(0, text="Analysis in progress ...")

    arabic_idx  = df.index[df["language"] == "arabic"].tolist()
    english_idx = df.index[df["language"].isin(["english", "other"])].tolist()
    mixed_idx   = df.index[df["language"] == "mixed"].tolist()

    if arabic_idx:
        texts = [str(df.at[i, "clean_text"])[:512] for i in arabic_idx]
        try:
            with torch.no_grad():
                batch_results = ar_model(texts)
            for i, result in zip(arabic_idx, batch_results):
                if isinstance(result, list):
                    inner = result[0] if isinstance(result[0], list) else result
                    sm = {normalize_label(x["label"]): float(x["score"]) for x in inner}
                else:
                    sm = {"neutral": 1.0}
                best = max(sm, key=sm.get)
                results_map[i] = (best if sm[best] >= confidence_threshold else "neutral",
                                  sm[best])
        except Exception:
            for i in arabic_idx:
                results_map[i] = ("neutral", 0.5)

    if english_idx:
        texts = [str(df.at[i, "clean_text"])[:512] for i in english_idx]
        try:
            batch_results = en_model(texts)
            for i, result in zip(english_idx, batch_results):
                if isinstance(result, dict):
                    label = normalize_label(result["label"])
                    score = float(result["score"])
                    results_map[i] = (label if score >= confidence_threshold else "neutral", score)
                else:
                    results_map[i] = ("neutral", 0.5)
        except Exception:
            for i in english_idx:
                results_map[i] = ("neutral", 0.5)

    if mixed_idx:
        mixed_texts = [str(df.at[i, "clean_text"])[:512] for i in mixed_idx]
        ar_mixed, en_mixed = {}, {}
        try:
            with torch.no_grad():
                ar_batch = ar_model(mixed_texts)
            for i, result in zip(mixed_idx, ar_batch):
                inner = result[0] if isinstance(result[0], list) else result
                sm = {normalize_label(x["label"]): float(x["score"]) for x in inner}
                best = max(sm, key=sm.get)
                ar_mixed[i] = (best, sm[best])
        except Exception:
            for i in mixed_idx:
                ar_mixed[i] = ("neutral", 0.5)
        try:
            en_batch = en_model(mixed_texts)
            for i, result in zip(mixed_idx, en_batch):
                if isinstance(result, dict):
                    en_mixed[i] = (normalize_label(result["label"]), float(result["score"]))
                else:
                    en_mixed[i] = ("neutral", 0.5)
        except Exception:
            for i in mixed_idx:
                en_mixed[i] = ("neutral", 0.5)

        for i in mixed_idx:
            ar_sent, ar_score = ar_mixed.get(i, ("neutral", 0.5))
            en_sent, en_score = en_mixed.get(i, ("neutral", 0.5))
            if ar_sent == en_sent:
                results_map[i] = (ar_sent, round((ar_score + en_score) / 2, 3))
            elif ar_score >= en_score:
                results_map[i] = (ar_sent, ar_score)
            else:
                results_map[i] = (en_sent, en_score)

    sentiments: list[str] = []
    scores: list[float]   = []

    for idx in range(total):
        bar.progress((idx + 1) / total, text=f"Analyzing {idx + 1}/{total}...")

        if df.at[idx, "is_spam"]:
            sentiments.append("neutral")
            scores.append(1.0)
            continue

        sentiment, score = results_map.get(idx, ("neutral", 0.5))
        is_engage = bool(df.at[idx, "_is_engage"])

        if is_engage:
            sentiment = "neutral"
            score     = max(score, 0.80)

        sentiment, score = emoji_boost(
            str(df.at[idx, "text"]), sentiment, score, is_engagement=is_engage
        )

        if df.at[idx, "is_sarcasm"] and sentiment == "positive":
            score = max(score - 0.20, 0.0)
            if score < confidence_threshold:
                sentiment = "neutral"

        if score < confidence_threshold:
            sentiment = "neutral"

        sentiments.append(sentiment)
        scores.append(round(score, 3))

    bar.empty()
    df["sentiment"] = sentiments
    df["score"]     = scores
    df.drop(columns=["_is_engage"], inplace=True, errors="ignore")
    return df

# ==================
# Summarize Results
# ==================
def summarize_results(df: pd.DataFrame) -> dict:
    total   = len(df)
    pos     = int((df["sentiment"] == "positive").sum())
    neg     = int((df["sentiment"] == "negative").sum())
    neu     = int((df["sentiment"] == "neutral").sum())
    spam    = int(df["is_spam"].sum())
    sarc    = int(df["is_sarcasm"].sum())
    avg_sc  = round(float(df["score"].mean()), 3)
    lang_counts = df["language"].value_counts().to_dict()
    avg_likes = round(float(df["likes"].mean()), 2)
    max_likes = int(df["likes"].max()) if total else 0
    most_liked_comment = ""
    if total:
        top_idx = df["likes"].idxmax()
        most_liked_comment = str(df.loc[top_idx, "text"])[:150]
    return {
        "total":   total,
        "pos":     pos, "pos_pct": round(pos / total * 100, 1) if total else 0,
        "neg":     neg, "neg_pct": round(neg / total * 100, 1) if total else 0,
        "neu":     neu, "neu_pct": round(neu / total * 100, 1) if total else 0,
        "spam":    spam,
        "sarcasm": sarc,
        "avg_confidence": avg_sc,
        "arabic":  lang_counts.get("arabic",  0),
        "english": lang_counts.get("english", 0),
        "mixed":   lang_counts.get("mixed",   0),
        "other":   lang_counts.get("other",   0),
        "spam_pct": round(spam / total * 100, 1) if total else 0,
        "sarcasm_pct": round(sarc / total * 100, 1) if total else 0,
        "avg_likes": avg_likes,
        "max_likes": max_likes,
        "top_comment": most_liked_comment,}

def generate_ai_insights(summary, df):
    insights = []
    if summary["pos_pct"] > 75:
        insights.append("🔥 The audience reaction was extremely positive overall.")

    elif summary["pos_pct"] > 60:
        insights.append("✅ Most viewers reacted positively to the video.")

    elif summary["neg_pct"] > 50:
        insights.append("⚠️ The video received a significant amount of negative feedback.")

    else:
        insights.append("📊 Audience reactions were relatively balanced.")

    avg_likes = df["likes"].mean()
    if avg_likes > 50:
        insights.append("🚀 Comments showed very high engagement levels.")

    elif avg_likes > 10:
        insights.append("👍 Viewer engagement was moderate.")

    else:
        insights.append("👀 Most comments received low engagement.")

    spam_ratio = (summary["spam"] / summary["total"]) * 100
    if spam_ratio > 15:
        insights.append("🚨 A high percentage of comments were classified as spam.")

    elif spam_ratio > 5:
        insights.append("⚠️ Some suspicious or promotional comments were detected.")

    sarcasm_ratio = (summary["sarcasm"] / summary["total"]) * 100
    if sarcasm_ratio > 10:
        insights.append("😏 Sarcastic reactions appeared frequently in the comments.")

    elif sarcasm_ratio > 3:
        insights.append("🙃 A small portion of comments contained sarcasm.")

    if summary["arabic"] > summary["english"]:
        insights.append("🌍 Arabic-speaking viewers dominated the conversation.")

    elif summary["english"] > summary["arabic"]:
        insights.append("🌎 English comments represented the majority of interactions.")

    avg_pos_likes = df[df["sentiment"] == "positive"]["likes"].mean()
    avg_neg_likes = df[df["sentiment"] == "negative"]["likes"].mean()

    if avg_pos_likes > avg_neg_likes * 1.5:
        insights.append("💚 Positive comments attracted significantly more engagement.")

    elif avg_neg_likes > avg_pos_likes * 1.5:
        insights.append("💥 Negative comments generated stronger audience interaction.")

    if summary["avg_confidence"] > 0.85:
        insights.append("🎯 The AI model showed very high confidence in sentiment predictions.")

    elif summary["avg_confidence"] < 0.60:
        insights.append("🤔 Some comments were difficult for the AI to classify accurately.")

    total_engagement = df["likes"].sum()

    if total_engagement > 5000:
        insights.append("📈 The video generated exceptionally high audience interaction.")

    elif total_engagement > 1000:
        insights.append("📊 The content achieved strong engagement performance.")

    if summary["neg_pct"] > 35 and sarcasm_ratio > 8:
        insights.append("☠️ The discussion showed signs of heated or toxic interactions.")

    if summary["pos_pct"] > 70 and spam_ratio < 5:
        insights.append("✨ The community response appeared healthy and genuinely positive.")

    return insights

def create_sentiment_chart(summary):

    labels = ["Positive", "Negative", "Neutral"]
    values = [
        summary["pos_pct"],
        summary["neg_pct"],
        summary["neu_pct"]]

    plt.figure(figsize=(5,5))
    plt.pie(values,labels=labels,autopct='%1.1f%%')

    chart_path = "sentiment_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return chart_path

def create_bar_chart(summary):

    labels = ["Positive", "Negative", "Neutral"]
    values = [summary["pos_pct"],summary["neg_pct"],summary["neu_pct"]]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    chart_path = "bar_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return chart_path

def create_language_chart(summary): 
    labels = ["Arabic", "English", "Mixed", "Other"] 
    values = [ summary["arabic"], 
              summary["english"], 
              summary["mixed"], 
              summary["other"] ] 
    plt.figure(figsize=(5,5)) 
    plt.pie( values, labels=labels, autopct='%1.1f%%' ) 
    chart_path = "language_chart.png" 
    plt.savefig(chart_path) 
    plt.close() 
    return chart_path


def generate_pdf_report(video_info, summary, insights):

    pdf_file = "report.pdf"
    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(
            f"YouTube Analysis Report: {video_info['title']}",
            styles['Title']))

    elements.append(Spacer(1, 20))
    chart_path = create_sentiment_chart(summary)
    elements.append(
    Paragraph("Sentiment Distribution", styles['Heading2']))
    elements.append(
        Image(chart_path, width=300, height=300))

    elements.append(
        Paragraph(
            f"Positive Comments: {summary['pos_pct']}%",
            styles['BodyText']))

    elements.append(
        Paragraph(
            f"Negative Comments: {summary['neg_pct']}%",
            styles['BodyText']))

    elements.append(
        Paragraph(
            f"Neutral Comments: {summary['neu_pct']}%",
            styles['BodyText']))

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph("AI Insights", styles['Heading2']))

    for insight in insights:
        elements.append(
            Paragraph(f"• {insight}", styles['BodyText']))
        
    bar_chart = create_bar_chart(summary) 
    elements.append(Spacer(1, 20)) 
    elements.append( Paragraph( "Sentiment Comparison", styles['Heading2'] ) ) 
    elements.append( Image(bar_chart, width=400, height=250) )
    language_chart = create_language_chart(summary) 
    elements.append(Spacer(1, 20)) 
    elements.append( Paragraph( "Language Distribution", styles['Heading2'] ) ) 
    elements.append( Image(language_chart, width=300, height=300) )
    doc.build(elements)

    return pdf_file

# ===========================
# Most frequently used words
# ===========================
COMMON_NOISE_WORDS = {
    "youtube", "video", "videos", "channel", "comment", "comments",
    "watching", "viewer", "watch", "like", "liked", "subscribe",
    "lol", "lmao", "haha", "omg", "wow", "yeah", "yep", "ok", "okay",
    "one", "two", "three", "good", "great", "nice", "make",
    "فيديو", "الفيديو", "يوتيوب", "القناة", "قناة", "تعليق", "التعليق",
    "مشاهد", "متابع", "شاهد", "يشاهد", "يتابع", "تابع","هههه", "ههههه",
    "هههههه", "هههههههه","جميل", "حلو", "تمام", "مرحبا", "اهلا"}

_AR_NORM = str.maketrans("أإآاى", "ااااي")

def _normalize_ar(word: str) -> str:
    word = word.translate(_AR_NORM)
    word = re.sub(r"ة$", "ه", word)
    word = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", word)  # strip diacritics
    return word

def get_top_words(df: pd.DataFrame, sentiment: str, top_n: int = 20) -> list[tuple]:
    subset = df[
        (df["sentiment"] == sentiment) & (~df["is_spam"])
    ]["clean_text"].dropna()

    word_freq: Counter = Counter()

    for text in subset:
        for raw_word in str(text).split():
            word = re.sub(r"[^\w\u0600-\u06FF]", "", raw_word).lower().strip()
            if not word:
                continue
            if re.fullmatch(r"[\d._]+", word):
                continue
            is_arabic_word = bool(re.search(r"[\u0600-\u06FF]", word))
            min_len = 3 if is_arabic_word else 3
            if len(word) < min_len:
                continue
            if word in STOP_WORDS:
                continue
            if word in COMMON_NOISE_WORDS:
                continue

            key = _normalize_ar(word) if is_arabic_word else word
            if key in COMMON_NOISE_WORDS:
                continue

            word_freq[key] += 1

    return word_freq.most_common(top_n)

# ===============
# Topic Detection
# ===============

def get_topics(df: pd.DataFrame, sentiment: str) -> dict[str, int]:
    subset = (
        df[(df["sentiment"] == sentiment) & (~df["is_spam"])]["clean_text"].dropna().tolist())
    topics: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        count = 0
        for text in subset:
            text_lower = str(text).lower()
            found = False
            for kw in keywords:
                kw_lower = kw.lower()
                if re.search(r"[a-zA-Z]", kw_lower):
                    pattern = rf"\b{re.escape(kw_lower)}\b"
                else:
                    pattern = re.escape(kw_lower)
                if re.search(pattern, text_lower):
                    found = True
                    break
            if found:
                count += 1
        if count > 0:
            topics[topic] = count
    return dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))

# =============
# Card Comment
# =============
def comment_card(row: pd.Series) -> str:
    sentiment = row.get("sentiment", "neutral")
    author    = row.get("author",    "مجهول")
    text      = row.get("text",      "")
    likes     = row.get("likes",     0)
    score     = row.get("score",     0)
    return f"""
    <div class="comment-card {sentiment}">
        <div class="author">👤 {author}</div>
        <div class="text">{text}</div>
        <div class="meta">❤️ {likes} إعجاب &nbsp;|&nbsp; ثقة: {score:.0%}</div>
    </div>"""

# =============
# Tab: Overview
# =============

def render_overview(df: pd.DataFrame, summary: dict, video_info: dict):
    st.markdown("""
<div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">🎥</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Video Information</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_thumb, col_meta = st.columns([1, 3])
    with col_thumb:
        if video_info.get("thumbnail"):
            st.image(video_info["thumbnail"], use_container_width=True)
    with col_meta:
        st.markdown(f"<h2 style='margin-bottom: 8px; color: white;'>{video_info.get('title','')}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #94a3b8;'>📺 <strong>{video_info.get('channel','')}</strong> &nbsp;|&nbsp; 📅 {video_info.get('published','')[:10]}</p>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👁️ Views", f"{video_info.get('views',0):,}")
        c2.metric("👍 Likes", f"{video_info.get('likes',0):,}")
        c3.metric("💬 Comments", f"{video_info.get('comments',0):,}")
        c4.metric("🔍 Analyzed", f"{summary['total']:,}")

    st.divider()
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📈</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Analysis Summary</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    row1 = st.columns(4)
    row2 = st.columns(4)
    
    with row1[0]:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h2>😊 Positive Comments</h2>
            <h1>{summary['pos_pct']}%</h1>
            <p style="color:#64748b; font-size:17px;">{summary['pos']} Comments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row1[1]:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h2>😡 Negative Comments</h2>
            <h1>{summary['neg_pct']}%</h1>
            <p style="color:#64748b; font-size:17px;">{summary['neg']} Comment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row1[2]:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h2>😐 Neutral Comments</h2>
            <h1>{summary['neu_pct']}%</h1>
            <p style="color:#64748b; font-size:17px;">{summary['neu']} Comment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with row1[3]:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h2>🎯  Average Confidence</h2>
            <h1>{summary['avg_confidence']:.0%}</h1>
            <p style="color:#64748b; font-size:17px;">Accuracy </p>
        </div>
        """, unsafe_allow_html=True)
    
    with row2[0]:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h2>🚨 Spam Comments</h2>
            <h1>{summary['spam']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with row2[1]:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h2>🙄 Sarcasm Comments</h2>
            <h1>{summary['sarcasm']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with row2[2]:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h2>Arabic Comments</h2>
            <h1>{summary['arabic']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with row2[3]:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h2>English Comments</h2>
            <h1>{summary['english']}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
<div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📊</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Graphs</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    counts_df = (
        df["sentiment"].value_counts()
        .reset_index()
        .rename(columns={"index": "sentiment", "sentiment": "count"})
    )
    if "sentiment" not in counts_df.columns or len(counts_df.columns) == 1:
        counts_df.columns = ["sentiment", "count"]

    col_pie, col_bar = st.columns(2)
    
    with col_pie:
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=counts_df["sentiment"],
            values=counts_df["count"],
            hole=0.55,
            marker=dict(
                colors=[COLOR_MAP.get(lbl, "#9CA3AF") for lbl in counts_df["sentiment"]],
                line=dict(color='rgba(255,255,255,0.15)', width=2)
            ),
            textinfo='percent',
            textposition='inside',
            textfont=dict(size=14, color='white'),
            hoverinfo='label+value+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>',
            pull=[0.02, 0.02, 0.02],
            showlegend=True
        ))
        
        fig.add_annotation(
            text=f"<b>{len(df):,}</b><br>Total",
            x=0.5, y=0.5,
            font=dict(size=16, color='white'),
            showarrow=False,
            align='center'
        )
        
        fig.update_layout(
            title=dict(
                text="<b>Sentiment Distribution</b>",
                font=dict(size=16, color='white'),
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            legend=dict(
                font=dict(color='white', size=11),
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
                x=0.82,
                y=0.5
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        fig2 = px.bar(
            counts_df, x="sentiment", y="count",
            color="sentiment", 
            color_discrete_map=COLOR_MAP,
            title="<b>Comments Count</b>",
            text="count",
            labels={"sentiment": "Sentiment", "count": "Number of Comments"}
        )
        
        fig2.update_traces(
            textposition='outside',
            textfont=dict(size=13, color='white'),
            marker=dict(
                line=dict(color='rgba(255,255,255,0.2)', width=1.5),
                cornerradius=8,
                opacity=0.9
            ),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )
        
        fig2.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=16, color='white'),
            font=dict(color='white'),
            xaxis=dict(
                title="Sentiment",
                title_font=dict(size=13, color='#94a3b8'),
                tickfont=dict(size=12, color='white'),
                showgrid=False
            ),
            yaxis=dict(
                title="Number of Comments",
                title_font=dict(size=13, color='#94a3b8'),
                tickfont=dict(size=11, color='#94a3b8'),
                gridcolor='rgba(255,255,255,0.08)',
                gridwidth=0.5
            ),
            height=450,
            margin=dict(t=50, b=50, l=50, r=30)
        )
        
        st.plotly_chart(fig2, use_container_width=True)

    # Language Distribution Chart
    st.markdown("""

    """, unsafe_allow_html=True)
    
    lang_df = (
        df["language"].value_counts()
        .reset_index()
        .rename(columns={"index": "language", "language": "count"})
    )
    if "language" not in lang_df.columns or len(lang_df.columns) == 1:
        lang_df.columns = ["language", "count"]

    # Language names
    lang_names = {
        "arabic": "Arabic",
        "english": "English",
        "mixed": "Mixed",
        "other": "Other"
    }
    lang_df["language_display"] = lang_df["language"].map(lambda x: lang_names.get(x, x))
    
    fig_lang = px.bar(
        lang_df, x="language_display", y="count", 
        title="<b>Language Distribution</b>",
        text="count",
        color_discrete_sequence=["#8B5CF6"],
        labels={"language_display": "Language", "count": "Number of Comments"}
    )
    
    fig_lang.update_traces(
        textposition='outside',
        textfont=dict(size=13, color='white'),
        marker=dict(
            line=dict(color='rgba(255,255,255,0.15)', width=1.5),
            cornerradius=8,
            opacity=0.85
        )
    )
    
    fig_lang.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=16, color='white'),
        font=dict(color='white'),
        xaxis=dict(
            title="Language",
            title_font=dict(size=13, color='#94a3b8'),
            tickfont=dict(size=12, color='white'),
            showgrid=False
        ),
        yaxis=dict(
            title="Number of Comments",
            title_font=dict(size=13, color='#94a3b8'),
            tickfont=dict(size=11, color='#94a3b8'),
            gridcolor='rgba(255,255,255,0.08)',
            gridwidth=0.5
        ),
        height=400,
        margin=dict(t=50, b=50, l=50, r=30),
        showlegend=False
    )
    
    st.plotly_chart(fig_lang, use_container_width=True)

    # Timeline Chart
    if "publishedAt" in df.columns and df["publishedAt"].notna().any():
        try:
            st.markdown("""
<div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">⏱️</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Sentiment Over Time</div>
        </div>
    </div>
    </div>
            """, unsafe_allow_html=True)
            
            df2 = df.copy()
            df2["date"] = pd.to_datetime(df2["publishedAt"], errors="coerce").dt.date
            timeline = (
                df2.groupby(["date", "sentiment"])
                .size()
                .reset_index(name="count")
            )
            
            fig_t = px.line(
                timeline, 
                x="date", 
                y="count",
                color="sentiment",
                color_discrete_map=COLOR_MAP,
                title="<b>Sentiment Over Time</b>",
                labels={"date": "Date", "count": "Number of Comments", "sentiment": "Sentiment"}
            )
            
            fig_t.update_traces(
                mode='lines+markers',
                line=dict(width=2.5),
                marker=dict(size=6, symbol='circle'),
                fill='tozeroy',
                fillcolor='rgba(139,92,246,0.1)'
            )
            
            fig_t.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font=dict(size=16, color='white'),
                font=dict(color='white'),
                xaxis=dict(
                    title="Date",
                    title_font=dict(size=13, color='#94a3b8'),
                    tickfont=dict(size=10, color='#94a3b8'),
                    gridcolor='rgba(255,255,255,0.05)',
                    tickangle=-45
                ),
                yaxis=dict(
                    title="Number of Comments",
                    title_font=dict(size=13, color='#94a3b8'),
                    tickfont=dict(size=11, color='#94a3b8'),
                    gridcolor='rgba(255,255,255,0.08)',
                    gridwidth=0.5
                ),
                legend=dict(
                    font=dict(color='white', size=11),
                    bgcolor='rgba(0,0,0,0.3)',
                    bordercolor='rgba(255,255,255,0.1)',
                    borderwidth=1,
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5
                ),
                height=450,
                margin=dict(t=70, b=80, l=50, r=30),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_t, use_container_width=True)
            
        except Exception:
            pass

    # Likes vs Sentiment Scatter Plot
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">❤️</div>
            <div style="font-size: 25px; font-weight: 700; color:#A78BFA;">Interactive: Likes vs Sentiment Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        show_all = st.button("📊 Show All", use_container_width=True, key="show_all")
    with col_f2:
        show_positive = st.button("😊 Positive Only", use_container_width=True, key="show_pos")
    with col_f3:
        show_negative = st.button("😡 Negative Only", use_container_width=True, key="show_neg")
    with col_f4:
        show_neutral = st.button("😐 Neutral Only", use_container_width=True, key="show_neu")
    
    filtered_scatter = df.copy()
    
    if 'scatter_filter' not in st.session_state:
        st.session_state.scatter_filter = 'all'
    
    if show_all:
        st.session_state.scatter_filter = 'all'
    elif show_positive:
        st.session_state.scatter_filter = 'positive'
    elif show_negative:
        st.session_state.scatter_filter = 'negative'
    elif show_neutral:
        st.session_state.scatter_filter = 'neutral'
    
    if st.session_state.scatter_filter == 'positive':
        filtered_scatter = filtered_scatter[filtered_scatter['sentiment'] == 'positive']
        filter_title = "😊 Positive Comments Only"
    elif st.session_state.scatter_filter == 'negative':
        filtered_scatter = filtered_scatter[filtered_scatter['sentiment'] == 'negative']
        filter_title = "😡 Negative Comments Only"
    elif st.session_state.scatter_filter == 'neutral':
        filtered_scatter = filtered_scatter[filtered_scatter['sentiment'] == 'neutral']
        filter_title = "😐 Neutral Comments Only"
    else:
        filter_title = "📊 All Comments"
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 16px;">
        <span style="background: rgba(139,92,246,0.2); padding: 6px 16px; border-radius: 20px; font-size: 13px;">
            🔍 Current Filter: {filter_title} ({len(filtered_scatter)} comments)
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    if not filtered_scatter.empty:
        scatter_df = filtered_scatter.copy()
        scatter_df['likes_display'] = scatter_df['likes'] + 1
        
        fig_scatter = px.scatter(
            scatter_df,
            x='likes_display',
            y='score',
            color='sentiment',
            color_discrete_map=COLOR_MAP,
            size='likes_display',
            size_max=20,
            hover_data={'text': True, 'author': True, 'likes': True},
            title=f"<b>{filter_title} - Likes vs Sentiment Confidence</b>",
            labels={
                'likes_display': 'Number of Likes (log scale)',
                'score': 'Sentiment Confidence Score',
                'sentiment': 'Sentiment'
            },
            log_x=True,
            opacity=0.8
        )
        
        fig_scatter.update_traces(
            marker=dict(
                line=dict(width=1.5, color='rgba(255,255,255,0.3)')
            ),
            hovertemplate='<b>Comment:</b> %{customdata[0]}<br>' +
                          '<b>Author:</b> %{customdata[1]}<br>' +
                          '<b>Likes:</b> %{customdata[2]}<br>' +
                          '<b>Confidence:</b> %{y:.0%}<br>' +
                          '<extra></extra>'
        )
        
        fig_scatter.add_hline(y=0.66, line_dash="dash", line_color="#22c55e", opacity=0.5)
        fig_scatter.add_hline(y=0.33, line_dash="dash", line_color="#ef4444", opacity=0.5)
        
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=16, color='white'),
            font=dict(color='white'),
            xaxis=dict(
                title="Number of Likes",
                title_font=dict(size=13, color='#94a3b8'),
                tickfont=dict(size=11, color='#94a3b8'),
                gridcolor='rgba(255,255,255,0.08)',
                gridwidth=0.5,
                type='log'
            ),
            yaxis=dict(
                title="Sentiment Confidence Score",
                title_font=dict(size=13, color='#94a3b8'),
                tickfont=dict(size=11, color='#94a3b8'),
                gridcolor='rgba(255,255,255,0.08)',
                gridwidth=0.5,
                range=[0, 1],
                tickformat='.0%'
            ),
            legend=dict(
                font=dict(color='white', size=11),
                bgcolor='rgba(0,0,0,0.3)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            height=500,
            margin=dict(t=80, b=50, l=50, r=30),
            hovermode='closest',
            clickmode='event+select'
        )
        
        fig_scatter.update_layout(
            legend=dict(
                title="Click on legend to filter",
                itemsizing='constant',
                tracegroupgap=5))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Statistics Box
        col1, col2, col3 = st.columns(3)
        
        pos_data = filtered_scatter[filtered_scatter['sentiment'] == 'positive']
        neg_data = filtered_scatter[filtered_scatter['sentiment'] == 'negative']
        
        avg_likes_pos = pos_data['likes'].mean() if not pos_data.empty else 0
        avg_likes_neg = neg_data['likes'].mean() if not neg_data.empty else 0
        engagement_ratio = (avg_likes_pos / avg_likes_neg) if avg_likes_neg > 0 else 0
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(34,197,94,0.1); border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 24px;">😊</div>
                <div style="font-size: 24px; font-weight: 700; color: #22c55e;">{avg_likes_pos:.1f}</div>
                <div style="font-size: 15px; color: #94a3b8;">Average Likes (Positive)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(239,68,68,0.1); border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 24px;">😡</div>
                <div style="font-size: 24px; font-weight: 700; color: #ef4444;">{avg_likes_neg:.1f}</div>
                <div style="font-size: 15px; color: #94a3b8;">Average Likes (Negative)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(139,92,246,0.1); border-radius: 12px; padding: 20px; text-align: center;margin-bottom:20px">
                <div style="font-size: 24px;">📊</div>
                <div style="font-size: 24px; font-weight: 700; color: #a78bfa;">{engagement_ratio:.1f}x</div>
                <div style="font-size: 15px; color: #94a3b8;">Positive/Negative Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Add insight message
        if engagement_ratio > 1.5:
            st.success(f"💡 **Insight:** Positive comments get {engagement_ratio:.1f}x more likes than negative comments!")
        elif engagement_ratio < 0.7:
            st.warning(f"💡 **Insight:** Negative comments get more engagement. Consider addressing viewer concerns.")
        else:
            st.info(f"💡 **Insight:** Engagement is balanced between positive and negative sentiments.")
    
    else:
        st.warning("⚠️ No comments available for the selected filter.")
    
    # Reset filter button
    if st.session_state.scatter_filter != 'all':
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.session_state.scatter_filter = 'all'
            st.rerun()
# ============
# Tab: Comments
# ============

def render_comments(df: pd.DataFrame):
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">⚙️</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Filter Options</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4, f5 = st.columns(5)
    sent_filter  = f1.selectbox("Sentiment",  ["All", "positive", "negative", "neutral"])
    lang_filter  = f2.selectbox("Language",    ["All", "arabic", "english", "mixed", "other"])
    spam_filter  = f3.selectbox("Spam",   ["All", "No Spam", "Spam Only"])
    min_likes    = f4.number_input("Min Likes", 0, value=0, step=1)
    min_conf     = f5.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05, format="%.0f%%")
    col_search, col_space = st.columns([1, 3])
    with col_search:
        search_query = st.text_input(
            "",
            placeholder="🔍 Search...",
            label_visibility="collapsed")
    
    filtered = df.copy()
    
    if sent_filter != "All":
        filtered = filtered[filtered["sentiment"] == sent_filter]
    if lang_filter != "All":
        filtered = filtered[filtered["language"] == lang_filter]
    if spam_filter == "No Spam":
        filtered = filtered[~filtered["is_spam"]]
    elif spam_filter == "Spam Only":
        filtered = filtered[filtered["is_spam"]]
    
    filtered = filtered[filtered["likes"] >= min_likes]
    filtered = filtered[filtered["score"] >= min_conf]
    
    if search_query:
        filtered = filtered[
            filtered["text"].str.contains(search_query, case=False, na=False)
        ]
    
    st.markdown(f"""
    <div style="
        background: rgba(139,92,246,0.15);
        border-radius: 12px;
        padding: 8px 16px;
        margin: 16px 0;
        text-align: center;
    ">
        <span style="color: #a78bfa; font-weight: 600;">📊 Showing {len(filtered)} comments</span>
        <span style="color: #64748b; margin-left: 12px;">(Total: {len(df)} comments)</span>
    </div>
    """, unsafe_allow_html=True)
    
    cols_show = ["author", "text", "sentiment", "score", "language", "likes", "is_spam", "is_sarcasm"]
    display_df = filtered[[c for c in cols_show if c in filtered.columns]].copy()
    display_df.columns = ["Author", "Comment", "Sentiment", "Confidence", "Language", "Likes", "Spam", "Sarcasm"]
    
    if "Confidence" in display_df.columns:
        display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.0%}")
    
    sentiment_emoji = {
        "positive": "😊",
        "negative": "😡",
        "neutral": "😐"
    }
    display_df["Sentiment"] = display_df["Sentiment"].map(lambda x: f"{sentiment_emoji.get(x, '')} {x.capitalize()}" if x in sentiment_emoji else x.capitalize())
    
    display_df["Spam"] = display_df["Spam"].map(lambda x: "🚨 Yes" if x else "✅ No")
    display_df["Sarcasm"] = display_df["Sarcasm"].map(lambda x: "🙄 Yes" if x else "➖ No")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Author": st.column_config.TextColumn("Author", width="small"),
            "Comment": st.column_config.TextColumn("Comment", width="large"),
            "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            "Language": st.column_config.TextColumn("Language", width="small"),
            "Likes": st.column_config.NumberColumn("Likes", width="small"),
            "Spam": st.column_config.TextColumn("Spam", width="small"),
            "Sarcasm": st.column_config.TextColumn("Sarcasm", width="small"),})
    
    st.divider()
    # ==============================
    # Top Comments Section with Tabs
    # ==============================
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">⭐</div>
        <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Top Comments</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_pos, tab_neg, tab_top = st.tabs(["😊 Top Positive", "😡 Top Negative", "👍 Most Liked"])
    
    with tab_pos:
        st.markdown("### 😊 Most Liked Positive Comments")
        top_pos = df[df["sentiment"] == "positive"].nlargest(10, "likes")
        if not top_pos.empty:
            for idx, (_, row) in enumerate(top_pos.iterrows()):
                st.markdown(comment_card(row), unsafe_allow_html=True)
                if idx < len(top_pos) - 1:
                    st.markdown("<hr style='margin: 8px 0; opacity: 0.3;'>", unsafe_allow_html=True)
        else:
            st.info("No positive comments found.")
    
    with tab_neg:
        st.markdown("### 😡 Most Liked Negative Comments")
        top_neg = df[df["sentiment"] == "negative"].nlargest(10, "likes")
        if not top_neg.empty:
            for idx, (_, row) in enumerate(top_neg.iterrows()):
                st.markdown(comment_card(row), unsafe_allow_html=True)
                if idx < len(top_neg) - 1:
                    st.markdown("<hr style='margin: 8px 0; opacity: 0.3;'>", unsafe_allow_html=True)
        else:
            st.info("No negative comments found.")
    
    with tab_top:
        st.markdown("### 👍 Most Liked Comments (All Sentiments)")
        top_liked = df.nlargest(10, "likes")
        if not top_liked.empty:
            for idx, (_, row) in enumerate(top_liked.iterrows()):
                st.markdown(comment_card(row), unsafe_allow_html=True)
                if idx < len(top_liked) - 1:
                    st.markdown("<hr style='margin: 8px 0; opacity: 0.3;'>", unsafe_allow_html=True)
        else:
            st.info("No comments found.")
    
    st.divider()
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📈</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 0px;">Quick Statistics</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    q1, q2, q3, q4 = st.columns(4)
    
    with q1:
        avg_likes = df['likes'].mean()
        st.markdown(f"""
        <div style="background: rgba(59,130,246,0.1); border-radius: 12px; padding: 12px; text-align: center;">
            <div style="font-size: 30px;">❤️</div>
            <div style="font-size: 30px; font-weight: 700; color: #3b82f6;">{avg_likes:.1f}</div>
            <div style="font-size: 17px; color: #94a3b8;">Average Likes per Comment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with q2:
        avg_conf = df['score'].mean()
        st.markdown(f"""
        <div style="background: rgba(139,92,246,0.1); border-radius: 12px; padding: 12px; text-align: center;">
            <div style="font-size: 30px;">🎯</div>
            <div style="font-size: 30px; font-weight: 700; color: #a78bfa;">{avg_conf:.0%}</div>
            <div style="font-size: 17px; color: #94a3b8;">Average Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with q3:
        spam_count = df['is_spam'].sum()
        st.markdown(f"""
        <div style="background: rgba(249,115,22,0.1); border-radius: 12px; padding: 12px; text-align: center;">
            <div style="font-size: 30px;">🚨</div>
            <div style="font-size: 30px; font-weight: 700; color: #f97316;">{spam_count}</div>
            <div style="font-size: 17px; color: #94a3b8;">Spam Comments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with q4:
        sarcasm_count = df['is_sarcasm'].sum()
        st.markdown(f"""
        <div style="background: rgba(236,72,153,0.1); border-radius: 12px; padding: 12px; text-align: center;">
            <div style="font-size: 30px;">🙄</div>
            <div style="font-size: 30px; font-weight: 700; color: #ec4899;">{sarcasm_count}</div>
            <div style="font-size: 17px; color: #94a3b8;">Sarcastic Comments</div>
        </div>
        """, unsafe_allow_html=True)

# ==============
# Tab: Keywords
# ==============

def render_keywords(df: pd.DataFrame):
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;"></div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">Most Frequant Keywords</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)

    for col, sentiment, label, color in [
        (w1, "positive", "😊 Positive", "#22c55e"),
        (w2, "negative", "😡 Negative", "#ef4444"),
        (w3, "neutral",  "😐 Neutral", "#9CA3AF"),
    ]:
        with col:
            st.markdown(f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border-radius: 16px;
                padding: 16px;
                margin-bottom: 16px;
                border-top: 3px solid {color};
            ">
                <div style="text-align: center; font-size: 20px; font-weight: 600; margin-bottom: 16px;">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            words = get_top_words(df, sentiment)
            
            if words:
                wdf = pd.DataFrame(words, columns=["Word", "Count"])
                wdf_display = wdf.head(10)
                
                if HAS_WORDCLOUD:
                    word_freq = dict(words[:50])
                    
                    wc = WordCloud(
                        width=500,
                        height=300,
                        background_color="#1e293b",
                        colormap="RdYlGn" if sentiment == "positive" else "RdBu",
                        max_words=50,
                        prefer_horizontal=0.7,
                        relative_scaling=0.5,
                        collocations=False
                    ).generate_from_frequencies(word_freq)
                    
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    fig.patch.set_facecolor("#1e293b")
                    plt.tight_layout(pad=0)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    
                else:
                    fig = px.bar(
                        wdf_display,
                        x="Count",
                        y="Word",
                        orientation="h",
                        color_discrete_sequence=[color],
                        text="Count"
                    )
                    fig.update_traces(
                        textposition="outside",
                        textfont=dict(size=11, color="white"),
                        marker=dict(
                            line=dict(width=0.5, color='rgba(255,255,255,0.2)'),
                            cornerradius=6,
                            opacity=0.85
                        )
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        xaxis_title="Frequency",
                        yaxis_title="",
                        height=400,
                        margin=dict(l=10, r=10, t=20, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        xaxis=dict(
                            title_font=dict(size=12, color="#94a3b8"),
                            tickfont=dict(size=10, color="#94a3b8"),
                            gridcolor='rgba(255,255,255,0.08)'
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div style="
                    background: rgba(255,255,255,0.02);
                    border-radius: 12px;
                    padding: 8px;
                    margin-top: 12px;
                ">
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    wdf_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Word": st.column_config.TextColumn("Keyword", width="medium"),
                        "Count": st.column_config.NumberColumn("Count", width="small")
                    }
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="
                    text-align: center;
                    padding: 8px;
                    margin-top: 8px;
                    border-radius: 8px;
                    background: rgba(139,92,246,0.1);
                ">
                    <span style="font-size: 16px; color: #a78bfa;">
                        📊 {len(words)} unique keywords found
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.03);
                    border-radius: 12px;
                    padding: 40px;
                    text-align: center;
                ">
                    <span style="font-size: 48px;">🔍</span>
                    <div style="color: #64748b; margin-top: 12px;">No keywords found</div>
                    <div style="font-size: 16px; color: #475569; margin-top: 4px;">Try analyzing more comments</div>
                </div>
                """, unsafe_allow_html=True)

# ===========
# Tab: Spam
# ===========

def render_spam(df: pd.DataFrame):
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">🚨</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Spam Detection Report</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    spam_df = df[df["is_spam"]].copy()
    total = len(spam_df)
    total_comments = len(df)
    spam_percentage = round(total / total_comments * 100, 1) if total_comments > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: rgba(249,115,22,0.15); border-radius: 16px; padding: 20px; text-align: center;">
            <div style="font-size: 35px;">🚨</div>
            <div style="font-size: 35px; font-weight: 700; color: #f97316;">{total}</div>
            <div style="font-size: 16px; color: #94a3b8;">Total Spam Comments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: rgba(59,130,246,0.15); border-radius: 16px; padding: 20px; text-align: center;">
            <div style="font-size: 35px;">📊</div>
            <div style="font-size: 35px; font-weight: 700; color: #3b82f6;">{spam_percentage}%</div>
            <div style="font-size: 16px; color: #94a3b8;">of Total Comments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        clean_count = total_comments - total
        st.markdown(f"""
        <div style="background: rgba(34,197,94,0.15); border-radius: 16px; padding: 20px; text-align: center;">
            <div style="font-size: 35px;">✅</div>
            <div style="font-size: 35px; font-weight: 700; color: #22c55e;">{clean_count}</div>
            <div style="font-size: 16px; color: #94a3b8;">Clean Comments</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if total == 0:
        st.markdown("""
        <div style="background: rgba(34,197,94,0.1); border-radius: 16px; padding: 48px; text-align: center;">
            <div style="font-size: 35px;">✨</div>
            <div style="font-size: 35px; font-weight: 600; color: #22c55e; margin-top: 16px;">No Spam Detected!</div>
            <div style="font-size: 16px; color: #94a3b8; margin-top: 8px;">Your comment section is clean and healthy</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📋</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Spam Comments List</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    display_spam = spam_df[["author", "text", "spam_reason", "likes"]].copy().reset_index(drop=True)
    display_spam.columns = ["Author", "Comment", "Reason", "Likes"]
    st.dataframe(
        display_spam,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Author": st.column_config.TextColumn("Author", width="small"),
            "Comment": st.column_config.TextColumn("Comment", width="large"),
            "Reason": st.column_config.TextColumn("Reason", width="medium"),
            "Likes": st.column_config.NumberColumn("Likes", width="small"),
        }
    )
    
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📖</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Spam Breakdown By reason</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    reasons = spam_df["spam_reason"].value_counts().reset_index()
    reasons.columns = ["Reason", "Count"]
    
    fig = px.bar(
        reasons,
        x="Count",
        y="Reason",
        orientation="h",
        title="<b>What type of spam is appearing?</b>",
        color_discrete_sequence=["#f97316"],
        text="Count"
    )
    
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12, color="white"),
        marker=dict(
            line=dict(width=0.5, color='rgba(255,255,255,0.2)'),
            cornerradius=6,
            opacity=0.9
        )
    )
    
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Number of Spam Comments",
        yaxis_title="",
        height=400,
        margin=dict(l=10, r=10, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(
            title_font=dict(size=12, color="#94a3b8"),
            tickfont=dict(size=11, color="#94a3b8"),
            gridcolor='rgba(255,255,255,0.08)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight message
    st.markdown("""
    <div style="background: rgba(139,92,246,0.1); border-radius: 12px; padding: 16px; margin-top: 20px; border-left: 3px solid #f97316;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 27px;">💡</span>
            <span style="font-size: 27px;font-weight: 600; color: #e2e8f0;">Recommendation</span>
        </div>
        <div style="font-size: 18px; color: #94a3b8; margin-top: 8px;">
            Enable YouTube's comment filters or add blocked keywords to automatically hide these types of spam.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============
# Tab: Insights
# =============

def render_insights(df: pd.DataFrame, summary: dict):
    st.markdown("""

    """, unsafe_allow_html=True)

    pos_pct  = summary["pos_pct"]
    neg_pct  = summary["neg_pct"]
    neu_pct  = summary["neu_pct"]
    spam_pct = round(summary["spam"] / summary["total"] * 100, 1) if summary["total"] else 0
    avg_conf = summary["avg_confidence"]
    total_comments = summary["total"]

    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.03);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 24px;
    ">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="font-size: 35px;">😊</div>
                <div style="font-size: 35px; font-weight: 700; color: #22c55e;">{pos_pct}%</div>
                <div style="font-size: 14px; color: #94a3b8;">Positive</div>
            </div>
            <div>
                <div style="font-size: 35px;">😡</div>
                <div style="font-size: 35px; font-weight: 700; color: #ef4444;">{neg_pct}%</div>
                <div style="font-size: 14px; color: #94a3b8;">Negative</div>
            </div>
            <div>
                <div style="font-size: 35px;">😐</div>
                <div style="font-size: 35px; font-weight: 700; color: #9CA3AF;">{neu_pct}%</div>
                <div style="font-size: 14px; color: #94a3b8;">Neutral</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">💡</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Actionable Insights</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    recs = []

    if pos_pct >= 60:
        recs.append(("✅", "Excellent content! Audience is very satisfied. Keep this approach!", "green"))
    elif pos_pct >= 40:
        recs.append(("🟡", "Good content. Some improvements could increase satisfaction.", "yellow"))
    else:
        recs.append(("⚠️", "Low positivity rate. Review your content strategy and presentation.", "red"))

    if neg_pct >= 30:
        neg_topics = get_topics(df, "negative")
        topics_str = ", ".join(list(neg_topics.keys())[:3]) if neg_topics else "Unspecified"
        recs.append(("🔴", f"High negative rate ({neg_pct}%). Main topics: {topics_str}.", "red"))

    if neu_pct >= 50:
        recs.append(("💬", "Many neutral comments. Add stronger CTAs to boost engagement.", "yellow"))

    if spam_pct >= 10:
        recs.append(("🚨", f"High spam rate ({spam_pct}%). Enable comment filters on your channel.", "red"))

    if avg_conf < 0.65:
        recs.append(("⚠️", f"Low average confidence ({avg_conf:.0%}). Results may need manual review.", "yellow"))

    for icon, rec, color_type in recs:
        if color_type == "green":
            bg_color = "rgba(34,197,94,0.1)"
            border_color = "#22c55e"
        elif color_type == "red":
            bg_color = "rgba(239,68,68,0.1)"
            border_color = "#ef4444"
        else:
            bg_color = "rgba(245,158,11,0.1)"
            border_color = "#f59e0b"
            
        st.markdown(f"""
        <div style="
            background: {bg_color};
            border-left: 5px solid {border_color};
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 16px;
        ">
            <div style="display: flex; align-items: center; gap: 16px;">
                <span style="font-size: 28px;">{icon}</span>
                <span style="color: #e2e8f0; font-size: 16px; font-weight: 500; line-height: 1.5;">{rec}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 5px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📖</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Topic Anlysis</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_pos_t, col_neg_t = st.columns(2)

    with col_pos_t:
        st.markdown("""
        <div style="
            background: rgba(34,197,94,0.08);
            border-radius: 16px;
            padding: 16px;
        ">
            <div style="font-size: 18px; margin-bottom: 12px;">🟢 Positive Topics</div>
        </div>
        """, unsafe_allow_html=True)
        
        pos_topics = get_topics(df, "positive")
        if pos_topics:
            topics_df = pd.DataFrame(pos_topics.items(), columns=["Topic", "Mentions"])
            st.dataframe(
                topics_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Topic": st.column_config.TextColumn("Topic", width="medium"),
                    "Mentions": st.column_config.NumberColumn("Mentions", width="small")
                }
            )
        else:
            st.info("No clear positive topics identified.")

    with col_neg_t:
        st.markdown("""
        <div style="
            background: rgba(239,68,68,0.08);
            border-radius: 16px;
            padding: 16px;
        ">
            <div style="font-size: 18px; margin-bottom: 12px;">🔴 Negative Topics</div>
        </div>
        """, unsafe_allow_html=True)
        
        neg_topics = get_topics(df, "negative")
        if neg_topics:
            topics_df = pd.DataFrame(neg_topics.items(), columns=["Topic", "Mentions"])
            st.dataframe(
                topics_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Topic": st.column_config.TextColumn("Topic", width="medium"),
                    "Mentions": st.column_config.NumberColumn("Mentions", width="small")
                }
            )
        else:
            st.info("No clear negative topics identified.")

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin: 24px 0 16px 0;">
        <span style="font-size: 20px;">🧪</span>
        <span style="font-size: 16px; font-weight: 600; color: #e2e8f0;">Methodology</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📖 How does the analysis work?", expanded=False):
        st.markdown(f"""
        **Models Used:**
        - **MARBERT** - Trained on Arabic and dialects for social media sentiment
        - **RoBERTa** - Trained on English tweets for sentiment analysis

        **Model Selection Logic:**
        - Arabic comment (>60% Arabic chars) → MARBERT only
        - English comment (<30% Arabic chars) → RoBERTa only
        - Mixed comment → Both models, choose higher confidence

        **Emoji Impact:**
        - Positive emojis boost confidence or change classification
        - Negative emojis decrease confidence score

        **Limitations:**
        - Gulf and Moroccan dialects may have lower accuracy
        - Current Confidence Threshold: **{st.session_state.get('confidence_threshold', 0.55):.0%}**
        - Sarcasm detection uses rule-based system, not a dedicated model
        """)
# ============
# Tab: Export
# ============

def render_export(df: pd.DataFrame, summary: dict):
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📥</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Export Reports</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background: rgba(59,130,246,0.1);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 24px;
        border-left: 3px solid #3b82f6;
    ">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 17px;">💡</span>
            <span style="font-size: 17px; color: #94a3b8;">
                Download your analysis results in different formats
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            margin-bottom: 16px;
            border: 1px solid rgba(255,255,255,0.05);
        ">
            <div style="font-size: 40px;">📄</div>
            <div style="font-size: 17px; font-weight: 600; margin: 8px 0; color: #e2e8f0;">CSV Format</div>
            <div style="font-size: 13px; color: #64748b;">Universal format</div>
        </div>
        """, unsafe_allow_html=True)
        
        csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    with col2:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            margin-bottom: 16px;
            border: 1px solid rgba(255,255,255,0.05);
        ">
            <div style="font-size: 40px;">📊</div>
            <div style="font-size: 17px; font-weight: 600; margin: 8px 0; color: #e2e8f0;">Excel Format</div>
            <div style="font-size: 13px; color: #64748b;">Multi-sheet report</div>
        </div>
        """, unsafe_allow_html=True)
        
        if HAS_OPENPYXL:
            buffer = io.BytesIO()
            pos_words = get_top_words(df, "positive", 20)
            neg_words = get_top_words(df, "negative", 20)
            neu_words = get_top_words(df, "neutral", 20)
            
            max_len = max(len(pos_words), len(neg_words), len(neu_words))
            
            kw_df = pd.DataFrame({
                "😊 Positive Keywords": [w for w, _ in pos_words] + [""] * (max_len - len(pos_words)),
                "Count": [c for _, c in pos_words] + [0] * (max_len - len(pos_words)),
                "😡 Negative Keywords": [w for w, _ in neg_words] + [""] * (max_len - len(neg_words)),
                "Count": [c for _, c in neg_words] + [0] * (max_len - len(neg_words)),
                "😐 Neutral Keywords": [w for w, _ in neu_words] + [""] * (max_len - len(neu_words)),
                "Count": [c for _, c in neu_words] + [0] * (max_len - len(neu_words)),
            })

            summary_df = pd.DataFrame([{
                "Total Comments": summary["total"],
                "Positive": summary["pos"], 
                "Positive %": summary["pos_pct"],
                "Negative": summary["neg"], 
                "Negative %": summary["neg_pct"],
                "Neutral": summary["neu"], 
                "Neutral %": summary["neu_pct"],
                "Spam": summary["spam"],
                "Sarcasm": summary["sarcasm"],
                "Avg Confidence": f"{summary['avg_confidence']:.0%}",
                "Arabic": summary.get("arabic", 0),
                "English": summary.get("english", 0),
                "Mixed": summary.get("mixed", 0),
            }])

            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                df.to_excel(writer, sheet_name="All Comments", index=False)
                df[df["sentiment"] == "positive"].to_excel(writer, sheet_name="Positive", index=False)
                df[df["sentiment"] == "negative"].to_excel(writer, sheet_name="Negative", index=False)
                df[df["sentiment"] == "neutral"].to_excel(writer, sheet_name="Neutral", index=False)
                df[df["is_spam"]].to_excel(writer, sheet_name="Spam", index=False)
                kw_df.to_excel(writer, sheet_name="Keywords", index=False)

            st.download_button(
                "⬇️ Download Excel",
                data=buffer.getvalue(),
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.warning("⚠️ openpyxl not installed")
            st.caption("Run: `pip install openpyxl`")
    
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📋</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Quick Copy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_copy1, col_copy2 = st.columns(2)
    
    with col_copy1:
        summary_text = f"""YouTube Sentiment Report
{'='*30}
📊 Total Comments: {summary['total']}
😊 Positive: {summary['pos']} ({summary['pos_pct']}%)
😡 Negative: {summary['neg']} ({summary['neg_pct']}%)
😐 Neutral: {summary['neu']} ({summary['neu_pct']}%)
🚨 Spam: {summary['spam']}
🙄 Sarcasm: {summary['sarcasm']}
🎯 Confidence: {summary['avg_confidence']:.0%}"""
        
        st.text_area(
            "📊 Summary",
            summary_text,
            height=180,
            disabled=True,
            label_visibility="collapsed"
        )
    
    with col_copy2:
        pos_kw = ", ".join([w for w, _ in get_top_words(df, "positive")[:8]])
        neg_kw = ", ".join([w for w, _ in get_top_words(df, "negative")[:8]])
        
        keywords_text = f"""Top Keywords
{'='*30}
😊 Positive: {pos_kw if pos_kw else '-'}
😡 Negative: {neg_kw if neg_kw else '-'}

Languages
{'='*30}
🇸🇦 Arabic: {summary.get('arabic', 0)}
🇬🇧 English: {summary.get('english', 0)}
🔀 Mixed: {summary.get('mixed', 0)}"""
        
        st.text_area(
            "🔤 Keywords",
            keywords_text,
            height=180,
            disabled=True,
            label_visibility="collapsed"
        )
    
    st.markdown("""
    <div style="
        text-align: center;
        padding: 16px;
        margin-top: 24px;
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.03);
    ">
        <span style="font-size: 16px; color: #475569;">
            📁 All exports include comments, sentiment analysis, and keywords
        </span>
    </div>
    """, unsafe_allow_html=True)

# ==================
# Compare Two Videos
# ==================

def render_comparison(confidence_threshold: float, max_comments, order: str):
    st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA; margin-top: 15px;">⚖️</div>
            <div style="font-size: 30px; font-weight: 700; color:#A78BFA;">Compare Two Videos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background: rgba(59,130,246,0.1);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 24px;
        border-left: 3px solid #3b82f6;
    ">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 20px;">💡</span>
            <span style="font-size: 20px; color: #94a3b8;">
                Compare sentiment analysis between two YouTube videos to see which performs better
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 20px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📹</div>
            <div style="font-size: 20px; font-weight: 700; color:#A78BFA;">Video 1</div>
        </div>
    </div>
        """, unsafe_allow_html=True)
        url1 = st.text_input(
            "",
            placeholder="https://youtube.com/watch?v=...",
            key="url1",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
    <div style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 20px; font-weight: 700; color:#A78BFA; margin-top: 15px;">📹</div>
            <div style="font-size: 20px; font-weight: 700; color:#A78BFA;">Video 1</div>
        </div>
    </div>
        """, unsafe_allow_html=True)
        url2 = st.text_input(
            "",placeholder="https://youtube.com/watch?v=...",key="url2",label_visibility="collapsed")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        compare_clicked = st.button("🔍 Compare Videos", type="primary", use_container_width=True)
    
    if compare_clicked:
        if not url1 or not url2:
            st.warning("⚠️ Please enter both video URLs")
            return
        
        results = {}
        progress_bar = st.progress(0, text="Analyzing videos...")
        
        for idx, (label, url) in enumerate([("Video 1", url1), ("Video 2", url2)]):
            progress_bar.progress((idx) / 2, text=f"Analyzing {label}...")
            
            vid_id = extract_video_id(url)
            if vid_id.startswith("ERROR"):
                st.error(f"❌ {label}: Invalid URL")
                return

            info, err = get_video_info(vid_id)
            if err:
                st.error(f"❌ {label}: {err}")
                return

            df_c, err = get_comments(vid_id, max_comments, order, include_replies)
            if err:
                st.error(f"❌ {label}: {err}")
                return

            if df_c.empty:
                st.warning(f"⚠️ {label}: No comments found")
                return

            df_c = analyze_sentiment(df_c, confidence_threshold)
            s = summarize_results(df_c)
            results[label] = {"info": info, "summary": s, "df": df_c}
        
        progress_bar.empty()
        st.markdown("---")
        st.markdown("""
  <div style="
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 5px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 24px;
    ">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
            <span style="font-size: 27px;">📊</span>
            <span style="font-size: 27px; font-weight: 600; color: #e2e8f0;">Comparison Results</span>
        </div>
    </div>
        """, unsafe_allow_html=True)
        
        col_v1, col_v2 = st.columns(2)
        
        video1_pos = results["Video 1"]["summary"]["pos_pct"]
        video2_pos = results["Video 2"]["summary"]["pos_pct"]
        
        for idx, (label, data) in enumerate(results.items()):
            info = data["info"]
            summary = data["summary"]
            
            sentiment_color = "#22c55e" if summary["pos_pct"] > summary["neg_pct"] else "#ef4444"
            
            winner_badge = ""
            if idx == 0 and video1_pos > video2_pos:
                winner_badge = "🏆 WINNER"
            elif idx == 1 and video2_pos > video1_pos:
                winner_badge = "🏆 WINNER"
            
            with col_v1 if idx == 0 else col_v2:
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.03);
                    border-radius: 20px;
                    padding: 16px;
                    margin-bottom: 16px;
                    border-top: 3px solid {sentiment_color};
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 14px; color: #64748b;">{label}</span>
                        <span style="font-size: 14px; color: #f59e0b; font-weight: 600;">{winner_badge}</span>
                    </div>
                    <div style="font-size: 14px; font-weight: 600; margin: 8px 0; color: #e2e8f0;">{info.get('title', 'N/A')[:50]}</div>
                    <div style="font-size: 12px; color: #94a3b8;">📺 {info.get('channel', 'N/A')}</div>
                    <div style="margin-top: 12px;">
                        <span style="font-size: 20px;">😊 {summary['pos_pct']}%</span>
                        <span style="font-size: 20px; margin-left: 16px;">😡 {summary['neg_pct']}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        comp_data = []
        for label, data in results.items():
            s = data["summary"]
            i = data["info"]
            comp_data.append({
                "Video": i.get("title", label)[:45],
                "Channel": i.get("channel", "")[:25],
                "Comments": s["total"],
                "Positive": f"{s['pos_pct']}%",
                "Negative": f"{s['neg_pct']}%",
                "Neutral": f"{s['neu_pct']}%",
                "Spam": s["spam"],
                "Confidence": f"{s['avg_confidence']:.0%}",
            })

        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        melted = pd.DataFrame({
            "Video": [comp_data[0]["Video"][:25], comp_data[1]["Video"][:25]],
            "Positive": [results["Video 1"]["summary"]["pos_pct"], results["Video 2"]["summary"]["pos_pct"]],
            "Negative": [results["Video 1"]["summary"]["neg_pct"], results["Video 2"]["summary"]["neg_pct"]],
            "Neutral": [results["Video 1"]["summary"]["neu_pct"], results["Video 2"]["summary"]["neu_pct"]],
        }).melt(id_vars="Video", var_name="Sentiment", value_name="Percentage")
        
        fig = px.bar(
            melted,
            x="Video",
            y="Percentage",
            color="Sentiment",
            barmode="group",
            title="<b>Sentiment Comparison</b>",
            color_discrete_map={
                "Positive": "#22c55e",
                "Negative": "#ef4444",
                "Neutral": "#9CA3AF"
            },
            text="Percentage"
        )
        
        fig.update_traces(textposition="outside", textfont=dict(size=11, color="white"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title_font=dict(size=16, color="white"),
            font=dict(color="white"),
            xaxis=dict(title="", tickfont=dict(size=11, color="white")),
            yaxis=dict(title="Percentage (%)", title_font=dict(size=12, color="#94a3b8")),
            height=450,
            margin=dict(t=50, b=50, l=40, r=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Winner announcement
        pos1 = results["Video 1"]["summary"]["pos_pct"]
        pos2 = results["Video 2"]["summary"]["pos_pct"]
        
        if pos1 > pos2:
            winner = "Video 1"
            diff = pos1 - pos2
            winner_emoji = "🏆"
            winner_color = "#22c55e"
        elif pos2 > pos1:
            winner = "Video 2"
            diff = pos2 - pos1
            winner_emoji = "🏆"
            winner_color = "#22c55e"
        else:
            winner = "Tie"
            diff = 0
            winner_emoji = "🤝"
            winner_color = "#f59e0b"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(236,72,153,0.1));
            border-radius: 20px;
            padding: 24px;
            margin-top: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        ">
            <div style="font-size: 48px;">{winner_emoji}</div>
            <div style="font-size: 25px; font-weight: 700; color: {winner_color}; margin-top: 8px;">
                {winner if winner == "Tie" else f"{winner} wins by {diff:.1f}%!"}
            </div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 8px;">
                Video 1: {pos1}% positive | Video 2: {pos2}% positive
            </div>
        </div>
        """, unsafe_allow_html=True)

# ========
# Sidebar
# ========

with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:center; gap:8px; margin-bottom:15px;">
        <span style="font-size:13px; font-weight:600; color:#8B5CF6; letter-spacing:0.5px;">🌐 LANGUAGE / اللغة</span>
    </div>
    """, unsafe_allow_html=True)

    _lang_col1, _lang_col2 = st.columns(2)
    with _lang_col1:
        _en_style = (
            "background:linear-gradient(135deg,#8B5CF6,#EC4899);color:white;border:none;"
            "border-radius:10px;padding:8px 0;width:100%;font-size:14px;font-weight:700;"
            "cursor:pointer;box-shadow:0 4px 12px rgba(139,92,246,0.4);"
        ) if st.session_state["ui_lang"] == "en" else (
            "background:rgba(255,255,255,0.07);color:#94a3b8;border:1px solid rgba(255,255,255,0.15);"
            "border-radius:10px;padding:8px 0;width:100%;font-size:14px;font-weight:600;"
            "cursor:pointer;transition:all 0.2s;"
        )
        if st.button("English", key="_btn_en", use_container_width=True):
            st.session_state["ui_lang"] = "en"
            st.rerun()
    with _lang_col2:
        if st.button("العربية", key="_btn_ar", use_container_width=True):
            st.session_state["ui_lang"] = "ar"
            st.rerun()

    _is_ar = st.session_state["ui_lang"] == "ar"
    st.markdown(f"""
    <div style="display:flex; gap:4px; margin:4px 0 10px 0;">
        <div style="flex:1; height:3px; border-radius:2px;
            background:{'rgba(139,92,246,0.35)' if not _is_ar else 'rgba(255,255,255,0.12)'};"></div>
        <div style="flex:1; height:3px; border-radius:2px;
            background:{'rgba(139,92,246,0.35)' if _is_ar else 'rgba(255,255,255,0.12)'};"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="font-size: 16px; font-weight: 500; color: #8B5CF6;">⚙️</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="font-size: 13px; font-weight: 500; color: white;">{t("Control Panel", "لوحة التحكم")}</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="font-size: 12px; font-weight: 500; color: gray;">{t("Advanced Sentiment Analysis", "تحليل المشاعر المتقدم")}</span>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span style="font-size: 16px;">🟢</span>
        <span style="font-size: 14px; font-weight: 600; color: #94a3b8;">{t("App Mode", "وضع التطبيق")}</span>
    </div>
    """, unsafe_allow_html=True)

    _mode_options_en = ["Single Video", "Compare Two Videos"]
    _mode_options_ar = ["فيديو واحد", "مقارنة فيديوين"]
    _mode_opts = _mode_options_ar if _is_ar else _mode_options_en

    _app_mode_raw = st.radio(
        "",
        _mode_opts,
        label_visibility="collapsed"
    )
    if _is_ar:
        app_mode = _mode_options_en[_mode_options_ar.index(_app_mode_raw)]
    else:
        app_mode = _app_mode_raw

    st.divider()

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span style="font-size: 16px;">🟢</span>
        <span style="font-size: 14px; font-weight: 600; color: #94a3b8;">{t("Comments Order By", "ترتيب التعليقات")}</span>
    </div>
    """, unsafe_allow_html=True)

    _order_opts_en = [ "time" , "relevance"]
    _order_opts_ar = [ "الأحدث" , "الأكثر صلة"]
    _order_display = _order_opts_ar if _is_ar else _order_opts_en
    _order_raw = st.selectbox(
        "",
        _order_display,
        label_visibility="collapsed",
        help=t("relevance = best comments first, time = newest first",
               "الأكثر صلة = أفضل التعليقات أولاً، الأحدث = الأجدد أولاً")
    )
    order = _order_opts_en[_order_display.index(_order_raw)]

    st.divider()

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; margin-top: 12px;">
        <span style="font-size: 16px;">🟢</span>
        <span style="font-size: 14px; font-weight: 600; color: #94a3b8;">{t("Number Of Comments", "عدد التعليقات")}</span>
    </div>
    """, unsafe_allow_html=True)

    _num_opts_en = ["Specific Number", "All Comments"]
    _num_opts_ar = ["عدد محدد", "كل التعليقات"]
    _num_display = _num_opts_ar if _is_ar else _num_opts_en
    _mode_raw = st.radio(
        "",
        _num_display,
        label_visibility="collapsed"
    )
    mode = _num_opts_en[_num_display.index(_mode_raw)]

    if mode == "Specific Number":
        max_comments = st.slider(
            "",
            50, 10000, 300, 50,
            label_visibility="collapsed",
            help=t("Number of comments to analyze", "عدد التعليقات للتحليل")
        )
    else:
        max_comments = None

    include_replies = st.checkbox(
        t("Include replies", "تضمين الردود"),
        value=False,
        help=t(
            "Also fetch first-page replies for each comment thread (uses more API quota).",
            "جلب الردود الأولى لكل خيط تعليق (يستهلك حصة API أكثر)."
        )
    )

    st.divider()

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span style="font-size: 16px;">🟢</span>
        <span style="font-size: 14px; font-weight: 600; color: #94a3b8;">{t("Confidence Threshold", "عتبة الثقة")}</span>
    </div>
    """, unsafe_allow_html=True)

    confidence_threshold = st.slider(
        "",
        0.30, 0.90, 0.55, 0.05,
        label_visibility="collapsed",
        help=t("Classification confidence (higher = more precise)",
               "ثقة التصنيف (أعلى = أدق)")
    )
    st.session_state["confidence_threshold"] = confidence_threshold

    st.divider()

    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span style="font-size: 16px;">🟢</span>
        <span style="font-size: 14px; font-weight: 600; color: #94a3b8;">{t("Display Filtering", "فلترة العرض")}</span>
    </div>
    """, unsafe_allow_html=True)

    show_spam = st.checkbox(t("Show Spam Comments", "عرض تعليقات السبام"), value=True)

    lang_display = st.multiselect(
        t("Language Filtering", "فلترة اللغة"),
        ["arabic", "english", "mixed", "other"],
        default=["arabic", "english", "mixed", "other"],
        help=t("Choose languages you want to display", "اختر اللغات التي تريد عرضها")
    )

    st.divider()

    st.markdown(f"""
    <div style="
        background: rgba(34,197,94,0.12);
        border-radius: 14px;
        padding: 14px;
        margin-top: 8px;
        text-align: center;
        border: 1px solid rgba(34,197,94,0.2);
    ">
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="font-size: 14px;">✅</span>
            <span style="font-size: 13px; font-weight: 500; color: #22c55e;">{t("YouTube API Connected", "متصل بـ YouTube API")}</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="font-size: 12px; font-weight: 500; color: #475569">Google YouTube Data API v3</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        text-align: center;
        padding: 16px;
        margin-top: 24px;
        border-top: 1px solid rgba(255,255,255,0.06);
    ">
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="font-size: 13px; font-weight: 500; color: #8B5CF6;">🚀 PRO Version 2.0</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="font-size: 12px; font-weight: 500; color: gray;">MarBERT + RoBERTa</span>
        </div>
    """, unsafe_allow_html=True)


# =============
# CSS — Sidebar
# =============

st.markdown("""
<style>

/* ── 1. PIN SIDEBAR TO THE LEFT ─────────────────────────────────────── */

[data-testid="stSidebar"] {
    left: 0 !important;
    right: auto !important;
    width: 360px !important;
    min-width: 360px !important;
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-left: none !important;
    transition: width 0.3s ease, min-width 0.3s ease, transform 0.3s ease !important;
}

/* ── 2. COLLAPSED STATE — zero width, no gap ────────────────────────── */

section[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0 !important;
    min-width: 0 !important;
    overflow: hidden !important;
    transform: translateX(-100%) !important;
    border-right: none !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] ~ .main,
section[data-testid="stSidebar"][aria-expanded="false"] ~ section.main {
    margin-left: 0 !important;
}

/* ── 3. LTR / LEFT-ALIGNED ──────────────────────────────────────────── */

[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    direction: ltr !important;
    text-align: left !important;
    font-size: 19px !important;
}

[data-testid="stSidebar"] label {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    gap: 6px !important;
}

[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 8px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    color: #e2e8f0 !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="tag"],
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stSlider *,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] .stMarkdown {
    direction: ltr !important;
    text-align: left !important;
}

/* ── 4. KEEP TOGGLE BUTTON VISIBLE ─────────────────────────────────── */

[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* ── 5. TOGGLE BUTTON STYLING — bigger, bolder, unmissable ─────────── */

button[data-testid="baseButton-headerNoPadding"] {
    background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%) !important;
    border-radius: 14px !important;
    padding: 12px 16px !important;
    margin: 12px 0 0 16px !important;
    border: none !important;
    box-shadow:
        0 0 0 3px rgba(139, 92, 246, 0.25),
        0 6px 20px rgba(139, 92, 246, 0.55) !important;
    cursor: pointer !important;
    position: relative !important;
    z-index: 9999 !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    animation: sidebar-btn-pulse 3s ease-in-out infinite !important;
}

@keyframes sidebar-btn-pulse {
    0%, 100% { box-shadow: 0 0 0 3px rgba(139,92,246,0.25), 0 6px 20px rgba(139,92,246,0.55); }
    50%       { box-shadow: 0 0 0 6px rgba(139,92,246,0.12), 0 8px 28px rgba(139,92,246,0.70); }
}

button[data-testid="baseButton-headerNoPadding"]:hover {
    transform: scale(1.10) translateY(-2px) !important;
    box-shadow:
        0 0 0 4px rgba(139, 92, 246, 0.35),
        0 10px 32px rgba(139, 92, 246, 0.70) !important;
    animation: none !important;
}

button[data-testid="baseButton-headerNoPadding"] svg {
    width: 24px !important;
    height: 24px !important;
    fill: white !important;
    color: white !important;
}

/* Tooltip label */
button[data-testid="baseButton-headerNoPadding"]::after {
    content: "⚙ Settings";
    position: absolute;
    left: calc(100% + 12px);
    top: 50%;
    transform: translateY(-50%);
    background: #1e293b;
    color: #f1f5f9;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
    padding: 6px 13px;
    border-radius: 10px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
    box-shadow: 0 2px 12px rgba(0,0,0,0.40);
    border: 1px solid rgba(255,255,255,0.10);
    z-index: 10000;
}

button[data-testid="baseButton-headerNoPadding"]:hover::after {
    opacity: 1;
}

/* Collapsed state — green gradient to stand out even more */
section[data-testid="stSidebar"][aria-expanded="false"] + div
    button[data-testid="baseButton-headerNoPadding"] {
    background: linear-gradient(135deg, #22c55e 0%, #3b82f6 100%) !important;
    box-shadow:
        0 0 0 3px rgba(34, 197, 94, 0.30),
        0 6px 20px rgba(34, 197, 94, 0.50) !important;
    animation: sidebar-btn-pulse-open 3s ease-in-out infinite !important;
}

@keyframes sidebar-btn-pulse-open {
    0%, 100% { box-shadow: 0 0 0 3px rgba(34,197,94,0.30), 0 6px 20px rgba(34,197,94,0.50); }
    50%       { box-shadow: 0 0 0 6px rgba(34,197,94,0.12), 0 8px 28px rgba(34,197,94,0.65); }
}

section[data-testid="stSidebar"][aria-expanded="false"] + div
    button[data-testid="baseButton-headerNoPadding"]::after {
    content: "⚙ Open Settings";
}

/* ── 6. MAIN CONTENT PADDING ────────────────────────────────────────── */

.main .block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── 7. RESPONSIVE ──────────────────────────────────────────────────── */

@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
    }

    button[data-testid="baseButton-headerNoPadding"] {
        padding: 8px 11px !important;
        margin: 8px 0 0 10px !important;
        border-radius: 10px !important;
    }

    button[data-testid="baseButton-headerNoPadding"] svg {
        width: 19px !important;
        height: 19px !important;
    }
}

</style>
""", unsafe_allow_html=True)

# =======
# Main UI
# =======
st.divider()

# Compare Mode
if app_mode == "Compare Two Videos":
    render_comparison(confidence_threshold, max_comments, order)
    st.stop()

# Single Video Mode
_, col_center, _ = st.columns([1, 2, 1])

with col_center:
    _dir_attr2 = 'dir="rtl"' if st.session_state["ui_lang"] == "ar" else ''
    st.markdown(f"""
    <div {_dir_attr2} style="
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 40px;">🔍</span>
            <div style="font-size: 30px; font-weight: 700; color: #e2e8f0; margin-top: 8px;">{t("Analyze YouTube Video", "تحليل فيديو يوتيوب")}</div>
            <div style="font-size: 17px; color: white; margin-top: 30px;">{t("Paste a YouTube link or Video ID below", "الصق رابط يوتيوب أو معرّف الفيديو أدناه")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    url = st.text_input(
        "",
        placeholder="https://youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

    analyze_btn = st.button(f"🔍 {t('Analyze', 'تحليل')}", type="primary", use_container_width=True)

if analyze_btn and url:
    vid_id = extract_video_id(url)

    if vid_id == "ERROR:apikey":
        st.error(t("❌ It looks like you entered an API Key instead of a video URL.",
                   "❌ يبدو أنك أدخلت مفتاح API بدلاً من رابط الفيديو."))
        st.stop()
    elif vid_id == "ERROR:invalid":
        st.error(t("❌ Invalid URL. Please enter a valid YouTube link.",
                   "❌ رابط غير صالح. يرجى إدخال رابط يوتيوب صحيح."))
        st.stop()

    with st.spinner(t("Fetching video info...", "جاري جلب معلومات الفيديو...")):
        video_info, err = get_video_info(vid_id)
    if err:
        st.error(f"❌ {err}")
        st.stop()

    with st.spinner(t("Fetching comments...", "جاري جلب التعليقات...")):
        df_raw, err = get_comments(vid_id, max_comments, order, include_replies)
    if err:
        st.error(f"❌ {err}")
        st.stop()

    if df_raw.empty:
        st.warning(t("⚠️ No comments found.", "⚠️ لم يتم العثور على تعليقات."))
        st.stop()

    df_filtered = df_raw.copy()
    st.toast(f"✅ {t('Fetched', 'تم جلب')} {len(df_raw)} {t('comments', 'تعليق')}")

    with st.spinner(t("Analyzing sentiment...", "جاري تحليل المشاعر...")):
        df_analyzed = analyze_sentiment(df_filtered, confidence_threshold)

    if lang_display:
        df_analyzed = df_analyzed[df_analyzed["language"].isin(lang_display)]

    if not show_spam:
        df_display = df_analyzed[~df_analyzed["is_spam"]]
    else:
        df_display = df_analyzed

    summary = summarize_results(df_display)
    save_analysis(
    video_id=vid_id,
    video_title=video_info["title"],
    positive_percent=summary["pos_pct"],
    negative_percent=summary["neg_pct"],
    neutral_percent=summary["neu_pct"],
    total_comments=summary["total"])
    
    insights = generate_ai_insights(summary, df_display)
    st.session_state["insights"] = insights
    
    pdf_file = generate_pdf_report(
        video_info,
        summary,
        insights)
    
    st.session_state["pdf_file"] = pdf_file

    st.session_state["df"]         = df_display
    st.session_state["video_info"] = video_info
    st.session_state["summary"]    = summary
    st.toast(f"✅ {t('Analysis complete!', 'اكتمل التحليل!')}")

if (
    "df" in st.session_state
    and st.session_state["df"] is not None
    and not st.session_state["df"].empty
):
    df         = st.session_state["df"]
    video_info = st.session_state["video_info"]
    summary    = st.session_state["summary"]

    st.success(f"✅ {t('Analysis completed successfully!', 'اكتمل التحليل بنجاح!')}")

    tab_overview, tab_comments, tab_keywords, tab_spam, tab_insights, tab_export , tab_history= st.tabs([
        f"{t('Overview','نظرة عامة')}",
        f"{t('Comments','التعليقات')}",
        f"{t('Keywords','الكلمات المفتاحية')}",
        f"{t('Spam','السبام')}",
        f"{t('Insights','الرؤى')}",
        f"{t('Export','تصدير')}",
        f"{t('History','التحليلات السابقة')}",
    ])
    with tab_overview:
        render_overview(df, summary, video_info)
        st.markdown("## AI Insights")
        if "insights" in st.session_state:
            for insight in st.session_state["insights"]:
                st.markdown(f"""
                            <div style="
                            background: rgba(139,92,246,0.12);
                            border-left: 4px solid #8B5CF6;
                            padding: 16px;
                            border-radius: 12px;
                            margin-bottom: 12px;
                            font-size: 20px;
                            ">
                {insight}
        </div>
        """, unsafe_allow_html=True)
            
        if "pdf_file" in st.session_state:
            with open(st.session_state["pdf_file"], "rb") as f:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=f,
                    file_name="youtube_analysis_report.pdf",
                    mime="application/pdf")



    with tab_comments:
        render_comments(df)

    with tab_keywords:
        render_keywords(df)

    with tab_spam:
        render_spam(df)

    with tab_insights:
        render_insights(df, summary)

    with tab_export:
        render_export(df, summary)
        
    with tab_history:
        st.title("Analysis History")
        history_df = get_all_analyses()
        st.dataframe(
            history_df,
            use_container_width=True)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
        display: flex;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        justify-content: center;
        font-size: 15px;
        font-weight: 600;
        padding: 10px 0;
    }
</style>
""", unsafe_allow_html=True)