"""
app.py
FaceTrack Classroom — a face-recognition attendance app for schools, built for Streamlit.
Supports multiple classes/sections, admin vs teacher logins, and per-class PDF reports.
"""
import cv2
import numpy as np
import os
import shutil
from datetime import datetime, date

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

import database
import face_utils
import pdf_report

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="FaceTrack Classroom", page_icon="🎓", layout="wide")

def build_css(theme: str) -> str:
    """Returns the full <style> block for the given theme ('light' or 'dark')."""
    if theme == "dark":
        v = dict(
            bg_start="#0B1120", bg_end="#111827",
            card="#1E293B", card_border="#334155",
            text_primary="#F1F5F9", text_secondary="#94A3B8",
            input_bg="#0F172A", input_border="#475569", input_text="#F1F5F9",
            placeholder="#64748B",
            secondary_btn_bg="#334155", secondary_btn_hover="#475569", secondary_btn_text="#F1F5F9",
            table_border="#334155", table_stripe="#1A2437",
            shadow="rgba(0,0,0,0.45)",
        )
    else:
        v = dict(
            bg_start="#F8FAFC", bg_end="#EEF2FF",
            card="#FFFFFF", card_border="#E2E8F0",
            text_primary="#0F172A", text_secondary="#64748B",
            input_bg="#FFFFFF", input_border="#CBD5E1", input_text="#0F172A",
            placeholder="#94A3B8",
            secondary_btn_bg="#F1F5F9", secondary_btn_hover="#E2E8F0", secondary_btn_text="#0F172A",
            table_border="#E2E8F0", table_stripe="#F8FAFC",
            shadow="rgba(15,23,42,0.08)",
        )

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
.stAppDeployButton {{display: none;}}
div[data-testid="stToolbar"] {{visibility: hidden;}}

/* ---------- Animation keyframes ---------- */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes pulseRing {{
    0%   {{ box-shadow: 0 0 0 0 rgba(79,70,229,0.45); }}
    70%  {{ box-shadow: 0 0 0 14px rgba(79,70,229,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(79,70,229,0); }}
}}
@keyframes floatIcon {{
    0%, 100% {{ transform: translateY(0); }}
    50%      {{ transform: translateY(-6px); }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -400px 0; }}
    100% {{ background-position: 400px 0; }}
}}

.stApp {{
    background: linear-gradient(180deg, {v['bg_start']} 0%, {v['bg_end']} 100%);
    transition: background 0.35s ease;
}}

/* fade the whole page content in on every rerun/navigation */
section.main > div.block-container {{
    animation: fadeIn 0.45s ease;
}}

label, .stTextInput label, .stTextInput p, .stMarkdown p,
div[data-testid="stForm"] label, div[data-testid="stWidgetLabel"] p {{
    color: {v['text_primary']} !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}}
.stCaption, [data-testid="stCaptionContainer"] p {{
    color: {v['text_secondary']} !important;
}}

div[data-baseweb="input"], div[data-baseweb="select"] {{
    background: {v['input_bg']} !important;
    border-radius: 10px !important;
    border: 1.5px solid {v['input_border']} !important;
    transition: border 0.2s ease, box-shadow 0.2s ease;
}}
div[data-baseweb="input"] input, div[data-baseweb="select"] * {{
    background: {v['input_bg']} !important;
    color: {v['input_text']} !important;
    caret-color: {v['input_text']} !important;
}}
div[data-baseweb="input"] input::placeholder {{
    color: {v['placeholder']} !important;
}}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {{
    border: 1.5px solid #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.18) !important;
}}

.stButton>button {{
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    border: none;
    transition: all 0.18s cubic-bezier(.34,1.56,.64,1);
}}
.stButton>button:active {{
    transform: scale(0.97);
}}
.stButton>button[kind="primary"] {{
    background: linear-gradient(135deg, #4F46E5, #6366F1);
    color: #fff;
}}
.stButton>button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #4338CA, #4F46E5);
    transform: translateY(-2px) scale(1.015);
    box-shadow: 0 8px 20px rgba(79,70,229,0.4);
}}
.stButton>button[kind="secondary"] {{
    background: {v['secondary_btn_bg']};
    color: {v['secondary_btn_text']};
    border: 1px solid {v['input_border']};
}}
.stButton>button[kind="secondary"]:hover {{
    background: {v['secondary_btn_hover']};
    transform: translateY(-1px);
}}

h1, h2, h3 {{ color: {v['text_primary']} !important; }}
h1 {{ animation: fadeInUp 0.5s ease; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {v['card']};
    border-radius: 16px !important;
    border: 1px solid {v['card_border']} !important;
    box-shadow: 0 10px 30px {v['shadow']};
    padding: 8px;
    animation: fadeInUp 0.45s ease;
    transition: background 0.3s ease, border 0.3s ease;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
    color: {v['text_secondary']};
    transition: color 0.2s ease;
}}
hr {{ border-color: {v['card_border']} !important; }}

div[data-testid="stDataFrame"] {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {v['table_border']};
    animation: fadeInUp 0.4s ease;
}}

div[data-testid="stCameraInput"] {{
    background: {v['card']};
    border-radius: 14px;
    padding: 12px;
    border: 1px solid {v['card_border']};
    transition: box-shadow 0.3s ease;
}}
div[data-testid="stCameraInput"]:hover {{
    box-shadow: 0 6px 20px {v['shadow']};
}}
div[data-testid="stCameraInput"] button {{
    border-radius: 8px !important;
}}

div[data-testid="stAlert"] {{
    border-radius: 10px !important;
    animation: fadeInUp 0.35s ease;
}}

.stProgress > div > div > div > div {{
    background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
    transition: width 0.4s ease;
}}

.role-badge {{
    display:inline-block; padding:4px 12px; border-radius:999px;
    font-size:12px; font-weight:700; letter-spacing:.03em;
    animation: fadeIn 0.6s ease;
}}
.app-title-text {{ color: {v['text_primary']}; }}

/* ---------- Metric cards ---------- */
.metric-card {{
    background: {v['card']};
    border-radius: 16px; padding: 22px 24px;
    box-shadow: 0 4px 18px {v['shadow']};
    border-left: 6px solid var(--accent-color, #6366F1);
    animation: fadeInUp 0.5s ease;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 28px {v['shadow']};
}}
.metric-card .metric-icon {{
    font-size: 26px;
    display: inline-block;
    animation: floatIcon 3s ease-in-out infinite;
}}
.metric-card .metric-label {{
    font-size: 14px; color: {v['text_secondary']}; margin-top: 6px; font-weight: 500;
}}
.metric-card .metric-value {{
    font-size: 34px; font-weight: 800; color: {v['text_primary']}; line-height: 1.2;
}}

/* ---------- Scanning pulse ring around camera capture result ---------- */
.scan-ring-wrap {{ animation: fadeInUp 0.5s ease; }}
.scan-ring-live {{ animation: pulseRing 1.6s infinite; }}

/* ---------- Theme toggle button ---------- */
.theme-toggle-btn button {{
    border-radius: 999px !important;
    width: 42px; height: 42px; padding: 0 !important;
    font-size: 18px !important;
}}

/* ---------- Login page: split panel ---------- */
@keyframes gradientShift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes blobFloat {{
    0%, 100% {{ transform: translate(0, 0) scale(1); }}
    33%      {{ transform: translate(20px, -25px) scale(1.08); }}
    66%      {{ transform: translate(-15px, 15px) scale(0.95); }}
}}
@keyframes slideInLeft {{
    from {{ opacity: 0; transform: translateX(-24px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes slideInRight {{
    from {{ opacity: 0; transform: translateX(24px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}

.login-hero-panel {{
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    min-height: 460px;
    padding: 44px 36px;
    background: linear-gradient(-45deg, #4338CA, #4F46E5, #6366F1, #4338CA);
    background-size: 300% 300%;
    animation: gradientShift 10s ease infinite, slideInLeft 0.6s ease;
    display: flex;
    flex-direction: column;
    justify-content: center;
    color: #fff;
    box-shadow: 0 20px 45px rgba(79,70,229,0.35);
}}
.login-hero-panel::before, .login-hero-panel::after {{
    content: "";
    position: absolute;
    border-radius: 50%;
    background: rgba(255,255,255,0.10);
    animation: blobFloat 9s ease-in-out infinite;
}}
.login-hero-panel::before {{ width: 220px; height: 220px; top: -60px; right: -50px; }}
.login-hero-panel::after  {{ width: 160px; height: 160px; bottom: -40px; left: -30px; animation-delay: 2s; }}

.login-hero-icon {{
    font-size: 48px;
    animation: floatIcon 3s ease-in-out infinite;
    display: inline-block;
    margin-bottom: 14px;
}}
.login-hero-title {{
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 6px;
    position: relative;
    z-index: 1;
}}
.login-hero-sub {{
    font-size: 1rem;
    opacity: 0.9;
    margin-bottom: 28px;
    position: relative;
    z-index: 1;
}}
.login-hero-feature {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.92rem;
    padding: 9px 0;
    opacity: 0;
    animation: fadeInUp 0.5s ease forwards;
    position: relative;
    z-index: 1;
}}
.login-hero-feature:nth-child(1) {{ animation-delay: 0.15s; }}
.login-hero-feature:nth-child(2) {{ animation-delay: 0.3s; }}
.login-hero-feature:nth-child(3) {{ animation-delay: 0.45s; }}
.login-hero-feature .tick {{
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px;
    flex-shrink: 0;
}}

.login-form-panel {{
    animation: slideInRight 0.6s ease;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: {v['secondary_btn_bg']};
    border-radius: 12px;
    padding: 4px;
    gap: 2px !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 9px;
    padding: 10px 20px !important;
    font-weight: 600;
    white-space: nowrap;
    transition: all 0.25s ease;
}}
.stTabs [aria-selected="true"] {{
    background: #4F46E5 !important;
    color: #fff !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
}}
.stTabs [data-baseweb="tab-border"] {{
    background: transparent !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 22px !important;
    animation: fadeInUp 0.35s ease;
}}


div[data-testid="stHorizontalBlock"]:has(.login-hero-panel) {{
    align-items: stretch;
}}
div[data-testid="stHorizontalBlock"]:has(.login-hero-panel) > div[data-testid="column"] {{
    display: flex;
}}
div[data-testid="stHorizontalBlock"]:has(.login-hero-panel) > div[data-testid="column"]:last-child {{
    align-items: center;
    justify-content: center;
}}
.login-hero-panel {{
    height: 100%;
    min-height: 0;
}}
.login-form-panel {{
    width: 100%;
}}
.login-form-panel div[data-testid="stVerticalBlockBorderWrapper"] {{
    padding: 8px 6px;
}}

@media (max-width: 768px) {{
    .login-hero-panel {{ padding: 30px 24px; min-height: 260px; }}
    .login-hero-title {{ font-size: 1.5rem; }}
    .login-hero-icon {{ font-size: 36px; }}
    .login-hero-feature {{ font-size: 0.85rem; }}
}}

</style>
"""


for key, default in [
    ("logged_in", False),
    ("role", None),            # "admin" or "teacher"
    ("teacher_name", None),
    ("teacher_class", None),   # class_section the logged-in teacher owns
    ("pending_name", None),
    ("pending_phone", None),
    ("pending_class", None),
    ("captured_images", []),
    ("theme", "light"),        # "light" or "dark"
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(build_css(st.session_state["theme"]), unsafe_allow_html=True)

database.init_db()
os.makedirs(face_utils.FACES_DIR, exist_ok=True)


def theme_toggle_button():
    icon = "☀️" if st.session_state["theme"] == "dark" else "🌙"
    st.markdown('<div class="theme-toggle-btn">', unsafe_allow_html=True)
    if st.button(icon, key="theme_toggle"):
        st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def get_admin_credentials():
    """Reads admin login credentials from st.secrets if configured, else falls back to defaults.
    For deployment, set these in .streamlit/secrets.toml:
        [credentials]
        username = "your_admin"
        password = "your_password"
    """
    try:
        username = st.secrets["credentials"]["username"]
        password = st.secrets["credentials"]["password"]
    except Exception:
        username = "Dharmateja"
        password = "Dharmateja@1234"
    return username, password


def metric_card(label, value, color, icon):
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color:{color};">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_to_base64(np_img):
    import base64
    from io import BytesIO
    pil = Image.fromarray(np_img)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def current_class_scope():
    """Returns the class_section this session is scoped to, or None for admin (all classes)."""
    if st.session_state["role"] == "teacher":
        return st.session_state["teacher_class"]
    return st.session_state.get("dashboard_class_filter") or None


# --------------------------------------------------------------------------------------
# Login page
# --------------------------------------------------------------------------------------

def login_page():
    top_l, top_r = st.columns([6, 1])
    with top_r:
        theme_toggle_button()

    st.write("")
    col_hero, col_form = st.columns([1.15, 1], gap="large")

    with col_hero:
        st.markdown(
            """
            <div class="login-hero-panel">
                <div class="login-hero-icon">🎓</div>
                <div class="login-hero-title">FaceTrack Classroom</div>
                <div class="login-hero-sub">Multi-class, camera-based attendance for schools</div>
                <div class="login-hero-feature"><span class="tick">✓</span> Face-recognition check-in in seconds</div>
                <div class="login-hero-feature"><span class="tick">✓</span> Separate admin & teacher dashboards</div>
                <div class="login-hero-feature"><span class="tick">✓</span> Exportable per-class attendance reports</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_form:
        st.markdown('<div class="login-form-panel">', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                "<h3 style='margin-top:4px; margin-bottom:2px;'>Welcome back 👋</h3>"
                "<p class='stCaption' style='margin-bottom:6px;'>Sign in to continue to your dashboard</p>",
                unsafe_allow_html=True,
            )
            tab_admin, tab_teacher = st.tabs(["🛡️ Admin", "🍎 Teacher"])

            with tab_admin:
                with st.form("admin_login_form"):
                    username = st.text_input("Admin Username", placeholder="Enter your username")
                    password = st.text_input("Admin Password", type="password", placeholder="Enter your password")
                    submitted = st.form_submit_button("Login as Admin", use_container_width=True, type="primary")
                if submitted:
                    correct_user, correct_pass = get_admin_credentials()
                    if username == correct_user and password == correct_pass:
                        st.session_state["logged_in"] = True
                        st.session_state["role"] = "admin"
                        st.rerun()
                    else:
                        st.error("Invalid admin username or password.")

            with tab_teacher:
                with st.form("teacher_login_form"):
                    t_username = st.text_input("Teacher Username", placeholder="Enter your username")
                    t_password = st.text_input("Teacher Password", type="password", placeholder="Enter your password")
                    t_submitted = st.form_submit_button("Login as Teacher", use_container_width=True, type="primary")
                if t_submitted:
                    match = database.get_teacher_by_login(t_username.strip(), t_password)
                    if match:
                        _tid, tname, tclass = match
                        st.session_state["logged_in"] = True
                        st.session_state["role"] = "teacher"
                        st.session_state["teacher_name"] = tname
                        st.session_state["teacher_class"] = tclass
                        st.rerun()
                    else:
                        st.error("Invalid teacher username or password.")
                st.caption("Teacher accounts are created by an admin under **Manage Teachers**.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center; margin-top:22px;' class='stCaption'>"
        "🔒 Your data is processed locally and never leaves this device."
        "</p>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------------------
# Dashboard page
# --------------------------------------------------------------------------------------
def dashboard_page():
    st.markdown("<h1 style='margin-bottom:0; color:#0F172A;'>📊 Attendance Dashboard</h1>", unsafe_allow_html=True)
    st.caption(datetime.now().strftime("%A, %d %B %Y"))
    st.write("")

    all_classes = database.get_all_classes()

    if st.session_state["role"] == "admin":
        options = ["All Classes"] + all_classes
        choice = st.selectbox("Class / Section", options, key="dashboard_class_select")
        scope_class = None if choice == "All Classes" else choice
        st.session_state["dashboard_class_filter"] = scope_class
    else:
        scope_class = st.session_state["teacher_class"]
        st.info(f"Showing attendance for your class: **{scope_class}**")

    members = database.get_all_members(class_section=scope_class)
    present_map = database.get_today_present_ids(class_section=scope_class)

    total = len(members)
    present_count = len(present_map)
    absent_count = total - present_count

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Total Registered", total, "#6366F1", "👥")
    with col2:
        metric_card("Present Today", present_count, "#22C55E", "✅")
    with col3:
        metric_card("Absent Today", absent_count, "#EF4444", "🚫")

    st.write("")

    # PDF export — only meaningful when scoped to a single class
    if scope_class:
        pdf_bytes = pdf_report.build_class_attendance_pdf(scope_class, date.today().isoformat())
        st.download_button(
            f"📄 Export {scope_class} Attendance Report (PDF)",
            data=pdf_bytes,
            file_name=f"attendance_{scope_class}_{date.today().isoformat()}.pdf",
            mime="application/pdf",
            type="primary",
        )
    else:
        st.caption("Select a specific class above to export its PDF attendance report.")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🟢 Present Members")
        present_rows = [(name, phone, cls, present_map[mid]) for (mid, name, phone, cls) in members if mid in present_map]
        if present_rows:
            cols = ["Name", "Phone", "Class", "Check-in Time"] if not scope_class else ["Name", "Phone", "Check-in Time"]
            trimmed = present_rows if not scope_class else [(n, p, t) for (n, p, c, t) in present_rows]
            df = pd.DataFrame(trimmed, columns=cols)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No one has been marked present yet today.")

    with c2:
        st.subheader("🔴 Absent Members")
        absent_rows = [(name, phone, cls) for (mid, name, phone, cls) in members if mid not in present_map]
        if absent_rows:
            cols = ["Name", "Phone", "Class"] if not scope_class else ["Name", "Phone"]
            trimmed = absent_rows if not scope_class else [(n, p) for (n, p, c) in absent_rows]
            df = pd.DataFrame(trimmed, columns=cols)
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download absentee list (CSV)", csv, "absent_members.csv", "text/csv")
        else:
            if total > 0:
                st.success("Everyone is present today! 🎉")
            else:
                st.info("No members registered yet.")

    if total > 0:
        st.markdown("---")
        col_a, col_b = st.columns([1, 1.4])
        with col_a:
            st.subheader("Today's Split")
            fig = px.pie(
                names=["Present", "Absent"],
                values=[present_count, absent_count],
                color=["Present", "Absent"],
                color_discrete_map={"Present": "#22C55E", "Absent": "#EF4444"},
                hole=0.55,
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True, height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Attendance History (last 7 days)")
            history = database.get_attendance_history(days=7, class_section=scope_class)
            if history:
                hist_df = pd.DataFrame(history, columns=["Date", "Present"])
                fig2 = px.bar(hist_df, x="Date", y="Present", color_discrete_sequence=["#6366F1"])
                fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No attendance history yet.")


# --------------------------------------------------------------------------------------
# Register new face page
# --------------------------------------------------------------------------------------
def register_page():
    st.markdown("<h1 style='color:#0F172A;'>🧑‍💻 Register New Student</h1>", unsafe_allow_html=True)
    st.write("Enter the student's details, then capture a few clear photos of their face "
             "(slightly different angles improve recognition accuracy).")

    is_teacher = st.session_state["role"] == "teacher"
    existing_classes = database.get_all_classes()

    if not st.session_state["pending_name"]:
        with st.form("member_form"):
            name = st.text_input("Full Name")
            phone = st.text_input("Phone Number")

            if is_teacher:
                st.text_input("Class / Section", value=st.session_state["teacher_class"], disabled=True)
                class_section = st.session_state["teacher_class"]
            else:
                class_mode = st.radio("Class / Section", ["Choose existing", "Create new"], horizontal=True) \
                    if existing_classes else "Create new"
                if class_mode == "Choose existing":
                    class_section = st.selectbox("Select class", existing_classes)
                else:
                    class_section = st.text_input("New class/section name (e.g. Grade 8 - B)")

            submitted_info = st.form_submit_button("Save Details & Continue to Capture", type="primary")

        if submitted_info:
            class_section = (class_section or "").strip()
            if not name.strip() or not phone.strip() or not class_section:
                st.error("Please enter name, phone number, and class/section.")
            elif database.phone_exists(phone.strip()):
                st.error(f"A student with phone number **{phone.strip()}** is already registered.")
            else:
                st.session_state["pending_name"] = name.strip()
                st.session_state["pending_phone"] = phone.strip()
                st.session_state["pending_class"] = class_section
                st.session_state["captured_images"] = []
                st.rerun()

    else:
        st.success(
            f"Now capturing photos for **{st.session_state['pending_name']}** "
            f"({st.session_state['pending_class']})"
        )
        num_captured = len(st.session_state["captured_images"])
        st.progress(min(num_captured / 5, 1.0), text=f"Photos captured: {num_captured} / 5 (minimum 3 required)")

        img = st.camera_input(f"Capture photo #{num_captured + 1}", key=f"cam_{num_captured}")
        if img is not None:
            st.session_state["captured_images"].append(Image.open(img))
            st.rerun()

        st.write("")
        colA, colB = st.columns(2)
        with colA:
            if num_captured >= 3:
                if st.button("✅ Finish & Train Model", type="primary", use_container_width=True):
                    dup_id = face_utils.find_matching_member(st.session_state["captured_images"])
                    if dup_id is not None:
                        dup_name = database.get_member_name(dup_id)
                        st.error(
                            f"❌ This face is already registered as **{dup_name}**. "
                            f"A student can only be registered once."
                        )
                    else:
                        member_id = database.add_member(
                            st.session_state["pending_name"],
                            st.session_state["pending_phone"],
                            st.session_state["pending_class"],
                        )
                        saved = face_utils.save_face_samples(member_id, st.session_state["captured_images"])
                        if saved == 0:
                            st.error("No face was detected in the captured images. Please retry with better lighting.")
                            database.delete_member(member_id)
                        else:
                            ok, msg = face_utils.train_model()
                            if ok:
                                st.success(f"🎉 {st.session_state['pending_name']} registered successfully with {saved} sample(s)!")
                                st.balloons()
                                st.session_state["pending_name"] = None
                                st.session_state["pending_phone"] = None
                                st.session_state["pending_class"] = None
                                st.session_state["captured_images"] = []
                            else:
                                st.error(msg)
            else:
                st.button("✅ Finish & Train Model", disabled=True, use_container_width=True,
                           help="Capture at least 3 photos first")
        with colB:
            if st.button("❌ Cancel Registration", use_container_width=True):
                st.session_state["pending_name"] = None
                st.session_state["pending_phone"] = None
                st.session_state["pending_class"] = None
                st.session_state["captured_images"] = []
                st.rerun()


# --------------------------------------------------------------------------------------
# Take attendance page
# --------------------------------------------------------------------------------------
def attendance_page():
    st.markdown("<h1 style='color:#0F172A;'>📸 Take Attendance</h1>", unsafe_allow_html=True)
    st.write("Capture a photo. Recognized faces get a green ring and are marked present; "
             "unrecognized faces get a red ring.")

    is_teacher = st.session_state["role"] == "teacher"
    scope_class = st.session_state["teacher_class"] if is_teacher else None

    all_members = {mid: (name, phone, cls) for mid, name, phone, cls in database.get_all_members()}
    if not all_members:
        st.warning("No students are registered yet. Go to **Register Face** first.")
        return

    if is_teacher:
        st.info(f"Marking attendance for class: **{scope_class}**")

    img = st.camera_input("Scan Face")
    if img is not None:
        pil_img = Image.open(img)
        with st.spinner("Analyzing face..."):
            results, error = face_utils.recognize_from_image(pil_img)

        if error:
            st.error(error)
            return
        if not results:
            st.warning("No face detected. Please try again with better lighting or move closer.")
            return

        r = results[0]
        mid = r["member_id"]
        x, y, w, h = r["bbox"]

        np_img = np.array(pil_img.convert("RGB"))
        cx, cy = x + w // 2, y + h // 2
        radius = max(w, h) // 2 + 20

        matched = r["matched"] and mid in all_members
        # if this session is scoped to a class, a match outside that class doesn't count
        if matched and is_teacher and all_members[mid][2] != scope_class:
            matched = False
            wrong_class = True
        else:
            wrong_class = False

        color = (34, 197, 94) if matched else (239, 68, 68)
        cv2.circle(np_img, (cx, cy), radius, color, 8)

        side = radius + 10
        top, bottom = max(0, cy - side), min(np_img.shape[0], cy + side)
        left, right = max(0, cx - side), min(np_img.shape[1], cx + side)
        cropped = np_img[top:bottom, left:right]

        ring_color = "#22C55E" if matched else "#EF4444"
        pulse_class = "scan-ring-live" if matched else ""
        st.markdown(
            f"""
            <div class="scan-ring-wrap" style='display:flex; justify-content:center; margin-top:10px;'>
                <div class="{pulse_class}" style='width:320px; height:320px; border-radius:50%; overflow:hidden;
                            border:6px solid {ring_color};
                            box-shadow:0 10px 30px rgba(15,23,42,0.25);'>
                    <img src="data:image/png;base64,{image_to_base64(cropped)}"
                         style="width:100%; height:100%; object-fit:cover;">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        if matched:
            name, phone, cls = all_members[mid]
            marked = database.mark_attendance(mid)
            if marked:
                st.success(f"✅ Attendance marked for **{name}** ({cls})")
            else:
                st.info(f"ℹ️ **{name}** was already marked present today.")
        elif wrong_class:
            name, phone, cls = all_members[mid]
            st.error(f"❌ **{name}** belongs to **{cls}**, not {scope_class}. Attendance not marked here.")
        else:
            st.error("❌ Face not recognized. This person doesn't appear to be registered.")


# --------------------------------------------------------------------------------------
# Manage members page
# --------------------------------------------------------------------------------------
def manage_page():
    st.markdown("<h1 style='color:#0F172A;'>🗂️ Manage Students</h1>", unsafe_allow_html=True)

    is_teacher = st.session_state["role"] == "teacher"
    if is_teacher:
        scope_class = st.session_state["teacher_class"]
        members = database.get_all_members(class_section=scope_class)
        st.caption(f"Showing students in **{scope_class}** only.")
    else:
        all_classes = ["All Classes"] + database.get_all_classes()
        choice = st.selectbox("Filter by class", all_classes)
        scope_class = None if choice == "All Classes" else choice
        members = database.get_all_members(class_section=scope_class)

    if not members:
        st.info("No students registered yet.")
        return

    st.write(f"**{len(members)}** student(s) found.")
    st.write("")

    for mid, name, phone, cls in members:
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        c1.write(f"**{name}**")
        c2.write(phone)
        c3.write(f"🏷️ {cls}")
        if c4.button("🗑️ Delete", key=f"del_{mid}", use_container_width=True):
            database.delete_member(mid)
            face_dir = os.path.join(face_utils.FACES_DIR, str(mid))
            if os.path.exists(face_dir):
                shutil.rmtree(face_dir)
            face_utils.train_model()
            st.rerun()
        st.divider()


# --------------------------------------------------------------------------------------
# Manage teachers page (admin only)
# --------------------------------------------------------------------------------------
def manage_teachers_page():
    st.markdown("<h1 style='color:#0F172A;'>🍎 Manage Teachers</h1>", unsafe_allow_html=True)
    st.write("Create teacher logins, each scoped to a single class/section.")

    existing_classes = database.get_all_classes()

    with st.container(border=True):
        with st.form("teacher_form"):
            col1, col2 = st.columns(2)
            with col1:
                t_name = st.text_input("Teacher Full Name")
                t_username = st.text_input("Login Username")
            with col2:
                t_password = st.text_input("Login Password", type="password")
                if existing_classes:
                    t_class_mode = st.radio("Class / Section", ["Choose existing", "Create new"], horizontal=True)
                else:
                    t_class_mode = "Create new"
                if t_class_mode == "Choose existing":
                    t_class = st.selectbox("Select class", existing_classes)
                else:
                    t_class = st.text_input("New class/section name (e.g. Grade 8 - B)")
            submitted = st.form_submit_button("➕ Create Teacher Account", type="primary")

        if submitted:
            t_username_clean = t_username.strip()
            t_class_clean = (t_class or "").strip()
            if not all([t_name.strip(), t_username_clean, t_password, t_class_clean]):
                st.error("Please fill in all fields.")
            elif database.username_exists(t_username_clean):
                st.error(f"Username **{t_username_clean}** is already taken.")
            else:
                database.add_teacher(t_username_clean, t_password, t_name.strip(), t_class_clean)
                st.success(f"Teacher account created for **{t_name.strip()}** ({t_class_clean}).")
                st.rerun()

    st.markdown("---")
    st.subheader("Existing Teacher Accounts")
    teachers = database.get_all_teachers()
    if not teachers:
        st.info("No teacher accounts yet.")
        return

    for tid, username, name, cls in teachers:
        c1, c2, c3, c4 = st.columns([2.5, 2, 2.5, 1])
        c1.write(f"**{name}**")
        c2.write(f"@{username}")
        c3.write(f"🏷️ {cls}")
        if c4.button("🗑️ Remove", key=f"delteacher_{tid}", use_container_width=True):
            database.delete_teacher(tid)
            st.rerun()
        st.divider()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    if not st.session_state["logged_in"]:
        login_page()
        return

    is_admin = st.session_state["role"] == "admin"
    role_label = "Admin" if is_admin else f"Teacher · {st.session_state['teacher_class']}"
    badge_color = "#4F46E5" if is_admin else "#22C55E"

    header_l, header_r = st.columns([6, 1])
    with header_l:
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; gap:14px; padding:4px 4px 14px 4px;'>
                <div style='display:flex; align-items:center; gap:10px;'>
                    <span style='font-size:30px; display:inline-block; animation: floatIcon 3s ease-in-out infinite;'>🎓</span>
                    <span class="app-title-text" style='font-size:22px; font-weight:800;'>FaceTrack Classroom</span>
                </div>
                <span class='role-badge' style='background:{badge_color}22; color:{badge_color};'>{role_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_r:
        theme_toggle_button()

    if is_admin:
        options = ["Dashboard", "Register Face", "Take Attendance", "Manage Students", "Manage Teachers", "Logout"]
        icons = ["bar-chart-line", "person-plus", "camera", "people", "person-badge", "box-arrow-right"]
    else:
        options = ["Dashboard", "Register Face", "Take Attendance", "Manage Students", "Logout"]
        icons = ["bar-chart-line", "person-plus", "camera", "people", "box-arrow-right"]

    is_dark = st.session_state["theme"] == "dark"
    nav_bg = "#1E293B" if is_dark else "#FFFFFF"
    nav_shadow = "0 4px 16px rgba(0,0,0,0.35)" if is_dark else "0 4px 16px rgba(15,23,42,0.08)"
    nav_text = "#CBD5E1" if is_dark else "#334155"
    nav_hover = "#334155" if is_dark else "#EEF2FF"

    selected = option_menu(
        menu_title=None,
        options=options,
        icons=icons,
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "8px 10px",
                "background-color": nav_bg,
                "border-radius": "14px",
                "box-shadow": nav_shadow,
                "margin-bottom": "24px",
                "transition": "background-color 0.3s ease",
            },
            "icon": {"color": "#818CF8" if is_dark else "#4F46E5", "font-size": "15px"},
            "nav-link": {
                "font-size": "14px",
                "font-weight": "600",
                "text-align": "center",
                "margin": "0 4px",
                "border-radius": "10px",
                "color": nav_text,
                "padding": "10px 16px",
                "--hover-color": nav_hover,
            },
            "nav-link:hover": {"background-color": nav_hover},
            "nav-link-selected": {"background-color": "#4F46E5", "color": "#FFFFFF"},
        },
    )

    if selected == "Dashboard":
        dashboard_page()
    elif selected == "Register Face":
        register_page()
    elif selected == "Take Attendance":
        attendance_page()
    elif selected == "Manage Students":
        manage_page()
    elif selected == "Manage Teachers":
        manage_teachers_page()
    elif selected == "Logout":
        st.session_state["logged_in"] = False
        st.session_state["role"] = None
        st.session_state["teacher_name"] = None
        st.session_state["teacher_class"] = None
        st.rerun()


if __name__ == "__main__":
    main()