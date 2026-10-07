from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.auth import init_db, register_user, login_user


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SleepCare AI",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DATABASE
# ============================================================

init_db()


# ============================================================
# CUSTOM HTML HELPER
# ============================================================

def render_html(content: str):
    """
    Render custom HTML using Streamlit's native HTML renderer.
    """
    st.html(content)


# ============================================================
# GLOBAL CSS
# ============================================================

render_html(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html,
    body,
    [class*="css"],
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background: #F8FAFC !important;
        color: #0F172A !important;
    }

    #MainMenu,
    footer,
    header {
        visibility: hidden !important;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
    }

    .block-container {
        max-width: 1180px !important;
        padding-top: 28px !important;
        padding-bottom: 50px !important;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .top-header {
        width: 100%;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 16px 22px;
        margin-bottom: 30px;
        box-shadow: 0 6px 24px rgba(15, 23, 42, 0.04);
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-badge {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            135deg,
            #6366F1 0%,
            #8B5CF6 100%
        );
        font-size: 23px;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.24);
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        color: #312E81;
        letter-spacing: -0.5px;
    }

    .user-pill {
        display: inline-block;
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #3730A3;
        padding: 9px 15px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        text-align: center;
        white-space: nowrap;
    }


    /* ========================================================
       PAGE INTRO
       ======================================================== */

    .page-eyebrow {
        color: #6366F1;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .page-title {
        color: #0F172A;
        font-size: 34px;
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }

    .page-subtitle {
        color: #64748B;
        font-size: 15px;
        line-height: 1.65;
        max-width: 800px;
        margin-bottom: 25px;
    }


    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-badge {
        display: inline-block;
        padding: 8px 14px;
        margin-top: 22px;
        margin-bottom: 15px;
        border-radius: 10px;
        background: #EEF2FF;
        border: 1px solid #DDD6FE;
        color: #4338CA;
        font-size: 13px;
        font-weight: 800;
    }


    /* ========================================================
       INPUTS
       ======================================================== */

    label {
        color: #334155 !important;
        font-size: 13px !important;
        font-weight: 700 !important;
    }

    .stTextInput input,
    .stNumberInput input {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 11px !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    }

    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border-color: #CBD5E1 !important;
        border-radius: 11px !important;
        color: #0F172A !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 12px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        min-height: 46px !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(
            135deg,
            #6366F1 0%,
            #8B5CF6 100%
        ) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 7px 18px rgba(99, 102, 241, 0.22) !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(99, 102, 241, 0.28) !important;
    }

    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #475569 !important;
        border: 1px solid #CBD5E1 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #94A3B8 !important;
    }


    /* ========================================================
       AUTH
       ======================================================== */

    .auth-page {
        max-width: 470px;
        margin: 45px auto 0 auto;
    }

    .auth-brand {
        text-align: center;
        margin-bottom: 25px;
    }

    .auth-logo {
        width: 64px;
        height: 64px;
        border-radius: 19px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            135deg,
            #6366F1,
            #8B5CF6
        );
        font-size: 31px;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.22);
        margin-bottom: 14px;
    }

    .auth-name {
        font-size: 27px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.7px;
    }

    .auth-tagline {
        color: #64748B;
        font-size: 14px;
        margin-top: 5px;
    }

    .auth-card-top {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-bottom: none;
        border-radius: 22px 22px 0 0;
        padding: 28px 30px 8px 30px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.05);
    }

    .auth-card-title {
        font-size: 21px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }

    .auth-card-subtitle {
        color: #64748B;
        font-size: 13px;
        line-height: 1.55;
    }

    .auth-form {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: none;
        border-radius: 0 0 22px 22px;
        padding: 10px 30px 30px 30px;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.05);
    }

    .auth-footer {
        text-align: center;
        color: #64748B;
        font-size: 13px;
        margin-top: 18px;
    }


    /* ========================================================
       CONTENT CARDS
       ======================================================== */

    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.025);
    }

    .content-card-title {
        color: #0F172A;
        font-size: 16px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .content-card-text {
        color: #64748B;
        font-size: 13px;
        line-height: 1.6;
    }


    /* ========================================================
       PREDICTION
       ======================================================== */

    .prediction-card {
        background: linear-gradient(
            135deg,
            #EEF2FF 0%,
            #FAF5FF 100%
        );
        border: 1px solid #C7D2FE;
        border-radius: 24px;
        padding: 35px 25px;
        text-align: center;
        margin: 30px 0 20px 0;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.08);
    }

    .prediction-icon {
        font-size: 54px;
        line-height: 1;
        margin-bottom: 14px;
    }

    .prediction-label {
        color: #4F46E5;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .prediction-value {
        color: #1E1B4B;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin-bottom: 10px;
    }

    .prediction-description {
        max-width: 650px;
        margin: auto;
        color: #475569;
        font-size: 14px;
        line-height: 1.65;
    }


    /* ========================================================
       INPUT SUMMARY
       ======================================================== */

    .why-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 23px;
        margin: 18px 0;
    }

    .why-title {
        color: #0F172A;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .why-subtitle {
        color: #64748B;
        font-size: 12px;
        line-height: 1.55;
        margin-bottom: 17px;
    }

    .observation {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 14px;
        color: #334155;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 9px;
    }


    /* ========================================================
       EDUCATIONAL CARDS
       ======================================================== */

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 22px;
        min-height: 150px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.025);
    }

    .info-card-icon {
        font-size: 28px;
        margin-bottom: 10px;
    }

    .info-card-title {
        color: #0F172A;
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .info-card-text {
        color: #64748B;
        font-size: 12px;
        line-height: 1.6;
    }


    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 17px 18px;
        color: #64748B;
        font-size: 11px;
        line-height: 1.6;
        margin-top: 25px;
    }

    .disclaimer strong {
        color: #334155;
    }


    /* ========================================================
       RESPONSIVE
       ======================================================== */

    @media (max-width: 700px) {

        .block-container {
            padding-left: 18px !important;
            padding-right: 18px !important;
        }

        .page-title {
            font-size: 28px;
        }

        .auth-card-top,
        .auth-form {
            padding-left: 20px;
            padding-right: 20px;
        }

    }

    </style>
    """
)


# ============================================================
# MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "sleep_quality_model.pkl"

STRESS_MAPPING = {
    "Very Low": 2.0,
    "Low": 4.0,
    "Moderate": 5.5,
    "High": 7.0,
    "Very High": 9.0,
}


@st.cache_resource
def load_model():
    """
    Load the saved SleepCare model once and reuse it.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model_data = load_model()

pipeline = model_data["model"]
features = model_data["features"]


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "login"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""


# ============================================================
# AUTH RESULT HANDLING
# ============================================================

def get_user_name(user):
    """
    Safely extract a name from dictionary, tuple, or string.
    """

    if user is None:
        return "User"

    if isinstance(user, dict):
        return str(
            user.get("name")
            or user.get("username")
            or user.get("email")
            or "User"
        )

    if isinstance(user, tuple):

        if len(user) >= 2:
            return str(user[1])

        if len(user) == 1:
            return str(user[0])

    if isinstance(user, str):
        return user

    return "User"


def normalize_login_result(result):
    """
    Convert different login_user() return formats into:

        success, user
    """

    if isinstance(result, bool):
        return result, None

    if isinstance(result, dict):
        return True, result

    if isinstance(result, tuple):

        if len(result) == 0:
            return False, None

        if isinstance(result[0], bool):

            success = result[0]

            if not success:
                return False, None

            if len(result) >= 2:
                return True, result[1]

            return True, None

        if len(result) >= 2:

            return True, {
                "id": result[0],
                "name": result[1],
                "email": result[2] if len(result) >= 3 else "",
            }

    return False, None


def normalize_register_result(result):
    """
    Convert register_user() results into:

        success, message
    """

    if isinstance(result, tuple):

        if len(result) >= 2:
            return bool(result[0]), str(result[1])

        if len(result) == 1:
            return bool(result[0]), ""

    if isinstance(result, bool):
        return result, ""

    if isinstance(result, str):
        return True, result

    return False, "Unable to create account."


# ============================================================
# NAVIGATION
# ============================================================

def navigate(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# HEADER
# ============================================================

def render_header():

    col1, col2 = st.columns([4, 1])

    with col1:

        render_html(
            """
            <div class="top-header">

                <div class="brand-wrap">

                    <div class="brand-badge">
                        🌙
                    </div>

                    <div class="brand-title">
                        SleepCare AI
                    </div>

                </div>

            </div>
            """
        )

    with col2:

        if st.session_state.logged_in:

            render_html(
                f"""
                <div style="
                    background:#FFFFFF;
                    border:1px solid #E2E8F0;
                    border-radius:20px;
                    padding:16px 10px;
                    text-align:center;
                    box-shadow:0 6px 24px rgba(15,23,42,0.04);
                ">

                    <div class="user-pill">
                        👤 {st.session_state.user_name}
                    </div>

                </div>
                """
            )

            if st.button(
                "Logout",
                key="logout_button",
                use_container_width=True,
                type="secondary",
            ):

                st.session_state.logged_in = False
                st.session_state.user_name = ""
                st.session_state.page = "login"

                st.rerun()


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    render_html(
        """
        <div class="auth-page">

            <div class="auth-brand">

                <div class="auth-logo">
                    🌙
                </div>

                <div class="auth-name">
                    SleepCare AI
                </div>

                <div class="auth-tagline">
                    Sleep Health Analytics
                </div>

            </div>

            <div class="auth-card-top">

                <div class="auth-card-title">
                    Welcome back
                </div>

                <div class="auth-card-subtitle">
                    Sign in to continue to your SleepCare dashboard.
                </div>

            </div>

            <div class="auth-form">
        """
    )

    email = st.text_input(
        "Email address",
        placeholder="name@example.com",
        key="login_email",
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password",
        key="login_password",
    )

    if st.button(
        "Sign In",
        type="primary",
        use_container_width=True,
        key="login_button",
    ):

        email_clean = email.strip().lower()

        if not email_clean:

            st.error("Please enter your email address.")

        elif not password:

            st.error("Please enter your password.")

        else:

            result = login_user(
                email_clean,
                password,
            )

            success, user = normalize_login_result(result)

            if success:

                st.session_state.logged_in = True
                st.session_state.user_name = get_user_name(user)
                st.session_state.page = "prediction"

                st.rerun()

            else:

                st.error(
                    "Invalid email or password."
                )

    render_html(
        """
            </div>
        </div>

        <div class="auth-footer">
            Don't have an account?
        </div>
        """
    )

    if st.button(
        "Create an Account",
        use_container_width=True,
        key="go_register",
    ):

        navigate("register")


# ============================================================
# REGISTER PAGE
# ============================================================

def register_page():

    render_html(
        """
        <div class="auth-page">

            <div class="auth-brand">

                <div class="auth-logo">
                    🌙
                </div>

                <div class="auth-name">
                    SleepCare AI
                </div>

                <div class="auth-tagline">
                    Create your SleepCare account
                </div>

            </div>

            <div class="auth-card-top">

                <div class="auth-card-title">
                    Create your account
                </div>

                <div class="auth-card-subtitle">
                    Set up your account to access sleep quality predictions.
                </div>

            </div>

            <div class="auth-form">
        """
    )

    name = st.text_input(
        "Full name",
        placeholder="Enter your full name",
        key="register_name",
    )

    email = st.text_input(
        "Email address",
        placeholder="name@example.com",
        key="register_email",
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="At least 6 characters",
        key="register_password",
    )

    confirm_password = st.text_input(
        "Confirm password",
        type="password",
        placeholder="Re-enter your password",
        key="register_confirm_password",
    )

    if st.button(
        "Create Account",
        type="primary",
        use_container_width=True,
        key="register_button",
    ):

        name_clean = name.strip()
        email_clean = email.strip().lower()

        if not name_clean:

            st.error("Please enter your name.")

        elif not email_clean:

            st.error("Please enter your email address.")

        elif "@" not in email_clean or "." not in email_clean:

            st.error("Please enter a valid email address.")

        elif not password:

            st.error("Please enter a password.")

        elif len(password) < 6:

            st.error(
                "Password must contain at least 6 characters."
            )

        elif password != confirm_password:

            st.error("Passwords do not match.")

        else:

            result = register_user(
                name_clean,
                email_clean,
                password,
            )

            success, message = normalize_register_result(
                result
            )

            if success:

                st.success(
                    message
                    or "Account created successfully."
                )

                st.session_state.page = "login"

                st.rerun()

            else:

                st.error(
                    message
                    or "Unable to create account."
                )

    render_html(
        """
            </div>
        </div>

        <div class="auth-footer">
            Already have an account?
        </div>
        """
    )

    if st.button(
        "Back to Login",
        use_container_width=True,
        key="back_login",
    ):

        navigate("login")


# ============================================================
# PREDICTION PAGE
# ============================================================

def prediction_page():

    render_header()

    render_html(
        """
        <div class="page-eyebrow">
            MACHINE LEARNING DEMO
        </div>

        <div class="page-title">
            Understand your sleep quality
        </div>

        <div class="page-subtitle">
            Enter your sleep and lifestyle information below.
            SleepCare uses a machine-learning model to classify
            the information into Poor, Average, or Good sleep quality.
        </div>
        """
    )


    # ========================================================
    # YOUR INFORMATION
    # ========================================================

    render_html(
        """
        <div class="section-badge">
            Your information
        </div>
        """
    )

    render_html(
        """
        <div class="content-card">

            <div class="content-card-title">
                Sleep and lifestyle details
            </div>

            <div class="content-card-text">
                Provide the information as accurately as possible.
            </div>

        </div>
        """
    )


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male",
                "Other",
            ],
        )

    with col3:

        occupation = st.selectbox(
            "Occupation",
            [
                "Student",
                "Engineer",
                "Teacher",
                "Doctor",
                "Nurse",
                "Manager",
                "Business",
                "Other",
            ],
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=22.0,
            step=0.1,
            format="%.2f",
        )

    with col2:

        country = st.selectbox(
            "Country",
            [
                "India",
                "USA",
                "UK",
                "Canada",
                "Australia",
                "Germany",
                "Other",
            ],
        )

    with col3:

        chronotype = st.selectbox(
            "Chronotype",
            [
                "Morning",
                "Neutral",
                "Evening",
            ],
        )


    # ========================================================
    # SLEEP DETAILS
    # ========================================================

    render_html(
        """
        <div class="section-badge">
            Sleep details
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        sleep_duration_hrs = st.number_input(
            "Sleep duration (hours)",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.25,
            format="%.2f",
        )

    with col2:

        rem_percentage = st.number_input(
            "REM sleep (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            format="%.2f",
        )

    with col3:

        deep_sleep_percentage = st.number_input(
            "Deep sleep (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            format="%.2f",
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        sleep_latency_mins = st.number_input(
            "Time to fall asleep (minutes)",
            min_value=0,
            max_value=180,
            value=20,
            step=1,
        )

    with col2:

        wake_episodes_per_night = st.number_input(
            "Wake episodes per night",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
        )

    with col3:

        nap_duration_mins = st.number_input(
            "Nap duration (minutes)",
            min_value=0,
            max_value=300,
            value=20,
            step=5,
        )


    # ========================================================
    # LIFESTYLE
    # ========================================================

    render_html(
        """
        <div class="section-badge">
            Lifestyle
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        caffeine_mg_before_bed = st.number_input(
            "Caffeine before bed (mg)",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=10.0,
            format="%.2f",
        )

    with col2:

        alcohol_units_before_bed = st.number_input(
            "Alcohol before bed (units)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.5,
            format="%.2f",
        )

    with col3:

        screen_time_before_bed_mins = st.number_input(
            "Screen time before bed (minutes)",
            min_value=0,
            max_value=600,
            value=60,
            step=5,
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        exercise = st.selectbox(
            "Exercise today",
            [
                "Yes",
                "No",
            ],
        )

    with col2:

        steps_that_day = st.number_input(
            "Steps today",
            min_value=0,
            max_value=50000,
            value=7000,
            step=500,
        )

    with col3:

        stress_level = st.selectbox(
            "Stress level",
            [
                "Very Low",
                "Low",
                "Moderate",
                "High",
                "Very High",
            ],
        )


    col1, col2 = st.columns(2)

    with col1:

        work_hours_that_day = st.number_input(
            "Work/study hours today",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.5,
            format="%.2f",
        )

    with col2:

        heart_rate_resting_bpm = st.number_input(
            "Resting heart rate (BPM)",
            min_value=40.0,
            max_value=150.0,
            value=70.0,
            step=1.0,
            format="%.2f",
        )


    # ========================================================
    # INPUT MAPPING
    # ========================================================

    stress_score = STRESS_MAPPING[stress_level]

    exercise_day = 1 if exercise == "Yes" else 0


    # ========================================================
    # VALIDATION
    # ========================================================

    validation_errors = []

    if sleep_duration_hrs <= 0:
        validation_errors.append(
            "Sleep duration must be greater than 0 hours."
        )

    if rem_percentage + deep_sleep_percentage > 100:
        validation_errors.append(
            "REM sleep and deep sleep percentages "
            "cannot exceed 100% combined."
        )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    st.write("")

    if st.button(
        "✨ Predict Sleep Quality",
        type="primary",
        use_container_width=True,
        key="predict_button",
    ):

        if validation_errors:

            for error in validation_errors:
                st.error(error)

            st.stop()


        # ====================================================
        # CREATE INPUT DATA
        # ====================================================

        input_data = pd.DataFrame(
            [
                {
                    "age": age,
                    "gender": gender,
                    "occupation": occupation,
                    "bmi": bmi,
                    "country": country,
                    "sleep_duration_hrs": sleep_duration_hrs,
                    "rem_percentage": rem_percentage,
                    "deep_sleep_percentage": deep_sleep_percentage,
                    "sleep_latency_mins": sleep_latency_mins,
                    "wake_episodes_per_night": wake_episodes_per_night,
                    "caffeine_mg_before_bed": caffeine_mg_before_bed,
                    "alcohol_units_before_bed": alcohol_units_before_bed,
                    "screen_time_before_bed_mins": screen_time_before_bed_mins,
                    "exercise_day": exercise_day,
                    "steps_that_day": steps_that_day,
                    "nap_duration_mins": nap_duration_mins,
                    "stress_score": stress_score,
                    "work_hours_that_day": work_hours_that_day,
                    "chronotype": chronotype,
                    "heart_rate_resting_bpm": heart_rate_resting_bpm,
                }
            ]
        )

        input_data = input_data[features]


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = pipeline.predict(input_data)[0]


        # ====================================================
        # RESULT DESIGN
        # ====================================================

        if prediction == "Good":
            icon = "🌟"
        elif prediction == "Average":
            icon = "🌙"
        else:
            icon = "💤"


        render_html(
            f"""
            <div class="prediction-card">

                <div class="prediction-icon">
                    {icon}
                </div>

                <div class="prediction-label">
                    Sleep quality classification
                </div>

                <div class="prediction-value">
                    {prediction}
                </div>

                <div class="prediction-description">
                    Based on the information you entered,
                    the model classified your sleep quality
                    as <strong>{prediction}</strong>.
                </div>

            </div>
            """
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        observations = []

        if sleep_duration_hrs < 7:

            observations.append(
                "😴 Your reported sleep duration is below 7 hours."
            )

        elif sleep_duration_hrs <= 9:

            observations.append(
                "😴 Your reported sleep duration is between 7 and 9 hours."
            )

        else:

            observations.append(
                "😴 Your reported sleep duration is above 9 hours."
            )


        if stress_score >= 7:

            observations.append(
                "🧘 Your reported stress level is high."
            )

        elif stress_score >= 5:

            observations.append(
                "🧘 Your reported stress level is moderate."
            )

        else:

            observations.append(
                "🧘 Your reported stress level is relatively low."
            )


        if screen_time_before_bed_mins >= 120:

            observations.append(
                "📱 You reported 2 hours or more of screen time before bed."
            )

        elif screen_time_before_bed_mins >= 60:

            observations.append(
                "📱 You reported between 1 and 2 hours of screen time before bed."
            )

        else:

            observations.append(
                "📱 You reported less than 1 hour of screen time before bed."
            )


        if wake_episodes_per_night >= 4:

            observations.append(
                "🌙 You reported several wake episodes during the night."
            )


        if caffeine_mg_before_bed >= 200:

            observations.append(
                "☕ You reported a higher amount of caffeine before bed."
            )


        if exercise_day == 0:

            observations.append(
                "🏃 No exercise was reported for today."
            )


        observations = observations[:3]


        render_html(
            """
            <div class="why-card">

                <div class="why-title">
                    🔎 Your input summary
                </div>

                <div class="why-subtitle">
                    These observations summarize some of the values you entered.
                    They are shown for educational purposes and are not medical advice.
                </div>
            """
        )


        for observation in observations:

            render_html(
                f"""
                <div class="observation">
                    {observation}
                </div>
                """
            )


        render_html(
            """
            </div>
            """
        )


        # ====================================================
        # EDUCATIONAL INFORMATION
        # ====================================================

        render_html(
            """
            <div class="section-badge">
                Understanding the inputs
            </div>
            """
        )


        info1, info2, info3 = st.columns(3)


        with info1:

            render_html(
                """
                <div class="info-card">

                    <div class="info-card-icon">
                        😴
                    </div>

                    <div class="info-card-title">
                        Sleep Duration
                    </div>

                    <div class="info-card-text">
                        Sleep duration is one of the variables
                        represented in this machine-learning model.
                    </div>

                </div>
                """
            )


        with info2:

            render_html(
                """
                <div class="info-card">

                    <div class="info-card-icon">
                        🧘
                    </div>

                    <div class="info-card-title">
                        Stress Balance
                    </div>

                    <div class="info-card-text">
                        Stress is one of the lifestyle variables
                        represented in this machine-learning model.
                    </div>

                </div>
                """
            )


        with info3:

            render_html(
                """
                <div class="info-card">

                    <div class="info-card-icon">
                        📱
                    </div>

                    <div class="info-card-title">
                        Evening Screen Time
                    </div>

                    <div class="info-card-text">
                        Screen time before bed is included as
                        one of the model's input features.
                    </div>

                </div>
                """
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        render_html(
            """
            <div class="disclaimer">

                <strong>Educational use only:</strong>

                SleepCare is a machine-learning demonstration
                project and is not a medical diagnostic tool.
                Predictions are based on patterns learned from
                the project dataset and should not be treated
                as medical advice.

            </div>
            """
        )


# ============================================================
# APPLICATION ROUTING
# ============================================================

if not st.session_state.logged_in:

    if st.session_state.page == "register":

        register_page()

    else:

        login_page()

else:

    prediction_page()