import html
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components


# --- Page & Backend Configuration ---
st.set_page_config(
    page_title="RTU Senior Tutor",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://keshavagarwal779-rtu-exam-assistant.hf.space",
).rstrip("/")

STREAMS: List[str] = ["CSE", "IT", "ECE", "EE", "ME", "CE"]
YEARS: List[int] = [1, 2, 3, 4]
YEAR_LABELS: Dict[int, str] = {
    1: "1st Year",
    2: "2nd Year",
    3: "3rd Year",
    4: "4th Year",
}

_YEAR1_COMMON_SUBJECTS: List[Tuple[str, str]] = [
    ("BEE", "Basic Electrical Engineering"),
    ("CS-ENG", "Communication Skills"),
    ("EM-1", "Engineering Mathematics-I"),
    ("PHY", "Engineering Physics"),
    ("BCE", "Basic Civil Engineering"),
    ("BME", "Basic Mechanical Engineering"),
    ("CHEM", "Engineering Chemistry"),
    ("EM-2", "Engineering Mathematics-II"),
    ("HV", "Human Values & Professional Ethics"),
]

_CSE_IT_YEAR_SUBJECTS: Dict[int, List[Tuple[str, str]]] = {
    1: _YEAR1_COMMON_SUBJECTS,
    2: [
        ("AEM", "Advanced Engineering Mathematics"),
        ("DE", "Digital Electronics"),
        ("DSA", "Data Structures and Algorithms"),
        ("MEFA", "Managerial Economics and Financial Accounting"),
        ("OOP", "Object Oriented Programming (C++)"),
        ("SE", "Software Engineering"),
        ("DBMS", "Database Management System"),
        ("DCCN", "Data Communication and Computer Networks"),
        ("DMS", "Discrete Mathematics Structure"),
        ("POC", "Principles of Communication"),
        ("TC", "Technical Communication"),
        ("TOC", "Theory of Computation"),
    ],
    3: [
        ("AOA", "Analysis of Algorithms"),
        ("CN", "Computer Networks"),
        ("ITC", "Information Theory and Coding"),
        ("MI", "Microprocessor and Interfaces"),
        ("OS", "Operating Systems"),
        ("AI", "Artificial Intelligence"),
        ("CA", "Computer Architecture and Organization"),
        ("CC", "Cloud Computing"),
        ("CD", "Compiler Design"),
        ("CG", "Computer Graphics & Multimedia"),
    ],
    4: [
        ("BDA", "Big Data Analytics"),
        ("IOT", "Internet of Things"),
        ("ISS", "Information System Security"),
        ("ML", "Machine Learning"),
        ("BC", "Blockchain Technologies"),
        ("DIP", "Digital Image Processing"),
        ("DM", "Disaster Management"),
        ("DS", "Distributed Systems"),
    ],
}

FALLBACK_SUBJECTS: Dict[str, Dict[int, List[Tuple[str, str]]]] = {
    "CSE": _CSE_IT_YEAR_SUBJECTS,
    "IT": _CSE_IT_YEAR_SUBJECTS,
    "ECE": {
        1: _YEAR1_COMMON_SUBJECTS,
        2: [("DE", "Digital Electronics")],
        3: [("MI", "Microprocessor and Interfaces")],
        4: [],
    },
    "EE": {
        1: _YEAR1_COMMON_SUBJECTS,
        2: [("DE", "Digital Electronics")],
        3: [("MI", "Microprocessor and Interfaces")],
        4: [],
    },
    "ME": {1: _YEAR1_COMMON_SUBJECTS, 2: [], 3: [], 4: []},
    "CE": {1: _YEAR1_COMMON_SUBJECTS, 2: [], 3: [], 4: []},
}


# --- HTTP Client Helpers ---

def _extract_error_detail(resp: requests.Response) -> str:
    """Extracts readable error messages from FastAPI JSON responses."""
    try:
        data = resp.json()
        detail = data.get("detail")
        if isinstance(detail, list) and detail:
            parts = []
            for item in detail:
                loc = item.get("loc") or []
                field = str(loc[-1]) if loc else ""
                msg = str(item.get("msg", "")).strip()
                parts.append(f"{field}: {msg}" if field and field != "body" else msg)
            return "; ".join(p for p in parts if p)
        if detail:
            return str(detail)
        if isinstance(data, dict) and data.get("error"):
            err_val = data["error"]
            if isinstance(err_val, dict):
                return str(err_val.get("message") or err_val)
            return str(err_val)
        if isinstance(data, dict) and data.get("message"):
            return str(data["message"])
    except ValueError:
        pass
    text = (resp.text or "").strip()
    return text[:200] if text else f"Request failed with status {resp.status_code}"


def api_request(
    method: str,
    endpoint: str,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 90,
) -> Tuple[Optional[Any], Optional[str]]:
    """Sends an HTTP request to the FastAPI backend and returns (data, error_message)."""
    url = f"{DEFAULT_BACKEND_URL}/api{endpoint}"
    try:
        resp = requests.request(
            method=method.upper(),
            url=url,
            json=payload,
            params=params,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        return None, "Request timed out while waiting for the backend server."
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to backend server at {DEFAULT_BACKEND_URL}."
    except requests.exceptions.RequestException as exc:
        return None, f"Network error: {exc}"

    if resp.status_code >= 400:
        return None, _extract_error_detail(resp)

    try:
        return resp.json(), None
    except ValueError:
        ctype = (resp.headers.get("content-type") or "").lower()
        if "text/html" in ctype or resp.text.lstrip().lower().startswith("<!doctype"):
            return None, f"Backend at {DEFAULT_BACKEND_URL} returned HTML instead of JSON for /api{endpoint}."
        return {"raw": resp.text}, None


# --- Session State & Curriculum Helpers ---

def init_session_state() -> None:
    """Initializes persistent Streamlit session state keys cleanly."""
    defaults: Dict[str, Any] = {
        "auth_user": None,
        "guest_user_id": f"guest_{uuid.uuid4().hex[:10]}",
        "active_auth_modal": None,
        "auth_modal_error": None,
        "session_id": None,
        "messages": [],
        "feedback_map": {},
        "conversations": [],
        "subjects_cache": {},
        "selected_stream": "CSE",
        "selected_year": 2,
        "selected_subject_label": "All Subjects",
        "pending_stream": None,
        "pending_year": None,
        "pending_subject_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def active_user_id() -> str:
    """Returns logged-in user_id or the current session's guest_user_id."""
    user = st.session_state.auth_user
    if user and user.get("user_id"):
        return str(user["user_id"])
    return str(st.session_state.guest_user_id)


def _apply_pending_selectors() -> None:
    """Applies deferred Stream/Year updates before rendering sidebar selectboxes."""
    if st.session_state.pending_stream in STREAMS:
        st.session_state.selected_stream = st.session_state.pending_stream
    st.session_state.pending_stream = None

    if st.session_state.pending_year in YEARS:
        st.session_state.selected_year = st.session_state.pending_year
    st.session_state.pending_year = None


def get_subjects_for_filter(stream: str, year: int) -> List[Dict[str, str]]:
    """Fetches subjects filtered by Stream and Year from backend with local fallback."""
    cache_key = f"{stream}:{year}"
    cached = st.session_state.subjects_cache.get(cache_key)
    if cached is not None:
        return cached

    data, err = api_request(
        "GET",
        "/subjects",
        params={"branch": stream, "year": year},
        timeout=15,
    )
    items: List[Dict[str, str]] = []
    if not err and isinstance(data, list):
        for row in data:
            short = str(row.get("short_name") or row.get("subject_short") or "").strip()
            name = str(row.get("name") or short).strip()
            subj_id = str(row.get("subject_id") or "").strip()
            if short:
                items.append({
                    "subject_id": subj_id,
                    "short_name": short,
                    "name": name,
                    "label": f"{short} — {name}",
                })
        st.session_state.subjects_cache[cache_key] = items
        return items

    for short, name in FALLBACK_SUBJECTS.get(stream, {}).get(year, []):
        items.append({
            "subject_id": f"SUBJ_{short.replace('-', '')}",
            "short_name": short,
            "name": name,
            "label": f"{short} — {name}",
        })
    return items


def refresh_conversations() -> None:
    """Refreshes the active user or guest's conversation history list from backend."""
    data, err = api_request(
        "GET",
        "/conversations",
        params={"limit": 50, "user_id": active_user_id()},
        timeout=20,
    )
    if not err and isinstance(data, list):
        st.session_state.conversations = data


def start_new_chat() -> None:
    """Resets the active thread while keeping current authentication intact."""
    st.session_state.session_id = None
    st.session_state.messages = []


def load_conversation(conversation_id: str) -> None:
    """Loads a saved conversation and its messages into the active workspace."""
    data, err = api_request("GET", f"/conversations/{conversation_id}", timeout=20)
    if err or not isinstance(data, dict):
        st.error(err or "Could not load conversation.")
        return

    st.session_state.session_id = data.get("session_id") or conversation_id
    st.session_state.messages = list(data.get("messages") or [])
    if data.get("branch") in STREAMS:
        st.session_state.pending_stream = data["branch"]
    if data.get("year") in YEARS:
        st.session_state.pending_year = data["year"]
    if data.get("subject_id"):
        st.session_state.pending_subject_id = data["subject_id"]
    st.rerun()


def delete_single_conversation(conversation_id: str) -> None:
    """Deletes one saved conversation and resets the view if it was currently open."""
    _, err = api_request("DELETE", f"/conversations/{conversation_id}", timeout=20)
    if err:
        st.error(err)
        return
    if st.session_state.session_id == conversation_id:
        start_new_chat()
    refresh_conversations()
    st.rerun()


def delete_all_history() -> None:
    """Deletes all saved conversations belonging to the current active account or guest."""
    _, err = api_request(
        "DELETE",
        "/conversations",
        params={"user_id": active_user_id()},
        timeout=20,
    )
    if err:
        st.error(err)
        return
    start_new_chat()
    st.session_state.conversations = []
    st.toast("Chat history deleted.")
    st.rerun()


def _close_auth_modal() -> None:
    """Dismisses the sidebar auth dialog cleanly."""
    st.session_state.active_auth_modal = None
    st.session_state.auth_modal_error = None


def _complete_login(profile: Dict[str, Any]) -> None:
    """Populates auth state cleanly from login or registration response."""
    st.session_state.auth_user = profile
    st.session_state.active_auth_modal = None
    st.session_state.auth_modal_error = None
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.feedback_map = {}
    st.session_state.conversations = []
    branch = str(profile.get("branch") or "CSE").upper()
    if branch in STREAMS:
        st.session_state.pending_stream = branch
    year_val = profile.get("year")
    if isinstance(year_val, int) and year_val in YEARS:
        st.session_state.pending_year = year_val
    refresh_conversations()
    st.rerun()


def logout_user() -> None:
    """Logs out and resets to an isolated empty guest session cleanly."""
    st.session_state.auth_user = None
    st.session_state.guest_user_id = f"guest_{uuid.uuid4().hex[:10]}"
    st.session_state.active_auth_modal = None
    st.session_state.auth_modal_error = None
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.feedback_map = {}
    st.session_state.conversations = []
    st.session_state.pending_subject_id = None
    st.rerun()


# --- Minimal Dark Gemini / ChatGPT Styling ---

def inject_gemini_dark_css() -> None:
    """Applies the dark charcoal Gemini/ChatGPT layout styling."""
    st.markdown(
        """
        <style>
        #MainMenu, footer {
            visibility: hidden;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #0b0b0c !important;
            color: #e8eaed !important;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 1.5rem !important;
            max-width: 780px !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #171718 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        [data-testid="stSidebarContent"] {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            padding-top: 0.75rem !important;
        }
        [data-testid="stSidebarUserContent"] {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            padding: 0.6rem 0.75rem 0.9rem 0.75rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.45rem !important;
        }
        [data-testid="stSidebar"] .stSelectbox label p {
            font-size: 0.76rem !important;
            color: #9aa0a6 !important;
            margin-bottom: 0.1rem !important;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
            background-color: #202123 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
            min-height: 36px !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stSidebar"] button {
            text-align: left !important;
            justify-content: flex-start !important;
            border-radius: 10px !important;
            border: none !important;
            font-size: 0.86rem !important;
            min-height: 36px !important;
            padding: 0.35rem 0.65rem !important;
            transition: background-color 0.15s ease;
        }
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: transparent !important;
            color: #d7dade !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: #26272a !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #26282b !important;
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background-color: #2e3034 !important;
        }

        .sidebar-history-hdr {
            font-size: 0.78rem;
            font-weight: 600;
            color: #9aa0a6;
            padding: 0.35rem 0.25rem 0.1rem 0.25rem;
        }
        .sidebar-account-row {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.45rem 0.35rem 0.35rem 0.35rem;
        }
        .sidebar-avatar {
            width: 32px;
            height: 32px;
            border-radius: 999px;
            background: #2b2d31;
            color: #e8eaed;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.84rem;
            font-weight: 600;
            flex-shrink: 0;
        }
        .sidebar-user-meta {
            line-height: 1.25;
            overflow: hidden;
        }
        .sidebar-user-name {
            font-size: 0.86rem;
            font-weight: 600;
            color: #eceff1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .sidebar-user-sub {
            font-size: 0.74rem;
            color: #9aa0a6;
        }

        /* Hide noisy default Streamlit chat avatars & boxy frames */
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"],
        [data-testid^="chatAvatarIcon"] {
            display: none !important;
        }
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            border: none !important;
            padding: 0.45rem 0 !important;
        }

        /* User right-aligned rounded pill matching Gemini dark view */
        .user-bubble-wrap {
            display: flex;
            justify-content: flex-end;
            width: 100%;
            margin: 0.15rem 0;
        }
        .user-bubble {
            background-color: #242527;
            color: #f1f3f4;
            padding: 0.65rem 1.15rem;
            border-radius: 22px;
            max-width: 78%;
            font-size: 0.96rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* Empty welcome screen */
        .welcome-empty {
            text-align: center;
            margin: 22vh auto 2rem auto;
            max-width: 540px;
        }
        .welcome-empty h2 {
            font-size: 1.9rem;
            font-weight: 500;
            color: #e8eaed;
            margin-bottom: 0.45rem;
            letter-spacing: -0.02em;
        }
        .welcome-empty p {
            font-size: 0.88rem;
            color: #9aa0a6;
            margin: 0;
        }

        /* Inline feedback buttons under assistant replies */
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] button {
            background: transparent !important;
            border: none !important;
            color: #ffffff !important;
            padding: 0.15rem 0.4rem !important;
            min-height: 30px !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] button span,
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] button svg {
            color: #ffffff !important;
            fill: currentColor !important;
        }
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] button:hover {
            background: rgba(255, 255, 255, 0.12) !important;
            color: #ffffff !important;
        }
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] button[kind="primary"] {
            background: rgba(255, 255, 255, 0.18) !important;
            color: #ffffff !important;
        }

        /* Bottom pill chat input */
        [data-testid="stBottom"] {
            background-color: #0b0b0c !important;
        }
        [data-testid="stBottom"] > div {
            background-color: #0b0b0c !important;
            padding-bottom: 0.45rem !important;
        }
        [data-testid="stBottom"] > div::after {
            content: "RTU Senior Tutor uses past-year papers and can make mistakes.";
            display: block;
            text-align: center;
            font-size: 0.74rem;
            color: #8e918f;
            margin-top: 0.35rem;
        }
        [data-testid="stChatInput"] {
            max-width: 780px !important;
            margin: 0 auto !important;
        }
        [data-testid="stChatInput"] > div {
            background-color: #1e1f20 !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 28px !important;
            padding: 0.25rem 0.5rem !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #e8eaed !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_copy_icon_button(text: str, button_id: str) -> None:
    """Renders a compact borderless clipboard copy icon matching the assistant toolbar."""
    escaped_text = json.dumps(text)
    widget_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{
        margin: 0;
        padding: 0;
        background: transparent;
        display: flex;
        align-items: center;
        height: 30px;
      }}
      .copy-btn {{
        background: transparent;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        transition: background 0.15s, color 0.15s;
      }}
      .copy-btn:hover {{
        background: rgba(255, 255, 255, 0.12);
        color: #ffffff;
      }}
    </style>
    </head>
    <body>
      <button id="btn_{button_id}" class="copy-btn" title="Copy" aria-label="Copy">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>
      <script>
        const textToCopy = {escaped_text};
        const btn = document.getElementById("btn_{button_id}");
        const checkSvg = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#8ab4f8" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        const copySvg = btn.innerHTML;
        btn.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText(textToCopy);
          }} catch (e) {{
            const ta = document.createElement("textarea");
            ta.value = textToCopy;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
          }}
          btn.innerHTML = checkSvg;
          setTimeout(() => {{ btn.innerHTML = copySvg; }}, 1400);
        }});
      </script>
    </body>
    </html>
    """
    components.html(widget_html, height=32)


def _submit_feedback(message_id: str, trace_id: Optional[str], rating: str) -> None:
    """Sends thumbs-up/down verdict for an assistant message to the backend."""
    _, err = api_request(
        "POST",
        "/feedback",
        payload={
            "session_id": st.session_state.session_id,
            "message_id": message_id,
            "trace_id": trace_id,
            "rating": rating,
        },
        timeout=15,
    )
    if err:
        st.toast(f"Feedback error: {err}")
        return
    st.session_state.feedback_map[message_id] = rating
    st.toast("Marked as correct." if rating == "up" else "Marked as incorrect.")


def render_assistant_footer(msg: Dict[str, Any], index: int) -> None:
    """Renders compact Thumbs Up, Thumbs Down, and Copy icons below an assistant reply."""
    content = str(msg.get("content") or "")
    msg_id = msg.get("message_id")
    trace_id = msg.get("trace_id")

    if msg_id:
        col_up, col_down, col_copy, _ = st.columns([0.055, 0.055, 0.055, 0.835])
        current = st.session_state.feedback_map.get(msg_id)
        with col_up:
            if st.button(
                "",
                icon=":material/thumb_up:",
                key=f"fb_up_{index}_{msg_id}",
                type="primary" if current == "up" else "secondary",
                help="Good response",
            ):
                _submit_feedback(msg_id, trace_id, "up")
                st.rerun()
        with col_down:
            if st.button(
                "",
                icon=":material/thumb_down:",
                key=f"fb_down_{index}_{msg_id}",
                type="primary" if current == "down" else "secondary",
                help="Bad response",
            ):
                _submit_feedback(msg_id, trace_id, "down")
                st.rerun()
        with col_copy:
            render_copy_icon_button(content, f"msg_{index}_{msg_id}")
    else:
        col_copy, _ = st.columns([0.055, 0.945])
        with col_copy:
            render_copy_icon_button(content, f"msg_{index}")


# --- Auth Modals (Opened from Sidebar Bottom-Left) ---

@st.dialog("Log In", width="small", on_dismiss=_close_auth_modal)
def show_login_dialog() -> None:
    """Modal login dialog triggered from bottom-left sidebar."""
    if st.session_state.auth_modal_error:
        st.error(st.session_state.auth_modal_error)

    with st.form("modal_login_form", clear_on_submit=False):
        identifier = st.text_input(
            "Roll Number or Email",
            key="login_identifier",
            placeholder="e.g. 22EGJCS001 or student@rtu.ac.in",
        )
        password = st.text_input("Password", type="password", key="login_password")
        col_submit, col_switch = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)
        with col_switch:
            go_signup = st.form_submit_button("Create account", use_container_width=True)

    if go_signup:
        st.session_state.auth_modal_error = None
        st.session_state.active_auth_modal = "signup"
        st.rerun()

    if submitted:
        if not identifier.strip() or not password:
            st.session_state.auth_modal_error = "Please enter both your roll number/email and password."
            st.rerun()
        profile, err = api_request(
            "POST",
            "/auth/login",
            payload={"roll_no_or_email": identifier.strip(), "password": password},
            timeout=20,
        )
        if err or not isinstance(profile, dict):
            st.session_state.auth_modal_error = err or "Invalid roll number/email or password."
            st.rerun()
        _complete_login(profile)


@st.dialog("Sign Up", width="small", on_dismiss=_close_auth_modal)
def show_signup_dialog() -> None:
    """Modal signup dialog triggered from bottom-left sidebar."""
    if st.session_state.auth_modal_error:
        st.error(st.session_state.auth_modal_error)

    with st.form("modal_signup_form", clear_on_submit=False):
        name = st.text_input("Full Name", key="signup_name", placeholder="e.g. Aarav Sharma")
        roll_no = st.text_input("RTU Roll Number", key="signup_roll", placeholder="e.g. 22EGJCS045")
        email = st.text_input("Email (Optional)", key="signup_email", placeholder="e.g. aarav@rtu.ac.in")
        col_stream, col_year = st.columns(2)
        with col_stream:
            stream = st.selectbox("Stream", STREAMS, index=0, key="signup_stream")
        with col_year:
            year = st.selectbox(
                "Year",
                YEARS,
                index=1,
                key="signup_year",
                format_func=lambda y: YEAR_LABELS[y],
            )
        password = st.text_input("Password", type="password", key="signup_password", help="Minimum 4 characters")
        col_submit, col_switch = st.columns(2)
        with col_submit:
            signed_up = st.form_submit_button("Sign Up", type="primary", use_container_width=True)
        with col_switch:
            go_login = st.form_submit_button("Already have account", use_container_width=True)

    if go_login:
        st.session_state.auth_modal_error = None
        st.session_state.active_auth_modal = "login"
        st.rerun()

    if signed_up:
        if len(name.strip()) < 2:
            st.session_state.auth_modal_error = "Name must be at least 2 characters."
            st.rerun()
        if len(roll_no.strip()) < 2:
            st.session_state.auth_modal_error = "Roll number must be at least 2 characters."
            st.rerun()
        if len(password) < 4:
            st.session_state.auth_modal_error = "Password must be at least 4 characters."
            st.rerun()
        profile, err = api_request(
            "POST",
            "/auth/register",
            payload={
                "name": name.strip(),
                "roll_no": roll_no.strip(),
                "email": email.strip() or None,
                "branch": stream,
                "year": year,
                "password": password,
            },
            timeout=20,
        )
        if err or not isinstance(profile, dict):
            st.session_state.auth_modal_error = err or "Registration failed."
            st.rerun()
        _complete_login(profile)


# --- Sidebar & Workspace Renderers ---

def render_sidebar() -> Tuple[str, int, Optional[str]]:
    """Renders sidebar: New chat, Stream/Year/Subject, History + Delete, and bottom-left Auth."""
    _apply_pending_selectors()
    auth_user = st.session_state.auth_user

    with st.sidebar:
        if st.button("New chat", key="new_chat_btn", icon=":material/edit_square:", use_container_width=True):
            start_new_chat()
            st.rerun()

        stream = st.selectbox("Stream", STREAMS, key="selected_stream")
        year = st.selectbox(
            "Year",
            YEARS,
            key="selected_year",
            format_func=lambda y: YEAR_LABELS[y],
        )

        subjects = get_subjects_for_filter(stream, year)
        subject_options = ["All Subjects"] + [item["label"] for item in subjects]
        label_to_short = {item["label"]: item["short_name"] for item in subjects}
        id_to_label = {item["subject_id"]: item["label"] for item in subjects}

        if st.session_state.pending_subject_id:
            matched_label = id_to_label.get(st.session_state.pending_subject_id)
            if matched_label in subject_options:
                st.session_state.selected_subject_label = matched_label
            st.session_state.pending_subject_id = None

        if st.session_state.selected_subject_label not in subject_options:
            st.session_state.selected_subject_label = "All Subjects"

        subject_label = st.selectbox("Subject", subject_options, key="selected_subject_label")
        subject_short = label_to_short.get(subject_label)

        conversations = st.session_state.conversations
        col_hdr, col_clr = st.columns([0.74, 0.26])
        with col_hdr:
            st.markdown('<div class="sidebar-history-hdr">Chats</div>', unsafe_allow_html=True)
        with col_clr:
            if conversations and st.button(
                "🗑",
                key="delete_all_history_btn",
                help="Delete all chat history",
                use_container_width=True,
            ):
                delete_all_history()

        if not conversations:
            st.caption("No chats yet.")
        else:
            for conv in conversations:
                conv_id = conv.get("conversation_id") or conv.get("session_id")
                if not conv_id:
                    continue
                title = str(conv.get("title") or "Untitled chat").strip()
                is_active = conv_id == st.session_state.session_id

                col_title, col_del = st.columns([0.82, 0.18])
                with col_title:
                    if st.button(
                        title,
                        key=f"hist_open_{conv_id}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                    ):
                        load_conversation(conv_id)
                with col_del:
                    if st.button(
                        "✕",
                        key=f"hist_del_{conv_id}",
                        help="Delete this chat",
                        use_container_width=True,
                    ):
                        delete_single_conversation(conv_id)

        st.markdown('<div style="flex: 1 1 auto; min-height: 1.5rem;"></div>', unsafe_allow_html=True)
        st.divider()

        # Bottom-left Account / Auth Bar
        if auth_user:
            display_name = str(auth_user.get("name") or "Student")
            initial = display_name[:1].upper()
            user_branch = auth_user.get("branch") or stream
            user_year = auth_user.get("year") or year
            st.markdown(
                f"""
                <div class="sidebar-account-row">
                    <div class="sidebar-avatar">{html.escape(initial)}</div>
                    <div class="sidebar-user-meta">
                        <div class="sidebar-user-name">{html.escape(display_name)}</div>
                        <div class="sidebar-user-sub">{html.escape(str(user_branch))} · Year {html.escape(str(user_year))}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Log Out", key="logout_btn", icon=":material/logout:", use_container_width=True):
                logout_user()
        else:
            col_login, col_signup = st.columns(2)
            with col_login:
                if st.button("Log In", key="open_login_btn", use_container_width=True):
                    st.session_state.auth_modal_error = None
                    st.session_state.active_auth_modal = "login"
                    st.rerun()
            with col_signup:
                if st.button("Sign Up", key="open_signup_btn", use_container_width=True):
                    st.session_state.auth_modal_error = None
                    st.session_state.active_auth_modal = "signup"
                    st.rerun()

    return stream, year, subject_short


def render_citations(
    citations: List[Dict[str, Any]],
    related_topics: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Renders compact collapsible matched RTU syllabus topics and past-paper citations."""
    related_topics = related_topics or []
    total_items = len(citations) + len(related_topics)
    if total_items == 0:
        return
    label_parts = []
    if related_topics:
        label_parts.append(f"{len(related_topics)} Topics")
    if citations:
        label_parts.append(f"{len(citations)} Papers")
    with st.expander(f"Sources ({' · '.join(label_parts)})", expanded=False):
        if related_topics:
            st.markdown("**Related RTU Syllabus Topics:**")
            for t in related_topics:
                subj = t.get("subject") or "RTU"
                unit_t = t.get("unit_title") or f"Unit {t.get('unit_no', '')}"
                t_name = t.get("topic_name") or ""
                st.markdown(f"- **{subj}** ({unit_t}) — {t_name}")
        if citations:
            if related_topics:
                st.markdown("**Matched Past-Paper Questions:**")
            for cit in citations:
                subj = cit.get("subject") or "RTU"
                yr = cit.get("year") or ""
                sec = cit.get("section") or ""
                marks = cit.get("marks")
                pdf = cit.get("source_pdf") or ""
                page = cit.get("page_number")
                meta_parts = [p for p in [subj, str(yr) if yr else "", sec, f"{marks}M" if marks else ""] if p]
                location = f"{pdf} (p. {page})" if pdf and page else pdf
                st.markdown(f"- **{' · '.join(meta_parts)}** — `{location}`" if location else f"- **{' · '.join(meta_parts)}**")


def render_chat_workspace(stream: str, year: int, subject_short: Optional[str]) -> None:
    """Renders the dark full-screen Gemini/ChatGPT conversation canvas."""
    active_subject = subject_short or "All Subjects"

    if not st.session_state.messages:
        st.markdown(
            f"""
            <div class="welcome-empty">
                <h2>What can I help you with today?</h2>
                <p>{html.escape(stream)} · {html.escape(YEAR_LABELS[year])} · {html.escape(active_subject)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for idx, msg in enumerate(st.session_state.messages):
            role = msg.get("role", "assistant")
            content = str(msg.get("content") or "")
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(
                        f'<div class="user-bubble-wrap"><div class="user-bubble">{html.escape(content)}</div></div>',
                        unsafe_allow_html=True,
                    )
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
                    render_citations(msg.get("citations") or [], msg.get("related_topics") or [])
                    render_assistant_footer(msg, idx)

    prompt = st.chat_input("Ask any RTU exam question...")
    if not prompt or not prompt.strip():
        return

    user_text = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-bubble-wrap"><div class="user-bubble">{html.escape(user_text)}</div></div>',
            unsafe_allow_html=True,
        )

    payload: Dict[str, Any] = {
        "message": user_text,
        "session_id": st.session_state.session_id,
        "user_id": active_user_id(),
        "branch": stream,
        "year": year,
    }
    if subject_short:
        payload["filters"] = {"subject": subject_short}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp_data, err = api_request("POST", "/chat", payload=payload, timeout=120)

        if err or not isinstance(resp_data, dict):
            error_reply = f"Error: {err or 'Failed to generate response.'}"
            st.error(error_reply)
            st.session_state.messages.append({"role": "assistant", "content": error_reply})
            return

        answer = str(resp_data.get("response") or resp_data.get("answer") or "")
        citations = list(resp_data.get("citations") or [])
        related_topics = list(resp_data.get("related_topics") or [])
        assistant_msg = resp_data.get("assistant_message") or {}
        message_id = assistant_msg.get("message_id") or resp_data.get("message_id")
        trace_id = resp_data.get("trace_id")

        if resp_data.get("session_id"):
            st.session_state.session_id = resp_data["session_id"]

        st.markdown(answer)
        render_citations(citations, related_topics)
        assistant_entry: Dict[str, Any] = {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "related_topics": related_topics,
            "message_id": message_id,
            "trace_id": trace_id,
        }
        st.session_state.messages.append(assistant_entry)
        render_assistant_footer(assistant_entry, len(st.session_state.messages) - 1)

    refresh_conversations()
    st.rerun()


def main() -> None:
    init_session_state()
    inject_gemini_dark_css()

    if not st.session_state.conversations:
        refresh_conversations()

    stream, year, subject_short = render_sidebar()
    render_chat_workspace(stream, year, subject_short)

    if not st.session_state.auth_user:
        if st.session_state.active_auth_modal == "login":
            show_login_dialog()
        elif st.session_state.active_auth_modal == "signup":
            show_signup_dialog()


if __name__ == "__main__":
    main()
