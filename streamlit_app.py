"""
RTU AI Senior Tutor & Exam Coach - Streamlit Production App
A dedicated academic assistant and revision planner for Rajasthan Technical University (RTU) B.Tech students.
Powered by 14 Years of RTU Past Question Papers (2012-2026), FAISS Vector Search, BM25 Indexing, and OpenAI/Groq.
Fully compatible with Streamlit Community Cloud (Self-Contained Embedded Engine) and Local/Remote FastAPI backends.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import requests
import streamlit as st

# --- Page Configuration (Zero Emojis) ---
st.set_page_config(
    page_title="RTU Senior Tutor & Exam Coach",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Ensure Backend Module Resolution ---
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_PATH = PROJECT_ROOT / "rtu-ai-assistant" / "backend"
if BACKEND_PATH.exists() and str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

# --- Default Backend Configuration (Live Hugging Face Cloud API) ---
DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "https://keshavagarwal779-rtu-exam-assistant.hf.space")
DEFAULT_BACKEND_MODE = os.getenv("BACKEND_MODE", "Hugging Face Space Backend (Cloud API)")

# --- Canonical RTU Curriculum Mapping (Year -> Semester -> Subjects) ---
RTU_CURRICULUM = {
    1: {
        1: [
            ("BEE", "Basic Electrical Engineering"),
            ("CS-ENG", "Communication Skills"),
            ("EM-1", "Engineering Mathematics-I"),
            ("PHY", "Engineering Physics")
        ],
        2: [
            ("BCE", "Basic Civil Engineering"),
            ("BME", "Basic Mechanical Engineering"),
            ("CHEM", "Engineering Chemistry"),
            ("EM-2", "Engineering Mathematics-II"),
            ("HV", "Human Values")
        ]
    },
    2: {
        3: [
            ("DSA", "Data Structures and Algorithms"),
            ("OOP", "Object Oriented Programming (C++)"),
            ("DE", "Digital Electronics"),
            ("AEM", "Advanced Engineering Mathematics"),
            ("SE", "Software Engineering"),
            ("MEFA", "Managerial Economics and Financial Accounting")
        ],
        4: [
            ("DBMS", "Database Management System"),
            ("TOC", "Theory of Computation"),
            ("DCCN", "Computer Networks"),
            ("DMS", "Discrete Mathematics Structure"),
            ("TC", "Technical Communication"),
            ("POC", "Principle of Communication")
        ]
    },
    3: {
        5: [
            ("OS", "Operating Systems"),
            ("CN", "Computer Networks"),
            ("AOA", "Analysis of Algorithms"),
            ("MI", "Microprocessor & Interfaces")
        ],
        6: [
            ("AI", "Artificial Intelligence"),
            ("CD", "Compiler Design"),
            ("CC", "Cloud Computing"),
            ("CG", "Computer Graphics"),
            ("DSR", "Data Science with R")
        ]
    },
    4: {
        7: [
            ("ML", "Machine Learning"),
            ("IOT", "Internet of Things"),
            ("ISS", "Information System Security"),
            ("BDA", "Big Data Analytics")
        ],
        8: [
            ("DL", "Deep Learning"),
            ("BC", "Blockchain Architecture"),
            ("DIP", "Digital Image Processing"),
            ("DS", "Distributed Systems")
        ]
    }
}

# --- Dynamic Subject Quick Starters ---
SUBJECT_QUICK_STARTERS: Dict[str, List[Tuple[str, str]]] = {
    "DSA": [
        ("AVL Tree Rotations", "Explain AVL tree rotations with diagrams and RTU exam scoring tips."),
        ("Dijkstra Algorithm", "Explain Dijkstra algorithm with trace steps for a 10-mark question.")
    ],
    "DBMS": [
        ("Normalization (1NF-BCNF)", "Explain 1NF, 2NF, 3NF, and BCNF with student database examples for a 10-mark question."),
        ("ACID Properties & Transactions", "Explain ACID properties and concurrency control in DBMS with RTU past questions.")
    ],
    "BCE": [
        ("Surveying & Levelling", "Explain methods of levelling and contouring in Basic Civil Engineering with diagrams."),
        ("Types of Cement & Concrete", "Explain properties of concrete and types of cement as per RTU syllabus.")
    ],
    "BEE": [
        ("Thevenin & Norton Theorems", "Explain Thevenin's and Norton's theorems with solved step-by-step numerical."),
        ("Transformer Working Principle", "Explain single-phase transformer construction and working principle with EMF equation.")
    ],
    "OOP": [
        ("Polymorphism & Virtual Functions", "Explain virtual functions and dynamic binding in C++ with code example."),
        ("Constructors & Destructors", "Explain constructor types and copy constructors with C++ syntax.")
    ],
    "DE": [
        ("K-Map Simplification", "Explain Karnaugh Map (K-Map) minimization with a 4-variable example."),
        ("Flip-Flops (JK vs D)", "Compare JK and D flip-flops with state tables and excitation diagrams.")
    ],
    "OS": [
        ("CPU Scheduling (RR / SJF)", "Compare Round Robin and SJF CPU scheduling algorithms with Gantt chart example."),
        ("Deadlock & Banker's Algorithm", "Explain deadlock prevention and Banker's algorithm with resource allocation example.")
    ],
    "DCCN": [
        ("OSI vs TCP/IP Model", "Compare 7-layer OSI model and TCP/IP model with layer protocols."),
        ("Subnetting & IP Addressing", "Explain IPv4 addressing and classless subnetting with solved numerical.")
    ],
    "CN": [
        ("OSI vs TCP/IP Model", "Compare 7-layer OSI model and TCP/IP model with layer protocols."),
        ("Routing Algorithms", "Explain Distance Vector Routing and Link State Routing algorithms.")
    ],
    "TOC": [
        ("DFA Construction", "Explain DFA construction with state diagrams and regular expressions."),
        ("Turing Machine Design", "Explain Turing machine components with transition table and tape representation.")
    ],
    "SE": [
        ("SDLC Models (Waterfall vs Agile)", "Compare Waterfall and Agile SDLC models with pros and cons."),
        ("COCOMO Model Estimation", "Explain Basic, Intermediate, and Detailed COCOMO model for software estimation.")
    ],
    "AEM": [
        ("Differential Equations", "Explain methods to solve second-order differential equations with RTU past numerical."),
        ("Laplace Transforms", "Explain Laplace transform properties and inverse Laplace with solved example.")
    ],
    "EM-1": [
        ("Matrices & Eigenvalues", "Explain eigenvalues and Cayley-Hamilton theorem with solved RTU numerical."),
        ("Taylor & Maclaurin Series", "Explain Taylor series expansion with step-by-step example.")
    ],
    "EM-2": [
        ("Vector Calculus (Green/Gauss)", "Explain Gauss Divergence and Stokes theorems with verification numerical."),
        ("Fourier Series", "Explain Dirichlet conditions and Fourier series expansion for periodic functions.")
    ],
    "AI": [
        ("Heuristic Search (A* / AO*)", "Explain A* search algorithm with admissibility of heuristic function and RTU example."),
        ("Minimax & Alpha-Beta Pruning", "Explain Minimax algorithm and Alpha-Beta pruning with game tree trace.")
    ],
    "AOA": [
        ("Master Theorem & Recurrences", "Explain Master Theorem cases with examples and solve recurrence relations."),
        ("Dynamic Programming (0/1 Knapsack)", "Explain 0/1 Knapsack problem using dynamic programming with step-by-step table.")
    ],
    "BC": [
        ("Consensus Mechanisms (PoW vs PoS)", "Compare Proof of Work (PoW) and Proof of Stake (PoS) consensus mechanisms in Blockchain."),
        ("Smart Contracts & Ethereum", "Explain smart contracts lifecycle and EVM architecture in Ethereum.")
    ],
    "BDA": [
        ("Hadoop MapReduce Architecture", "Explain Hadoop architecture and MapReduce execution flow with diagram."),
        ("HDFS & Spark Ecosystem", "Explain HDFS architecture, block replication, and Spark Resilient Distributed Datasets (RDD).")
    ],
    "BME": [
        ("Carnot Cycle & Heat Engines", "Explain Carnot heat engine cycle with P-V and T-S diagrams and efficiency formula."),
        ("2-Stroke vs 4-Stroke Engines", "Compare 2-stroke and 4-stroke IC engines with working diagrams and valve timing.")
    ],
    "CC": [
        ("Cloud Service Models (IaaS/PaaS/SaaS)", "Compare IaaS, PaaS, and SaaS cloud computing delivery models with real-world examples."),
        ("Virtualization & Hypervisors", "Explain Type-1 and Type-2 hypervisors and virtualization techniques in cloud architecture.")
    ],
    "CD": [
        ("LR Parsing (SLR, CLR, LALR)", "Compare SLR, CLR, and LALR parsing tables with grammar construction steps."),
        ("Intermediate Code Generation", "Explain Three-Address Code (TAC), quadruples, and triples with syntax tree representation.")
    ],
    "CG": [
        ("Bresenham Line Algorithm", "Derive Bresenham line generation algorithm with integer arithmetic and trace example."),
        ("2D Geometric Transformations", "Explain 2D Translation, Rotation, and Scaling matrix representations with homogeneous coordinates.")
    ],
    "CHEM": [
        ("Water Hardness & EDTA Method", "Explain determination of hardness of water by EDTA method with chemical reactions."),
        ("Corrosion & Prevention", "Explain electrochemical mechanism of corrosion and methods of corrosion prevention.")
    ],
    "CS-ENG": [
        ("Formal Letter & Report Writing", "Explain the structure of technical reports and formal business letters as per RTU format."),
        ("Barriers to Communication", "Explain verbal, non-verbal, and psychological barriers to communication with remediation tips.")
    ],
    "DIP": [
        ("Histogram Equalization", "Explain histogram equalization for image enhancement with mathematical formulation."),
        ("Spatial Filtering & Edge Detection", "Explain Sobel, Prewitt, and Laplacian spatial filters for image edge detection.")
    ],
    "DL": [
        ("Backpropagation Algorithm", "Explain neural network backpropagation algorithm with mathematical weight update rules."),
        ("CNN Architecture & Pooling", "Explain Convolutional Neural Network (CNN) layers, feature maps, and pooling operations.")
    ],
    "DMS": [
        ("Graph Theory & Trees", "Explain planar graphs, Euler path, Hamiltonian circuits, and tree traversal theorems."),
        ("Pigeonhole Principle & Sets", "Explain Pigeonhole principle and mathematical induction with solved RTU discrete math problems.")
    ],
    "DS": [
        ("Remote Procedure Call (RPC)", "Explain RPC mechanism and client-server stub binding with architectural diagram."),
        ("Lamport Logical Clocks", "Explain Lamport timestamps and vector clocks for ordering events in distributed systems.")
    ],
    "DSR": [
        ("Data Frames & Vectors in R", "Explain R data structures: vectors, factors, data frames, and lists with code snippets."),
        ("Data Visualization with ggplot2", "Explain grammar of graphics and data visualization using ggplot2 in R.")
    ],
    "HV": [
        ("Self-Exploration Process", "Explain self-exploration as the process for value education with natural acceptance."),
        ("Harmony in Family & Society", "Explain 9 values in relationships (Trust, Respect, Affection) and harmony in society.")
    ],
    "IOT": [
        ("IoT Architecture & Protocols", "Explain 4-layer IoT architecture and compare MQTT vs CoAP protocols."),
        ("Sensors & Actuators in IoT", "Explain sensor interfacing, microcontroller roles, and edge computing in IoT applications.")
    ],
    "ISS": [
        ("RSA Public Key Cryptography", "Explain RSA algorithm step-by-step with key generation and encryption/decryption numerical."),
        ("DES & AES Comparison", "Compare Data Encryption Standard (DES) and Advanced Encryption Standard (AES) architectures.")
    ],
    "MEFA": [
        ("Law of Demand & Elasticity", "Explain Law of Demand, elasticity types, and demand forecasting methods with graphs."),
        ("Break-Even Analysis (BEP)", "Explain Break-Even Analysis with graphical representation and margin of safety formula.")
    ],
    "MI": [
        ("8085 Microprocessor Architecture", "Explain 8085 microprocessor architecture, registers, and bus structure with block diagram."),
        ("8085 Interrupts & Pinout", "Explain 8085 hardware and software interrupts with priority table and RIM/SIM instructions.")
    ],
    "ML": [
        ("Supervised vs Unsupervised ML", "Compare supervised, unsupervised, and reinforcement learning with real-world examples."),
        ("Decision Trees & Information Gain", "Explain ID3 Decision Tree algorithm, entropy, and information gain calculation.")
    ],
    "PHY": [
        ("Interference & Newton's Rings", "Derive formula for diameter of dark and bright rings in Newton's rings experiment."),
        ("Lasers & Einstein Coefficients", "Explain Einstein's A and B coefficients and population inversion for laser action.")
    ],
    "POC": [
        ("AM & FM Modulation", "Compare Amplitude Modulation (AM) and Frequency Modulation (FM) with frequency spectra."),
        ("Sampling Theorem & Nyquist Rate", "State and prove Nyquist-Shannon sampling theorem with aliasing effect diagram.")
    ],
    "TC": [
        ("Technical Report Structure", "Explain formal structure, components, and format of technical reports for RTU exams."),
        ("7 Cs of Effective Communication", "Explain the 7 Cs of effective communication with engineering workplace examples.")
    ]
}

def get_quick_starters_for_subject(subj_code: str) -> List[Tuple[str, str]]:
    if subj_code in SUBJECT_QUICK_STARTERS:
        return SUBJECT_QUICK_STARTERS[subj_code]
    return [
        ("10-Mark Model Question", f"Show me a standard 10-mark Part C question and model answer for {subj_code}."),
        ("Core Concepts & Formulas", f"Explain the core fundamental concepts, laws, and formulas of {subj_code}.")
    ]

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = None
if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL
if "backend_mode" not in st.session_state:
    st.session_state.backend_mode = DEFAULT_BACKEND_MODE
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Senior Tutor Chat"
if "student_year" not in st.session_state:
    st.session_state.student_year = 2
if "student_sem" not in st.session_state:
    st.session_state.student_sem = "Sem 3"
if "student_branch" not in st.session_state:
    st.session_state.student_branch = "CSE"
if "active_subject_code" not in st.session_state:
    st.session_state.active_subject_code = "DSA"

# --- Cached Embedded Engine Loader (Fallback Mode) ---
@st.cache_resource(show_spinner="Initializing RTU Academic Intelligence Engine (FAISS + BM25 + SQLite)...")
def get_embedded_client():
    """
    Initializes backend services and ASGI TestClient in-process.
    Enables Streamlit Community Cloud to run 100% self-contained if remote server is unreachable.
    """
    try:
        from app.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        client.__enter__()
        return client
    except Exception as e:
        st.error(f"Embedded engine initialization error: {e}")
        return None

# --- Unified API Dispatcher (Remote HTTP with In-Process Fallback) ---
def api_request(
    method: str,
    endpoint: str,
    backend_mode: str,
    remote_url: str,
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Dispatches request to the remote FastAPI cloud backend,
    with automatic seamless fallback to the in-process engine.
    """
    if backend_mode == "Embedded Engine (Streamlit Cloud / In-Process)":
        client = get_embedded_client()
        if client is None:
            return {"error": "Embedded engine is not available."}
        try:
            if method.upper() == "GET":
                resp = client.get(endpoint, params=params)
            else:
                resp = client.post(endpoint, json=json_data)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    else:
        url = f"{remote_url.rstrip('/')}{endpoint}"
        try:
            if method.upper() == "GET":
                resp = requests.get(url, params=params, timeout=timeout)
            else:
                resp = requests.post(url, json=json_data, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            # Automatic fallback to embedded engine if remote server fails
            client = get_embedded_client()
            if client is not None:
                try:
                    if method.upper() == "GET":
                        return client.get(endpoint, params=params).json()
                    else:
                        return client.post(endpoint, json=json_data).json()
                except Exception:
                    pass
            return {"error": f"Connection error: {str(e)}"}

def api_get(endpoint: str, backend_mode: str, remote_url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15):
    return api_request("GET", endpoint, backend_mode, remote_url, params=params, timeout=timeout)

def api_post(endpoint: str, backend_mode: str, remote_url: str, json_data: Dict[str, Any], timeout: int = 45):
    return api_request("POST", endpoint, backend_mode, remote_url, json_data=json_data, timeout=timeout)

# ==============================================================================
# SIDEBAR (Navigation & Quick Starters)
# ==============================================================================
nav_items = [
    "Senior Tutor Chat",
    "Crash Course Planner",
    "High-Yield Important Topics",
    "Practice Quiz & Evaluation",
    "Past Papers Archive"
]

with st.sidebar:
    st.title("RTU Senior Tutor")
    st.caption("Rajasthan Technical University Exam Coach")
    st.markdown("---")

    # 1. Navigation Menu
    st.subheader("Navigation")
    cur_nav_idx = nav_items.index(st.session_state.selected_page) if st.session_state.selected_page in nav_items else 0
    selected_page = st.radio(
        "Select Feature",
        nav_items,
        index=cur_nav_idx,
        label_visibility="collapsed"
    )
    st.session_state.selected_page = selected_page

    st.markdown("---")

    # 2. Quick Starters (Using Currently Active Subject)
    st.subheader("Quick Starters")
    active_subj = st.session_state.get("active_subject_code", "DSA")
    qs_topics = get_quick_starters_for_subject(active_subj)

    if st.button("6-Hour Crash Plan", use_container_width=True, key="qs_side_1"):
        st.session_state.selected_page = "Senior Tutor Chat"
        st.session_state.pending_prompt = f"I have 6 hours left for {active_subj} exam. Give me a crash revision roadmap."
        st.rerun()

    if st.button("Important Topics", use_container_width=True, key="qs_side_2"):
        st.session_state.selected_page = "Senior Tutor Chat"
        st.session_state.pending_prompt = f"What are the most important topics in {active_subj}?"
        st.rerun()

    if len(qs_topics) >= 1:
        t1_lbl, t1_prompt = qs_topics[0]
        if st.button(t1_lbl, use_container_width=True, key="qs_side_3"):
            st.session_state.selected_page = "Senior Tutor Chat"
            st.session_state.pending_prompt = t1_prompt
            st.rerun()

    if len(qs_topics) >= 2:
        t2_lbl, t2_prompt = qs_topics[1]
        if st.button(t2_lbl, use_container_width=True, key="qs_side_4"):
            st.session_state.selected_page = "Senior Tutor Chat"
            st.session_state.pending_prompt = t2_prompt
            st.rerun()

    st.markdown("---")

    # 3. Observability & Guardrails Status
    st.subheader("Observability & Safety")
    with st.expander("LangSmith Tracing", expanded=False):
        st.markdown("**Project**: `rtu-ai-academic-assistant`")
        st.markdown("**Status**: Active (Auto-instrumented)")
        st.markdown("**Screening Evaluators**:")
        st.markdown("- Syllabus Scope Gate")
        st.markdown("- Anti-Leakage Sanitizer")
        st.markdown("- 10-Mark Rubric Compliance")
        st.caption("Thumbs up/down feedback is synced to LangSmith runs.")

    # 4. Actions
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.pending_prompt = None
        st.rerun()

# ==============================================================================
# MAIN WORKSPACE HEADER
# ==============================================================================
st.title("RTU AI Academic Assistant & Exam Coach")
st.markdown(
    "Your dedicated Senior Tutor for **Rajasthan Technical University (RTU)** B.Tech examinations. "
    "Get instant 10-mark structured answers, time-constrained crash revision roadmaps, topic weightages, and practice quizzes."
)
st.markdown("---")

# Callbacks for strict hierarchy propagation: Year & Semester -> Branch -> Subject
def _sync_year_change():
    yr = st.session_state.selector_year
    st.session_state.student_year = yr
    st.session_state.student_sem = f"Sem {2 * yr - 1}"
    st.session_state.selector_sem = f"Sem {2 * yr - 1}"
    if "selector_subject" in st.session_state:
        del st.session_state["selector_subject"]
    st.session_state.active_subject_code = ""

def _sync_sem_change():
    st.session_state.student_sem = st.session_state.selector_sem
    if "selector_subject" in st.session_state:
        del st.session_state["selector_subject"]
    st.session_state.active_subject_code = ""

def _sync_branch_change():
    st.session_state.student_branch = st.session_state.selector_branch
    if "selector_subject" in st.session_state:
        del st.session_state["selector_subject"]
    st.session_state.active_subject_code = ""

# ==============================================================================
# STUDENT ACADEMIC PROFILE (Priority 1: Year & Semester, Priority 2: Branch, Priority 3: Subject)
# ==============================================================================
col_yr, col_sem, col_branch, col_subj = st.columns([1, 1, 1, 2])

# Priority 1: Academic Year Selection
with col_yr:
    year_options = [1, 2, 3, 4]
    cur_year = st.session_state.student_year
    year_idx = year_options.index(cur_year) if cur_year in year_options else 1
    selected_year = st.selectbox(
        "1. Year",
        year_options,
        index=year_idx,
        key="selector_year",
        on_change=_sync_year_change
    )

# Priority 1B: Semester Selection (Dynamic according to selected Year)
sem_map = {
    1: ["All Year 1", "Sem 1", "Sem 2"],
    2: ["All Year 2", "Sem 3", "Sem 4"],
    3: ["All Year 3", "Sem 5", "Sem 6"],
    4: ["All Year 4", "Sem 7", "Sem 8"]
}
avail_sems = sem_map.get(selected_year, ["All"])

with col_sem:
    cur_sem_idx = avail_sems.index(st.session_state.student_sem) if st.session_state.student_sem in avail_sems else 0
    selected_sem = st.selectbox(
        "2. Semester",
        avail_sems,
        index=cur_sem_idx,
        key="selector_sem",
        on_change=_sync_sem_change
    )

# Priority 2: Engineering Branch
branch_options = ["CSE", "IT", "ECE", "EE", "ME", "CE"]
with col_branch:
    cur_branch_idx = branch_options.index(st.session_state.student_branch) if st.session_state.student_branch in branch_options else 0
    selected_branch = st.selectbox(
        "3. Branch",
        branch_options,
        index=cur_branch_idx,
        key="selector_branch",
        on_change=_sync_branch_change
    )

# Resolve integer semester filter for querying subjects
int_sem: Optional[int] = None
if "Sem " in selected_sem:
    try:
        int_sem = int(selected_sem.replace("Sem ", "").strip())
    except ValueError:
        int_sem = None

# Query subjects filtered strictly by Year, Semester, and Branch
query_params: Dict[str, Any] = {"year": selected_year, "branch": selected_branch}
if int_sem is not None:
    query_params["semester"] = int_sem

curric_data = api_get("/api/subjects", st.session_state.backend_mode, st.session_state.backend_url, params=query_params, timeout=5)
subject_options: List[str] = []
subject_map: Dict[str, str] = {}

if isinstance(curric_data, list) and len(curric_data) > 0:
    for s in curric_data:
        p_cnt = s.get("paper_count", 0)
        status_tag = f"({p_cnt} papers)" if p_cnt > 0 else "(0 papers - Ingestion Pending)"
        lbl = f"{s.get('short_name', '')} - {s.get('name', '')} {status_tag}"
        subject_options.append(lbl)
        subject_map[lbl] = s.get("short_name", "")
else:
    # Guaranteed canonical fallback from RTU_CURRICULUM mapping
    fallback_subjects = []
    if selected_year in RTU_CURRICULUM:
        for sem_no, s_list in RTU_CURRICULUM[selected_year].items():
            if int_sem is None or sem_no == int_sem:
                fallback_subjects.extend(s_list)
    for s_code, s_name in fallback_subjects:
        lbl = f"{s_code} - {s_name} (0 papers - Ingestion Pending)"
        subject_options.append(lbl)
        subject_map[lbl] = s_code

# Priority 3: Subject Focus (Filtered strictly according to Year/Sem/Branch)
with col_subj:
    # Ensure current selector_subject is valid for new options
    if "selector_subject" in st.session_state and st.session_state.selector_subject not in subject_options:
        del st.session_state["selector_subject"]

    subj_idx = 0
    current_active_code = st.session_state.get("active_subject_code", "")
    for idx, lbl in enumerate(subject_options):
        if subject_map.get(lbl) == current_active_code:
            subj_idx = idx
            break

    def _sync_subject_change():
        sel_lbl = st.session_state.get("selector_subject")
        if sel_lbl and " - " in sel_lbl:
            st.session_state.active_subject_code = sel_lbl.split(" - ")[0].strip()

    selected_subject_label = st.selectbox(
        f"4. Subject Focus ({len(subject_options)} Available)",
        subject_options,
        index=subj_idx,
        key="selector_subject",
        on_change=_sync_subject_change
    )
    active_subject_code = subject_map.get(selected_subject_label, "DSA")
    if not active_subject_code and subject_options:
        active_subject_code = subject_options[0].split(" - ")[0].strip()
    st.session_state.active_subject_code = active_subject_code

# Improvement 4: Sidebar paper count caption (added after subject is resolved)
with st.sidebar:
    _paper_count = None
    if isinstance(curric_data, list):
        for _s in curric_data:
            if _s.get("short_name") == active_subject_code:
                _paper_count = _s.get("paper_count")
                break
    if _paper_count is not None:
        st.caption(f"{_paper_count} RTU past papers available for {active_subject_code}")

st.markdown("---")

# ==============================================================================
# PAGE 1: Senior Tutor Chat
# ==============================================================================
if st.session_state.selected_page == "Senior Tutor Chat":
    st.subheader(f"Chat with RTU Senior Tutor: {active_subject_code} (Year {selected_year}, {selected_branch})")
    st.caption("Ask any concept, numerical, algorithm, or ask for important repeated exam questions.")

    # Quick Starters Bar
    st.write("**Quick Starters:**")
    qs_c1, qs_c2, qs_c3, qs_c4 = st.columns(4)
    page_qs = get_quick_starters_for_subject(active_subject_code)

    with qs_c1:
        if st.button("6-Hour Crash Plan", use_container_width=True, key="qs_main_1"):
            st.session_state.pending_prompt = f"I have 6 hours left for {active_subject_code} exam. Give me a crash revision roadmap."
            st.rerun()
    with qs_c2:
        if st.button("Important Topics", use_container_width=True, key="qs_main_2"):
            st.session_state.pending_prompt = f"What are the most important topics in {active_subject_code}?"
            st.rerun()
    with qs_c3:
        if len(page_qs) >= 1:
            t1_lbl, t1_prompt = page_qs[0]
            if st.button(t1_lbl, use_container_width=True, key="qs_main_3"):
                st.session_state.pending_prompt = t1_prompt
                st.rerun()
    with qs_c4:
        if len(page_qs) >= 2:
            t2_lbl, t2_prompt = page_qs[1]
            if st.button(t2_lbl, use_container_width=True, key="qs_main_4"):
                st.session_state.pending_prompt = t2_prompt
                st.rerun()

    # Render Conversation History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations") and len(msg["citations"]) > 0:
                hist_citations = msg["citations"]
                hist_years = [c.get("year") for c in hist_citations if c.get("year") and str(c.get("year")).isdigit()]
                h_pids = set(c.get("source_pdf") or c.get("paper_id") for c in hist_citations if (c.get("source_pdf") or c.get("paper_id")))
                h_src_cnt = len(h_pids)
                h_paper_str = f"{len(hist_citations)} questions from {h_src_cnt} {'paper' if h_src_cnt == 1 else 'papers'}"
                if hist_years:
                    h_min = min(int(y) for y in hist_years)
                    h_max = max(int(y) for y in hist_years)
                    h_range = str(h_min) if h_min == h_max else f"{h_min}-{h_max}"
                    hist_label = f"RTU Past Paper References ({h_paper_str}, {h_range})"
                else:
                    hist_label = f"RTU Past Paper References ({h_paper_str})"
                with st.expander(hist_label):
                    for c_idx, c in enumerate(hist_citations):
                        st.markdown(
                            f"**{c_idx + 1}. {c.get('paper_name', 'RTU Past Paper')}** (Year: {c.get('year', '-')}) | "
                            f"*Question ID:* `{c.get('question_id', '-')}` | "
                            f"*Marks:* **{c.get('marks', '-')}**"
                        )
                        if c.get("question_text"):
                            st.caption(f"\"{c.get('question_text')}\"")

            if msg["role"] == "assistant":
                msg_id = msg.get("message_id", f"msg_{idx}")
                trace_id = msg.get("trace_id")
                screening = msg.get("screening", {})

                # LangSmith Tracing & Academic Screening Status
                if trace_id or screening:
                    sc_ok = screening.get("is_safe", True) and screening.get("is_in_syllabus", True)
                    sc_badge = "Screening: Passed" if sc_ok else "Screening: Guardrail Active"
                    trace_display = f"`{trace_id[:12]}...`" if trace_id else "Local"
                    st.caption(
                        f"LangSmith Trace: {trace_display} | {sc_badge} "
                        f"(Scope: {screening.get('scope_score', 1.0):.1f}, "
                        f"Format: {screening.get('format_score', 1.0):.1f})"
                    )

                feedback_val = st.feedback("thumbs", key=f"fb_{msg_id}_{idx}")
                if feedback_val is not None and not msg.get("feedback_submitted"):
                    rating_score = 1 if feedback_val == 1 else -1
                    api_post(
                        "/api/feedback",
                        st.session_state.backend_mode,
                        st.session_state.backend_url,
                        {
                            "session_id": st.session_state.session_id or "default_session",
                            "message_id": msg_id,
                            "trace_id": trace_id,
                            "rating": rating_score,
                            "comment": "Streamlit thumbs feedback synced with LangSmith"
                        }
                    )
                    msg["feedback_submitted"] = True
                    st.caption("Feedback saved to database & synced to LangSmith.")

    # Chat Input Handling
    user_input = st.chat_input(f"Ask what you want to prepare in {active_subject_code}... (e.g., Explain QuickSort algorithm and time complexity)")
    prompt_to_send = st.session_state.pending_prompt or user_input

    if prompt_to_send:
        # Clear pending prompt
        st.session_state.pending_prompt = None

        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner(f"Searching 123 RTU past papers (2012-2024) for {active_subject_code} and preparing your answer..."):
                payload = {
                    "message": prompt_to_send,
                    "session_id": st.session_state.session_id,
                    "filters": {
                        "subject": active_subject_code,
                        "branch": selected_branch,
                        "year": selected_year
                    }
                }
                _t0 = time.perf_counter()
                res = api_post(
                    "/api/chat",
                    st.session_state.backend_mode,
                    st.session_state.backend_url,
                    payload,
                    timeout=60
                )
                _t1 = time.perf_counter()

                if "error" in res:
                    err_text = res["error"]
                    if "Connection" in err_text or "connection" in err_text:
                        error_msg = "Backend is starting up. Please wait 10 seconds and try again."
                    else:
                        error_msg = f"Connection error: {err_text}. Please make sure your backend engine is running."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    assistant_reply = res.get("response", "")
                    if not assistant_reply:
                        assistant_reply = "No matching RTU past examination questions found for this topic. Try rephrasing with the subject name (e.g., 'DSA quicksort' or 'DBMS normalization')."
                    citations = res.get("citations", [])
                    st.session_state.session_id = res.get("session_id", st.session_state.session_id)

                    st.markdown(assistant_reply)

                    # Improvement 2: Latency indicator
                    latency_ms = res.get("latency_ms") or int((_t1 - _t0) * 1000)
                    st.caption(f"Response time: {latency_ms} ms")

                    if citations:
                        # Citation count badge with real source paper count and year range
                        citation_years = [c.get("year") for c in citations if c.get("year") and str(c.get("year")).isdigit()]
                        pids = set(c.get("source_pdf") or c.get("paper_id") for c in citations if (c.get("source_pdf") or c.get("paper_id")))
                        src_cnt = len(pids)
                        paper_str = f"{len(citations)} questions from {src_cnt} {'paper' if src_cnt == 1 else 'papers'}"
                        if citation_years:
                            min_yr = min(int(y) for y in citation_years)
                            max_yr = max(int(y) for y in citation_years)
                            year_range = str(min_yr) if min_yr == max_yr else f"{min_yr}-{max_yr}"
                            expander_label = f"RTU Past Paper References ({paper_str}, {year_range})"
                        else:
                            expander_label = f"RTU Past Paper References ({paper_str})"
                        with st.expander(expander_label):
                            for idx, c in enumerate(citations):
                                st.markdown(
                                    f"**{idx + 1}. {c.get('paper_name', 'RTU Past Paper')}** (Year: {c.get('year', '-')}) | "
                                    f"*Question ID:* `{c.get('question_id', '-')}` | "
                                    f"*Marks:* **{c.get('marks', '-')}**"
                                )
                                if c.get("question_text"):
                                    st.caption(f"\"{c.get('question_text')}\"")

                    assistant_msg_id = res.get("assistant_message", {}).get("message_id") or f"msg_{len(st.session_state.messages)}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_reply,
                        "citations": citations,
                        "message_id": assistant_msg_id,
                        "trace_id": res.get("trace_id"),
                        "screening": res.get("screening", {})
                    })

# ------------------------------------------------------------------------------
# PAGE 2: Crash Course Planner
# ------------------------------------------------------------------------------
elif st.session_state.selected_page == "Crash Course Planner":
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        hours_input = st.slider(f"Hours left until your {active_subject_code} exam", min_value=1.0, max_value=12.0, value=6.0, step=0.5)
    with col_p2:
        generate_btn = st.button("Generate Roadmap", use_container_width=True, type="primary")

    st.subheader(f"Urgent Exam Revision Planner: {active_subject_code} (Year {selected_year})")
    st.caption("Generate a customized, hour-by-hour revision plan based on real topic mark weightages from verified RTU past exam papers.")

    if generate_btn:
        with st.spinner(f"Analyzing RTU paper frequency for {active_subject_code} across {hours_input} hours..."):
            plan_res = api_post(
                "/api/plan",
                st.session_state.backend_mode,
                st.session_state.backend_url,
                {"subject_id": active_subject_code, "hours_left": hours_input},
                timeout=20
            )

            if "error" in plan_res:
                st.error(f"Failed to generate plan: {plan_res['error']}")
            else:
                schedule = plan_res.get("schedule", [])
                st.success(f"Generated {len(schedule)}-Slot Revision Plan for **{active_subject_code}** ({hours_input:.1f} Hours Available)")

                for item in schedule:
                    with st.container():
                        st.markdown(f"#### Slot {item.get('slot_number', '-')}: {item.get('time_window', '')}")
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.write(f"**Topic:** {item.get('topic_name', '-')}")
                            st.caption(f"Strategy: {item.get('action', item.get('strategy', 'Revise key definitions & diagrams'))}")
                        with col_b:
                            st.info(f"**Priority:** {item.get('marks_priority', 'High Yield')}")
                        st.divider()

                st.markdown("### RTU Examiner Scoring Strategy")
                st.markdown(
                    "- **Part C (10/15 Marks):** Always draw a neat, labeled block diagram or state-transition table.\n"
                    "- **Time Complexity:** Include explicit recurrence relations and Big-O notation.\n"
                    "- **Highlight:** Put boxes around final mathematical derivations or numerical values."
                )

# ------------------------------------------------------------------------------
# PAGE 3: High-Yield Important Topics
# ------------------------------------------------------------------------------
elif st.session_state.selected_page == "High-Yield Important Topics":
    st.subheader(f"High-Yield Important Topics: {active_subject_code} (Year {selected_year})")
    st.caption("Topics ranked by marks weightage and frequency across verified RTU past examination papers.")

    with st.spinner(f"Fetching topic frequency data for {active_subject_code}..."):
        topics_data = api_get(
            f"/api/topics/SUBJ_{active_subject_code}/important",
            st.session_state.backend_mode,
            st.session_state.backend_url,
            timeout=10
        )
        if "error" in topics_data:
            topics_data = api_get(
                f"/api/topics/{active_subject_code}/important",
                st.session_state.backend_mode,
                st.session_state.backend_url,
                timeout=10
            )

        ranked = topics_data.get("ranked_topics", [])
        if ranked:
            df = pd.DataFrame(ranked)
            display_cols = ["rank", "topic_name", "unit_no", "total_marks", "paper_appearances", "importance_level"]
            present_cols = [c for c in display_cols if c in df.columns]
            renamed = {
                "rank": "Rank",
                "topic_name": "Topic Name",
                "unit_no": "Unit",
                "total_marks": "Total Marks in PYQs",
                "paper_appearances": "Papers Appeared",
                "importance_level": "Priority Level"
            }
            st.dataframe(df[present_cols].rename(columns=renamed), use_container_width=True)
        else:
            st.info(f"No explicit ranking records found for {active_subject_code}.")

# ------------------------------------------------------------------------------
# PAGE 4: Practice Quiz & Examiner Evaluation
# ------------------------------------------------------------------------------
elif st.session_state.selected_page == "Practice Quiz & Evaluation":
    st.subheader(f"RTU Practice Exam & Instant Evaluation: {active_subject_code} (Year {selected_year})")
    st.caption("Test your knowledge on real RTU past examination questions and get evaluated on RTU's official 0-10 rubric.")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        st.write(f"Click below to draw a verified RTU examination question for **{active_subject_code}**:")
    with col_q2:
        fetch_question_btn = st.button("Fetch New RTU Question", type="primary", use_container_width=True)

    if fetch_question_btn:
        with st.spinner(f"Selecting question from RTU {active_subject_code} question papers..."):
            quiz_q = api_post(
                "/api/quiz/next",
                st.session_state.backend_mode,
                st.session_state.backend_url,
                {"subject_id": active_subject_code},
                timeout=15
            )
            if "error" in quiz_q:
                st.error(f"Error fetching quiz question: {quiz_q['error']}")
            else:
                st.session_state.current_quiz = quiz_q
                st.session_state.quiz_feedback = None

    if st.session_state.current_quiz:
        q = st.session_state.current_quiz
        st.markdown(f"### Question (Max Marks: {q.get('max_marks', 10)})")
        st.info(f"**\"{q.get('question_text', '')}\"**\n\n*(Unit {q.get('unit_no', '-')}, RTU Past Exam)*")

        student_answer = st.text_area(
            "Write your answer here (definitions, key algorithm steps, diagrams in text, complexities):",
            height=200,
            placeholder="Type your exam-standard answer here..."
        )

        if st.button("Submit for RTU Examiner Grading", type="primary"):
            if not student_answer.strip():
                st.warning("Please type your answer before submitting.")
            else:
                with st.spinner("Examiner is grading your answer against RTU rubric..."):
                    eval_res = api_post(
                        "/api/quiz/answer",
                        st.session_state.backend_mode,
                        st.session_state.backend_url,
                        {
                            "question_id": q.get("question_id"),
                            "student_answer": student_answer
                        },
                        timeout=30
                    )
                    if "error" in eval_res:
                        st.error(f"Evaluation failed: {eval_res['error']}")
                    else:
                        st.session_state.quiz_feedback = eval_res

    if st.session_state.quiz_feedback:
        fb = st.session_state.quiz_feedback
        st.markdown("---")
        st.subheader("RTU Examiner Scorecard")
        score = fb.get("score", 0)
        max_marks = fb.get("max_score", 10)

        col_s1, col_s2 = st.columns([1, 3])
        with col_s1:
            st.metric("Total Score", f"{score:.1f} / {max_marks}")
        with col_s2:
            st.write(f"**Examiner Verdict:** {fb.get('feedback', '')}")

        breakdown = fb.get("breakdown", {})
        if breakdown:
            st.write("**Rubric Breakdown:**")
            st.json(breakdown)

# ------------------------------------------------------------------------------
# PAGE 5: Past Papers Archive
# ------------------------------------------------------------------------------
elif st.session_state.selected_page == "Past Papers Archive":
    st.subheader("Verified RTU Past Papers Archive (2012-2024)")
    st.caption("Browse and review official Rajasthan Technical University semester examination papers.")

    with st.spinner("Loading papers catalog..."):
        papers = api_get("/api/papers", st.session_state.backend_mode, st.session_state.backend_url, timeout=10)
        if isinstance(papers, list) and len(papers) > 0:
            df_papers = pd.DataFrame(papers)

            # Filter options
            years = sorted(list(set(df_papers["year"].dropna().astype(int).tolist())), reverse=True)
            filter_year = st.multiselect("Filter by Examination Year", years, default=years[:5])

            filtered_df = df_papers
            if filter_year:
                filtered_df = filtered_df[filtered_df["year"].isin(filter_year)]

            st.write(f"Showing **{len(filtered_df)}** verified RTU question papers:")

            display_cols = ["paper_id", "subject", "subject_code", "year", "session", "question_count"]
            renamed = {
                "paper_id": "Paper ID",
                "subject": "Subject Name",
                "subject_code": "Code",
                "year": "Year",
                "session": "Session",
                "question_count": "Questions"
            }
            st.dataframe(filtered_df[display_cols].rename(columns=renamed), use_container_width=True)
        else:
            st.info("No papers returned from backend.")
