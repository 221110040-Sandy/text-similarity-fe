import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from io import StringIO
import requests
import json
import time
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

load_dotenv()

# Initialize BERT tokenizer for token counting
@st.cache_resource
def get_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

sys.path.append(str(Path(__file__).parent.parent))
from utils.auth import initialize_auth_state, require_auth, logout

def read_csv_strip_quotes(uploaded_file, delimiter=None):
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8-sig")
    except Exception:
        text = raw.decode("latin1", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    fixed = []
    for ln in lines:
        s = ln
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        s = s.replace('""', '"')
        fixed.append(s)
    fixed_text = "\n".join(fixed)
    if delimiter:
        return pd.read_csv(StringIO(fixed_text), sep=delimiter)
    else:
        return pd.read_csv(StringIO(fixed_text), sep=None, engine='python')

def safe_read_csv(uploaded_file, required_cols):
    if uploaded_file is None:
        return None, False, None
    df = None
    for delimiter in [None, ',', ';']:
        try:
            uploaded_file.seek(0)
            if delimiter:
                df = pd.read_csv(uploaded_file, sep=delimiter)
            else:
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
            if df.shape[1] > 1:
                break
        except Exception:
            continue
    
    if df is None or df.shape[1] == 1:
        try:
            uploaded_file.seek(0)
            df = read_csv_strip_quotes(uploaded_file)
        except Exception:
            return None, False, "Cannot read CSV (format/encoding not recognized)."
    
    cols = list(df.columns)
    missing = [c for c in required_cols if c not in cols]
    if missing:
        return df, False, f"Required columns missing → {', '.join(missing)}"
    return df, True, None

st.set_page_config(page_title="Admin Panel", layout="wide", initial_sidebar_state="collapsed")

initialize_auth_state()

st.markdown("""
<style>
  .admin-header { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:2rem; border-radius:15px; color:#fff; text-align:center; margin-bottom:2rem; box-shadow:0 10px 30px rgba(102,126,234,0.3);} 
  #MainMenu, header, footer { visibility:hidden; }
  [data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none; }
  .summary-card { padding: 12px 16px; border-radius:8px; background:rgba(255,255,255,0.02); }
</style>
""", unsafe_allow_html=True)

require_auth()

if "job_started" not in st.session_state:
    st.session_state.job_started = False
if "job_running" not in st.session_state:
    st.session_state.job_running = False
if "last_status" not in st.session_state:
    st.session_state.last_status = None

st.markdown("<div class='admin-header'><h1>Admin Dashboard</h1></div>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Back to Home", width='stretch'):
        st.switch_page("app_frontend.py")
with col2:
    if st.button("Logout", width='stretch'):
        logout()
        st.success("Logged out successfully!")
        st.rerun()

st.markdown("---")

# config
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

st.markdown("### Upload CSV (Train / Val / Test)")
col_map, _ = st.columns([2,1])
with col_map:
    st.markdown("**Column mapping (adjust to your CSV headers).**")
    st.warning("Make sure all three datasets (Train, Validation, Test) have the exact same headers.")
    col_sent1 = st.text_input("Header name for sentence1", value="sentence1")
    col_sent2 = st.text_input("Header name for sentence2", value="sentence2")
    col_label = st.text_input("Header name for label", value="label")

required_cols = [col_sent1, col_sent2, col_label]

st.info("Train & Validation required. Test optional (if empty, will be split from Train). Max 25MB per file.")

col_train, col_val, col_test = st.columns(3)

train_file = val_file = test_file = None

df_train = df_val = df_test = None

def handle_upload(uploader, key, required_cols):
    f = uploader
    if not f:
        return None, False, "not_uploaded"
    if hasattr(f, 'size') and f.size is not None:
        if f.size > MAX_UPLOAD_BYTES:
            return None, False, f"file_too_large: {f.size} bytes"
    df, valid, err = safe_read_csv(f, required_cols)
    return df, valid, err

with col_train:
    st.subheader("Train CSV (required)")
    train_file = st.file_uploader("Upload train CSV", type=['csv'], key='train_file')
    train_valid = False
    train_err = None
    if train_file:
        if train_file.size > MAX_UPLOAD_BYTES:
            st.error("File too large — maximum 25 MB")
            train_file = None
            train_valid = False
        else:
            df_train, train_valid, train_err = safe_read_csv(train_file, required_cols)
            if train_err:
                st.error(train_err)
            if df_train is not None and train_valid:
                st.metric("Rows", len(df_train))
                st.metric("Columns", len(df_train.columns))
                st.success("Document valid — all columns found")
            elif df_train is not None and not train_valid:
                st.error("Required columns missing or headers mismatch: " + ", ".join([c for c in required_cols if c not in list(df_train.columns)]))
    else:
        st.error("Train CSV is required")

with col_val:
    st.subheader("Validation CSV (required)")
    val_file = st.file_uploader("Upload validation CSV", type=['csv'], key='val_file')
    val_valid = False
    val_err = None
    if val_file:
        if val_file.size > MAX_UPLOAD_BYTES:
            st.error("File too large — maximum 25 MB")
            val_file = None
            val_valid = False
        else:
            df_val, val_valid, val_err = safe_read_csv(val_file, required_cols)
            if val_err:
                st.error(val_err)
            if df_val is not None and val_valid:
                st.metric("Rows", len(df_val))
                st.metric("Columns", len(df_val.columns))
                st.success("Document valid — all columns found")
            elif df_val is not None and not val_valid:
                st.error("Required columns missing or headers mismatch: " + ", ".join([c for c in required_cols if c not in list(df_val.columns)]))
    else:
        st.error("Validation CSV is required")

with col_test:
    st.subheader("Test CSV (optional)")
    test_file = st.file_uploader("Upload test CSV", type=['csv'], key='test_file')
    test_valid = False
    test_err = None
    if test_file:
        if test_file.size > MAX_UPLOAD_BYTES:
            st.error("File too large — maximum 25 MB")
            test_file = None
            test_valid = False
        else:
            df_test, test_valid, test_err = safe_read_csv(test_file, required_cols)
            if test_err:
                st.error(test_err)
            if df_test is not None and test_valid:
                st.metric("Rows", len(df_test))
                st.metric("Columns", len(df_test.columns))
                st.success("Document valid — all columns found")
            elif df_test is not None and not test_valid:
                st.error("Required columns missing or headers mismatch: " + ", ".join([c for c in required_cols if c not in list(df_test.columns)]))
    else:
        st.info("Test CSV empty — will be split from Train when process starts with 80:20 ratio")

st.markdown("---")

# summaries
train_summary = None
val_summary = None
test_summary = None
if df_train is not None and train_valid:
    def summarize_df(df):
        total = len(df)
        s1_nonnull = int(df[col_sent1].notna().sum())
        s2_nonnull = int(df[col_sent2].notna().sum())
        label_nonnull = int(df[col_label].notna().sum()) if col_label in df.columns else 0
        vc = df[col_label].value_counts(dropna=False) if col_label in df.columns else pd.Series(dtype=int)
        label_0 = int(vc.get(0, 0))
        label_1 = int(vc.get(1, 0))
        pct_0 = round(label_0 / total * 100, 2) if total > 0 else 0.0
        pct_1 = round(label_1 / total * 100, 2) if total > 0 else 0.0
        # Calculate token count using BERT tokenizer
        tokenizer = get_tokenizer()
        combined = (df[col_sent1].fillna("").astype(str) + " " + df[col_sent2].fillna("").astype(str))
        token_counts = combined.apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))
        qs = token_counts.quantile([0.25, 0.5, 0.75, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]).to_dict()
        qtable = {25: int(qs.get(0.25, 0)), 50: int(qs.get(0.5, 0)), 75: int(qs.get(0.75, 0)), 95: int(qs.get(0.95, 0)), 96: int(qs.get(0.96, 0)), 97: int(qs.get(0.97, 0)), 98: int(qs.get(0.98, 0)), 99: int(qs.get(0.99, 0)), 100: int(qs.get(1.0, 0))}
        max_len = qtable[95]
        return {"total": total, "s1_nonnull": s1_nonnull, "s2_nonnull": s2_nonnull, "label_nonnull": label_nonnull, "label_0": label_0, "label_1": label_1, "pct_0": pct_0, "pct_1": pct_1, "qtable": qtable, "max_len": max_len}
    train_summary = summarize_df(df_train)
if df_val is not None and val_valid:
    val_summary = summarize_df(df_val)
if df_test is not None and test_valid:
    test_summary = summarize_df(df_test)

st.markdown("### Dataset summary")
c1, c2, c3 = st.columns(3)
for col, name, summary, df in zip((c1, c2, c3), ("Train", "Validation", "Test"), (train_summary, val_summary, test_summary), (df_train, df_val, df_test)):
    with col:
        st.markdown(f"#### {name}")
        if summary is None:
            st.write("N/A")
            continue
        st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
        st.metric("Rows", summary["total"]) 
        st.write(f"{col_sent1} non-null: {summary['s1_nonnull']}")
        st.write(f"{col_sent2} non-null: {summary['s2_nonnull']}")
        st.write(f"{col_label} non-null: {summary['label_nonnull']}")
        st.write(f"Label distribution: 0 = {summary['label_0']} ({summary['pct_0']}%), 1 = {summary['label_1']} ({summary['pct_1']}%)")
        qt = summary["qtable"]
        qdf = pd.DataFrame({"percentile": ["P25","P50","P75","P95","P96","P97","P98","P99","P100"], "tokens": [qt[25], qt[50], qt[75], qt[95], qt[96], qt[97], qt[98], qt[99], qt[100]]})
        st.table(qdf)
        st.write(f"Max tokens (use P95): {summary['max_len']}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("Sample rows:")
        st.dataframe(df[[col_sent1, col_sent2, col_label]].head(5), width='stretch')

st.markdown("---")

st.markdown("### Training Options")

col_batch, col_maxlen = st.columns(2)

with col_batch:
    batch_size = st.radio("Batch size", options=[16,32,64,128,256,512], index=1)

with col_maxlen:
    # Build maxlen options from train percentiles
    if train_summary is not None:
        qt = train_summary["qtable"]
        maxlen_options = {
            f"P25 ({qt[25]} tokens)": qt[25],
            f"P50 ({qt[50]} tokens)": qt[50],
            f"P75 ({qt[75]} tokens)": qt[75],
            f"P95 ({qt[95]} tokens)": qt[95],
            f"P96 ({qt[96]} tokens)": qt[96],
            f"P97 ({qt[97]} tokens)": qt[97],
            f"P98 ({qt[98]} tokens)": qt[98],
            f"P99 ({qt[99]} tokens)": qt[99],
            f"P100 ({qt[100]} tokens)": qt[100],
        }
        selected_maxlen = st.radio("Max sequence length", options=list(maxlen_options.keys()), index=3)  # Default to P95
        maxlen_value = maxlen_options[selected_maxlen]
    else:
        st.warning("Upload Train CSV to see maxlen options")
        maxlen_value = 128  # Default fallback

st.markdown("### Hyperband Configuration")

col_bilstm, col_attn, col_dense, col_lr, col_drop, col_wd = st.columns(6)

with col_bilstm:
    bilstm_count = st.number_input("Number of BiLSTM values", min_value=1, max_value=20, value=3, step=1, key="bilstm_count")
    bilstm_values = []
    for i in range(bilstm_count):
        v = st.number_input(f"BiLSTM value {i+1}", min_value=1, max_value=16384, value=64*(i+1), step=1, key=f"bilstm_v_{i}")
        bilstm_values.append(int(v))

with col_attn:
    attn_count = st.number_input("Number of Attention values", min_value=1, max_value=20, value=3, step=1, key="attn_count")
    attention_values = []
    for i in range(attn_count):
        v = st.number_input(f"Attention value {i+1}", min_value=1, max_value=16384, value=128*(i+1), step=1, key=f"attn_v_{i}")
        attention_values.append(int(v))

with col_dense:
    dense_count = st.number_input("Number of Dense values", min_value=1, max_value=20, value=3, step=1, key="dense_count")
    dense_values = []
    for i in range(dense_count):
        v = st.number_input(f"Dense value {i+1}", min_value=1, max_value=16384, value=16*(i+1), step=1, key=f"dense_v_{i}")
        dense_values.append(int(v))

with col_lr:
    lr_count = st.number_input("Number of learning rate values", min_value=1, max_value=20, value=3, step=1, key="lr_count")
    lr_values = []
    for i in range(lr_count):
        v = st.number_input(f"LR value {i+1}", value=1e-4 * (10**i) if i>0 else 1e-4, format="%.8g", key=f"lr_v_{i}")
        lr_values.append(float(v))

with col_drop:
    drop_count = st.number_input("Number of dropout values", min_value=1, max_value=20, value=3, step=1, key="drop_count")
    drop_values = []
    for i in range(drop_count):
        v = st.number_input(f"Dropout value {i+1}", min_value=0.0, max_value=1.0, value=0.1 + 0.1*i, step=0.01, format="%.2f", key=f"drop_v_{i}")
        drop_values.append(float(v))

with col_wd:
    wd_count = st.number_input("Number of weight decay values", min_value=1, max_value=20, value=2, step=1, key="wd_count")
    wd_values = []
    for i in range(wd_count):
        v = st.number_input(f"WD value {i+1}", value=1e-4 * (i+1), format="%.8g", key=f"wd_v_{i}")
        wd_values.append(float(v))

st.markdown("---")

hb_col3, hb_col4, hb_col5 = st.columns(3)
with hb_col3:
    max_epochs = st.number_input("Max epochs (Hyperband)", min_value=1, value=5, step=1)
with hb_col4:
    max_trials = st.number_input("Max trials (Hyperband)", min_value=1, value=5, step=1)
with hb_col5:
    pruner_factor = st.number_input("Pruner factor", min_value=2, value=2, step=1)

st.markdown("---")

st.markdown("### Sampling for Hyperband")
samp_train_pct = st.slider("Sampling percentage from train (%)", min_value=1, max_value=100, value=20)
samp_val_pct = st.slider("Sampling percentage from validation (%)", min_value=1, max_value=100, value=20)
samp_test_pct = st.slider("Sampling percentage from test (%)", min_value=1, max_value=100, value=20)

st.markdown("---")

st.markdown("### After Hyperband")
full_training = st.radio("Perform full training after Hyperband?", options=["Yes","No"], index=1)

full_training_epochs = 10 
if full_training == "Yes":
    full_training_epochs = st.number_input("Full training epochs", min_value=1, value=10, step=1)

any_uploaded = all([train_file is not None, val_file is not None])
all_uploaded_valid = all([train_valid, val_valid]) and (not test_file or test_valid)

disabled = False
reasons = []
if not any_uploaded:
    disabled = True
    reasons.append("Train and Validation are required")
if not all_uploaded_valid:
    disabled = True
    reasons.append("Train/Validation/Test not valid (check message in each column)")

if disabled:
    st.info("Cannot start process: " + "; ".join(reasons))
URL = os.getenv("API_BASE_URL", "https://desertlike-nonrecognized-keagan.ngrok-free.dev")
API_URL = URL + "/find-hyperparam"
API_TIMEOUT = 60

# Check if job is actually running (not in terminal states)
actual_running = st.session_state.job_running and st.session_state.last_status is not None and st.session_state.last_status.get("status") not in ["COMPLETED", "FAILED", "ERROR"]

if st.button(
    "Start Training",
    type="primary",
    disabled=disabled or actual_running
):
    st.session_state.job_running = True
    with st.spinner("Sending configuration and files to backend..."):

        files = {
            "dataset_train": (train_file.name, train_file.getvalue(), "text/csv"),
            "dataset_dev": (val_file.name, val_file.getvalue(), "text/csv"),
        }
        if test_file:
            files["dataset_test"] = (test_file.name, test_file.getvalue(), "text/csv")

        data = {
            "header_col1": col_sent1,
            "header_col2": col_sent2,
            "header_label": col_label,
            "batch_size": int(batch_size),
            "maxlen": int(maxlen_value),
            "bilstm_units": ",".join(map(str, bilstm_values)),
            "attention_units": ",".join(map(str, attention_values)),
            "dense_units": ",".join(map(str, dense_values)),
            "learning_rate": ",".join(map(str, lr_values)),
            "dropout_rate": ",".join(map(str, drop_values)),
            "weight_decay": ",".join(map(str, wd_values)),
            "max_epochs": int(max_epochs),
            "max_trials": int(max_trials),
            "pruner_factor": int(pruner_factor),
            "sampling_train": int(samp_train_pct),
            "sampling_dev": int(samp_val_pct),
            "sampling_test": int(samp_test_pct),
            "train_full": full_training == "Yes",
            "full_training_epochs": int(full_training_epochs),
        }

        try:
            r = requests.post(API_URL, data=data, files=files, timeout=60)
            r.raise_for_status()

            st.session_state.job_started = True
            st.session_state.last_status = None

            st.success("Job successfully sent to backend")
            st.rerun()

        except requests.RequestException as e:
            st.error(f"Error calling API: {e}")

STATUS_URL = URL + "/status"

if st.session_state.job_started:
    try:
        resp = requests.get(STATUS_URL, timeout=10)
        resp.raise_for_status()
        s = resp.json()
        st.session_state.last_status = s
    except Exception as e:
        st.error(f"Failed to get status: {e}")
        s = st.session_state.last_status

    if s:
        status_text = s.get("status", "UNKNOWN")
        is_running = status_text in ["TUNING", "TRAINING"]

        st.subheader("Training Status")
        st.write(f"**Status:** {status_text}")
        st.write(f"**Running:** {is_running}")

        prog = s.get("progress", {})
        cur = prog.get("current_trial", 0)
        total = prog.get("total_trials", 0)
        best = prog.get("best_loss")

        if total > 0:
            st.progress(min(cur / total, 1.0))
            st.write(f"Trial {cur} / {total}")
        else:
            st.progress(0.0)
            st.write(f"Trial {cur} (total trials not yet determined)")

        if best is not None:
            st.write(f"Best loss: `{best:.4f}`")

        logs_list = s.get("logs", [])
        is_retraining = any("Retraining Final Model" in log for log in logs_list)
        if is_retraining and status_text not in ["COMPLETED", "FAILED", "FULL TRAIN", "ERROR"]:
            st.markdown("---")
            st.markdown("""
            <style>
            .spinner-container {
                display: flex;
                align-items: center;
                gap: 15px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                margin: 10px 0;
            }
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid rgba(102, 126, 234, 0.3);
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .spinner-text {
                font-size: 18px;
                font-weight: 600;
                color: #667eea;
            }
            </style>
            <div class="spinner-container">
                <div class="spinner"></div>
                <span class="spinner-text">Training Final Model...</span>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Logs")
        for log in s.get("logs", [])[::-1]:
            st.code(log)

        # Terminal states: stop refreshing and reset job state
        if status_text in ["COMPLETED", "FAILED", "ERROR"]:
            st.session_state.job_running = False
            st.session_state.job_started = False

            if status_text == "COMPLETED":
                st.success("Training completed 🎉")
            elif status_text == "ERROR":
                st.error("Training encountered an error ❌")
            else:
                st.error("Training failed")
        else:
            # Still running, keep refreshing
            time.sleep(5)
            st.rerun()