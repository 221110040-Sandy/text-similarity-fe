import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from io import StringIO
import requests
import json
import time

sys.path.append(str(Path(__file__).parent.parent))
from utils.auth import initialize_auth_state, require_auth, logout

def read_csv_strip_quotes(uploaded_file):
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
    return pd.read_csv(StringIO(fixed_text))

st.set_page_config(page_title="Admin Panel", layout="wide", initial_sidebar_state="collapsed")

initialize_auth_state()

st.markdown("""
<style>
  .admin-header { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:2rem; border-radius:15px; color:#fff; text-align:center; margin-bottom:2rem; box-shadow:0 10px 30px rgba(102,126,234,0.3);} 
  #MainMenu, header, footer { visibility:hidden; }
  [data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none; }
</style>
""", unsafe_allow_html=True)

require_auth()

st.markdown("<div class='admin-header'><h1>Admin Dashboard</h1></div>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Kembali ke Home", use_container_width=True):
        st.switch_page("app_frontend.py")
with col2:
    if st.button("Logout", use_container_width=True):
        logout()
        st.success("Berhasil logout!")
        st.rerun()

st.markdown("---")

st.markdown("### Upload CSV (Train / Val / Test)")
col_train, col_val, col_test = st.columns(3)

train_file = val_file = test_file = None
df_train = df_val = df_test = None

col_sent1 = "sentence1"
col_sent2 = "sentence2"
col_label = "label"

st.info("CSV harus memiliki header: 'sentence1', 'sentence2', 'label'")

with col_train:
    st.subheader("Train CSV (required)")
    train_file = st.file_uploader("Upload train CSV", type=['csv'], key='train_file')
    train_valid = False
    if train_file:
        try:
            df_train = pd.read_csv(train_file)
            if df_train.shape[1] == 1:
                raise ValueError("single column, fallback")
        except Exception:
            try:
                df_train = read_csv_strip_quotes(train_file)
            except Exception as e:
                st.error(f"Error membaca Train CSV: {e}")
                df_train = None
        if df_train is not None:
            st.dataframe(df_train.head(5), use_container_width=True)
            st.metric("Rows", len(df_train))
            st.metric("Columns", len(df_train.columns))
            cols = list(df_train.columns)
            missing = [c for c in [col_sent1, col_sent2, col_label] if c not in cols]
            if missing:
                st.error(f"Kolom wajib hilang → {', '.join(missing)}")
            else:
                st.success("Dokumen valid — semua kolom ditemukan")
                train_valid = True
    else:
        st.warning("Train CSV wajib diupload")

with col_val:
    st.subheader("Validation CSV (optional)")
    val_file = st.file_uploader("Upload validation CSV", type=['csv'], key='val_file')
    val_valid = False
    if val_file:
        try:
            df_val = pd.read_csv(val_file)
            if df_val.shape[1] == 1:
                raise ValueError("single column, fallback")
        except Exception:
            try:
                df_val = read_csv_strip_quotes(val_file)
            except Exception as e:
                st.error(f"Error membaca Validation CSV: {e}")
                df_val = None
        if df_val is not None:
            st.dataframe(df_val.head(5), use_container_width=True)
            st.metric("Rows", len(df_val))
            st.metric("Columns", len(df_val.columns))
            cols = list(df_val.columns)
            missing = [c for c in [col_sent1, col_sent2, col_label] if c not in cols]
            if missing:
                st.error(f"Kolom wajib hilang → {', '.join(missing)}")
            else:
                st.success("Dokumen valid — semua kolom ditemukan")
                val_valid = True

with col_test:
    st.subheader("Test CSV (required)")
    test_file = st.file_uploader("Upload test CSV", type=['csv'], key='test_file')
    test_valid = False
    if test_file:
        try:
            df_test = pd.read_csv(test_file)
            if df_test.shape[1] == 1:
                raise ValueError("single column, fallback")
        except Exception:
            try:
                df_test = read_csv_strip_quotes(test_file)
            except Exception as e:
                st.error(f"Error membaca Test CSV: {e}")
                df_test = None
        if df_test is not None:
            st.dataframe(df_test.head(5), use_container_width=True)
            st.metric("Rows", len(df_test))
            st.metric("Columns", len(df_test.columns))
            cols = list(df_test.columns)
            missing = [c for c in [col_sent1, col_sent2, col_label] if c not in cols]
            if missing:
                st.error(f"Kolom wajib hilang → {', '.join(missing)}")
            else:
                st.success("Dokumen valid — semua kolom ditemukan")
                test_valid = True
    else:
        st.warning("Test CSV wajib diupload")

st.markdown("---")

st.markdown("### Training Options")
batch_size = st.radio("Batch size", options=[16,32,64,128], index=1)

st.markdown("### Hyperband configuration")
left, right = st.columns(2)

with left:
    bilstm_count = st.number_input("Jumlah nilai BiLSTM", min_value=1, max_value=20, value=3, step=1, key="bilstm_count")
    bilstm_values = []
    n = int(bilstm_count)
    rows = (n + 1) // 2
    for r in range(rows):
        c1, c2 = st.columns(2)
        idx1 = r*2
        idx2 = r*2 + 1
        v1 = c1.number_input(f"BiLSTM value {idx1+1}", min_value=1, max_value=16384, value=64*(idx1+1), step=1, key=f"bilstm_v_{idx1}")
        bilstm_values.append(int(v1))
        if idx2 < n:
            v2 = c2.number_input(f"BiLSTM value {idx2+1}", min_value=1, max_value=16384, value=64*(idx2+1), step=1, key=f"bilstm_v_{idx2}")
            bilstm_values.append(int(v2))

with right:
    st.markdown("Attention units (BiLSTM * 2)")
    attention_values = [v * 2 for v in bilstm_values]
    rows_att = (len(attention_values) + 1) // 2
    for r in range(rows_att):
        ca, cb = st.columns(2)
        i1 = r*2
        i2 = i1 + 1
        ca.markdown(f"**Attention value {i1+1}**")
        ca.write(attention_values[i1])
        if i2 < len(attention_values):
            cb.markdown(f"**Attention value {i2+1}**")
            cb.write(attention_values[i2])

st.markdown("---")

st.markdown("### Optimizer / regularization")
opt_col1, opt_col2 = st.columns(2)
with opt_col1:
    lr_min = st.number_input("Learning rate (min)", value=1e-5, format="%.6g", key="lr_min")
    lr_max = st.number_input("Learning rate (max)", value=1e-3, format="%.6g", key="lr_max")
with opt_col2:
    drop_min = st.number_input("Dropout (min, 0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="drop_min")
    drop_max = st.number_input("Dropout (max, 0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="drop_max")

wd_col1, wd_col2 = st.columns(2)
with wd_col1:
    wd_min = st.number_input("Weight decay (min)", min_value=0.0, value=0.0, step=1e-8, format="%.8g", key="wd_min")
with wd_col2:
    wd_max = st.number_input("Weight decay (max)", min_value=0.0, value=1e-4, step=1e-8, format="%.8g", key="wd_max")

hb_col3, hb_col4 = st.columns(2)
with hb_col3:
    max_epochs = st.number_input("Max epochs (Hyperband)", min_value=1, value=50, step=1)
with hb_col4:
    max_trials = st.number_input("Max trials (Hyperband)", min_value=1, value=50, step=1)

st.markdown("---")

st.markdown("### Sampling untuk Hyperband")
samp_train_pct = st.slider("Persentase sampling dari train (%)", min_value=1, max_value=100, value=20)
samp_val_pct = st.slider("Persentase sampling dari validation (%)", min_value=1, max_value=100, value=20)
samp_test_pct = st.slider("Persentase sampling dari test (%)", min_value=1, max_value=100, value=20)

st.markdown("---")

st.markdown("### Setelah Hyperband")
full_training = st.radio("Lakukan full training setelah Hyperband?", options=["Ya","Tidak"], index=0)

any_uploaded = any([train_file, test_file])
all_uploaded_valid = True
if not train_file or not train_valid:
    all_uploaded_valid = False
if not test_file or not test_valid:
    all_uploaded_valid = False

disabled = False
reasons = []
if not any_uploaded:
    disabled = True
    reasons.append("Belum ada CSV train/test yang diupload")
if not all_uploaded_valid:
    disabled = True
    reasons.append("Train dan Test wajib valid (cek pesan di tiap kolom)")

if disabled:
    st.info("Tidak dapat memulai proses: " + "; ".join(reasons))

def prepare_splits(df_train, df_val, df_test, label_col, seed=42):
    from sklearn.model_selection import train_test_split
    train_df = df_train.copy() if df_train is not None else None
    val_df = df_val.copy() if df_val is not None else None
    test_df = df_test.copy() if df_test is not None else None
    if val_df is None and test_df is not None:
        if label_col in test_df.columns:
            stratify_col = test_df[label_col]
            dev, test = train_test_split(test_df, test_size=0.2, random_state=seed, stratify=stratify_col)
        else:
            dev, test = train_test_split(test_df, test_size=0.2, random_state=seed)
        val_df = dev.reset_index(drop=True)
        test_df = test.reset_index(drop=True)
    return train_df, val_df, test_df

def sample_for_hyperband(df, pct, seed=42):
    if df is None or pct <= 0:
        return None
    n = max(1, int(len(df) * pct / 100))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

API_URL = "https://localhost:8000/find-hyperparameter"
API_TIMEOUT = 60

if st.button("Mulai Training", type="primary", disabled=disabled):
    with st.spinner("Memproses..."):
        time.sleep(0.5)
    try:
        train_df_proc, val_df_proc, test_df_proc = prepare_splits(df_train, df_val, df_test, col_label)
        payload = {
            "batch_size": int(batch_size),
            "bilstm_values": bilstm_values,
            "attention_values": attention_values,
            "lr_range": [float(lr_min), float(lr_max)],
            "dropout_range": [float(drop_min), float(drop_max)],
            "weight_decay_range": [float(wd_min), float(wd_max)],
            "max_epochs": int(max_epochs),
            "max_trials": int(max_trials),
            "sampling_pct": {"train": int(samp_train_pct), "val": int(samp_val_pct), "test": int(samp_test_pct)},
            "full_training": full_training == "Ya"
        }
        st.write("Summary konfigurasi:")
        st.json(payload)
        files = {}
        if train_file:
            files["train"] = (train_file.name, train_file.getvalue(), "text/csv")
        if val_file:
            files["val"] = (val_file.name, val_file.getvalue(), "text/csv")
        if test_file:
            files["test"] = (test_file.name, test_file.getvalue(), "text/csv")
        data = {"payload": json.dumps(payload)}
        try:
            r = requests.post(API_URL, data=data, files=files, timeout=API_TIMEOUT)
            r.raise_for_status()
            try:
                resp = r.json()
                st.success("Request sukses")
                st.json(resp)
            except Exception:
                st.success("Request sukses (non-JSON response)")
                st.text(r.text[:1000])
        except requests.RequestException as e:
            st.error(f"Error saat memanggil API: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat persiapan: {e}")
