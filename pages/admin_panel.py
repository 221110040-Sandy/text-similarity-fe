import streamlit as st
import sys
from pathlib import Path
import pandas as pd
from io import StringIO
import requests
import json
import time
from sklearn.model_selection import train_test_split

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

def safe_read_csv(uploaded_file, required_cols):
    if uploaded_file is None:
        return None, False, None
    df = None
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] == 1:
            raise ValueError("single column")
    except Exception:
        try:
            df = read_csv_strip_quotes(uploaded_file)
        except Exception:
            return None, False, "Tidak dapat membaca CSV (format/encoding tidak dikenali)."
    cols = list(df.columns)
    missing = [c for c in required_cols if c not in cols]
    if missing:
        return df, False, f"Kolom wajib hilang → {', '.join(missing)}"
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

# config
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

st.markdown("### Upload CSV (Train / Val / Test)")
col_map, _ = st.columns([2,1])
with col_map:
    st.markdown("**Mapping kolom (sesuaikan dengan header CSV kamu).**")
    col_sent1 = st.text_input("Nama header untuk sentence1", value="sentence1")
    col_sent2 = st.text_input("Nama header untuk sentence2", value="sentence2")
    col_label = st.text_input("Nama header untuk label", value="label")

required_cols = [col_sent1, col_sent2, col_label]

st.info("Train & Validation wajib. Test optional (jika kosong akan di-split dari Train). File max 25MB per file.")

col_train, col_val, col_test = st.columns(3)

train_file = val_file = test_file = None

df_train = df_val = df_test = None

# uploader helper that checks size and reads
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
            st.error("File terlalu besar — maksimal 25 MB")
            train_file = None
            train_valid = False
        else:
            df_train, train_valid, train_err = safe_read_csv(train_file, required_cols)
            if train_err:
                st.error(train_err)
            if df_train is not None and train_valid:
                st.metric("Rows", len(df_train))
                st.metric("Columns", len(df_train.columns))
                st.success("Dokumen valid — semua kolom ditemukan")
            elif df_train is not None and not train_valid:
                st.error("Kolom wajib hilang atau header tidak sesuai: " + ", ".join([c for c in required_cols if c not in list(df_train.columns)]))
    else:
        st.error("Train CSV wajib diupload")

with col_val:
    st.subheader("Validation CSV (required)")
    val_file = st.file_uploader("Upload validation CSV", type=['csv'], key='val_file')
    val_valid = False
    val_err = None
    if val_file:
        if val_file.size > MAX_UPLOAD_BYTES:
            st.error("File terlalu besar — maksimal 25 MB")
            val_file = None
            val_valid = False
        else:
            df_val, val_valid, val_err = safe_read_csv(val_file, required_cols)
            if val_err:
                st.error(val_err)
            if df_val is not None and val_valid:
                st.metric("Rows", len(df_val))
                st.metric("Columns", len(df_val.columns))
                st.success("Dokumen valid — semua kolom ditemukan")
            elif df_val is not None and not val_valid:
                st.error("Kolom wajib hilang atau header tidak sesuai: " + ", ".join([c for c in required_cols if c not in list(df_val.columns)]))
    else:
        st.error("Validation CSV wajib diupload")

with col_test:
    st.subheader("Test CSV (optional)")
    test_file = st.file_uploader("Upload test CSV", type=['csv'], key='test_file')
    test_valid = False
    test_err = None
    if test_file:
        if test_file.size > MAX_UPLOAD_BYTES:
            st.error("File terlalu besar — maksimal 25 MB")
            test_file = None
            test_valid = False
        else:
            df_test, test_valid, test_err = safe_read_csv(test_file, required_cols)
            if test_err:
                st.error(test_err)
            if df_test is not None and test_valid:
                st.metric("Rows", len(df_test))
                st.metric("Columns", len(df_test.columns))
                st.success("Dokumen valid — semua kolom ditemukan")
            elif df_test is not None and not test_valid:
                st.error("Kolom wajib hilang atau header tidak sesuai: " + ", ".join([c for c in required_cols if c not in list(df_test.columns)]))
    else:
        st.info("Test CSV kosong — akan di-split dari Train saat proses dimulai jika diperlukan")

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
        combined = (df[col_sent1].fillna("").astype(str) + " " + df[col_sent2].fillna("").astype(str)).str.len()
        qs = combined.quantile([0.25, 0.5, 0.75, 0.95, 1.0]).to_dict()
        qtable = {25: int(qs.get(0.25, 0)), 50: int(qs.get(0.5, 0)), 75: int(qs.get(0.75, 0)), 95: int(qs.get(0.95, 0)), 100: int(qs.get(1.0, 0))}
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
        qdf = pd.DataFrame({"percentile": ["P25","P50","P75","P95","P100"], "chars": [qt[25], qt[50], qt[75], qt[95], qt[100]]})
        st.table(qdf)
        st.write(f"Max len (use P95): {summary['max_len']}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("Sample rows:")
        st.dataframe(df[[col_sent1, col_sent2, col_label]].head(5), use_container_width=True)

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
        ca.write(f"Attention value {i1+1}: {attention_values[i1]}")
        if i2 < len(attention_values):
            cb.write(f"Attention value {i2+1}: {attention_values[i2]}")

st.markdown("---")

st.markdown("### Hyperband search lists (learning rate, dropout, weight decay)")
col_lr, col_drop, col_wd = st.columns(3)
with col_lr:
    lr_count = st.number_input("Jumlah nilai learning rate", min_value=1, max_value=20, value=3, step=1, key="lr_count")
    lr_values = []
    for i in range(lr_count):
        v = st.number_input(f"LR value {i+1}", value=1e-4 * (10**i) if i>0 else 1e-4, format="%.8g", key=f"lr_v_{i}")
        lr_values.append(float(v))
with col_drop:
    drop_count = st.number_input("Jumlah nilai dropout", min_value=1, max_value=20, value=3, step=1, key="drop_count")
    drop_values = []
    for i in range(drop_count):
        v = st.number_input(f"Dropout value {i+1}", min_value=0.0, max_value=1.0, value=0.1 + 0.1*i, step=0.01, format="%.2f", key=f"drop_v_{i}")
        drop_values.append(float(v))
with col_wd:
    wd_count = st.number_input("Jumlah nilai weight decay", min_value=1, max_value=20, value=2, step=1, key="wd_count")
    wd_values = []
    for i in range(wd_count):
        v = st.number_input(f"WD value {i+1}", value=1e-4 * (i+1), format="%.8g", key=f"wd_v_{i}")
        wd_values.append(float(v))

st.markdown("---")

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

any_uploaded = all([train_file is not None, val_file is not None])
all_uploaded_valid = all([train_valid, val_valid]) and (not test_file or test_valid)

disabled = False
reasons = []
if not any_uploaded:
    disabled = True
    reasons.append("Train dan Validation wajib diupload")
if not all_uploaded_valid:
    disabled = True
    reasons.append("Train/Validation/Test tidak valid (cek pesan di tiap kolom)")

if disabled:
    st.info("Tidak dapat memulai proses: " + "; ".join(reasons))

API_URL = "https://your.api.server/find-hyperparameter"
API_TIMEOUT = 60

# ketika ditekan -> langsung hit API tanpa dialog konfirmasi
if st.button("Mulai Training", type="primary", disabled=disabled):
    cfg = {
        "batch_size": int(batch_size),
        "bilstm_values": bilstm_values,
        "attention_values": attention_values,
        "lr_values": lr_values,
        "dropout_values": drop_values,
        "weight_decay_values": wd_values,
        "max_epochs": int(max_epochs),
        "max_trials": int(max_trials),
        "sampling_pct": {"train": int(samp_train_pct), "val": int(samp_val_pct), "test": int(samp_test_pct)},
        "full_training": full_training == "Ya",
        "column_mapping": {"sentence1": col_sent1, "sentence2": col_sent2, "label": col_label}
    }

    with st.spinner("Mengirim konfigurasi dan file ke backend..."):
        files = {}
        if train_file:
            files["train"] = (train_file.name, train_file.getvalue(), "text/csv")
        if val_file:
            files["val"] = (val_file.name, val_file.getvalue(), "text/csv")
        if test_file:
            files["test"] = (test_file.name, test_file.getvalue(), "text/csv")
        data = {"payload": json.dumps(cfg)}
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
