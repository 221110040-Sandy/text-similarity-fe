import streamlit as st
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from utils.auth import initialize_auth_state, require_auth, logout

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

st.markdown("""
<div class='admin-header'>
  <h1>Admin Dashboard</h1>
</div>
""", unsafe_allow_html=True)

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

required = ['sentence1', 'sentence2', 'label']

# Train column
with col_train:
    st.subheader("Train CSV")
    train_file = st.file_uploader("Upload train CSV", type=['csv'], key='train_file')
    train_valid = False
    if train_file:
        try:
            df_train = pd.read_csv(train_file)
            st.dataframe(df_train.head(5), use_container_width=True)
            st.metric("Rows", len(df_train))
            st.metric("Columns", len(df_train.columns))
            cols = list(df_train.columns)
            missing = [c for c in required if c not in cols]
            if missing:
                st.error(f"Kolom wajib hilang → {', '.join(missing)}")
            else:
                st.success("Dokumen valid — semua kolom ditemukan")
                st.table(df_train[required].head(3))
                train_valid = True
        except Exception as e:
            st.error(f"Error membaca Train CSV: {e}")
    else:
        st.info("Belum ada file diupload untuk train")

# Validation column
with col_val:
    st.subheader("Validation CSV")
    val_file = st.file_uploader("Upload validation CSV", type=['csv'], key='val_file')
    val_valid = False
    if val_file:
        try:
            df_val = pd.read_csv(val_file)
            st.dataframe(df_val.head(5), use_container_width=True)
            st.metric("Rows", len(df_val))
            st.metric("Columns", len(df_val.columns))
            cols = list(df_val.columns)
            missing = [c for c in required if c not in cols]
            if missing:
                st.error(f"Kolom wajib hilang → {', '.join(missing)}")
            else:
                st.success("Dokumen valid — semua kolom ditemukan")
                st.table(df_val[required].head(3))
                val_valid = True
        except Exception as e:
            st.error(f"Error membaca Validation CSV: {e}")
    else:
        st.info("Belum ada file diupload untuk validation")

# Test column
with col_test:
    st.subheader("Test CSV")
    test_file = st.file_uploader("Upload test CSV", type=['csv'], key='test_file')
    test_valid = False
    if test_file:
        try:
            df_test = pd.read_csv(test_file)
            st.dataframe(df_test.head(5), use_container_width=True)
            st.metric("Rows", len(df_test))
            st.metric("Columns", len(df_test.columns))
            cols = list(df_test.columns)
            missing = [c for c in required if c not in cols]
            if missing:
                st.error(f"Kolom wajib hilang → {', '.join(missing)}")
            else:
                st.success("Dokumen valid — semua kolom ditemukan")
                st.table(df_test[required].head(3))
                test_valid = True
        except Exception as e:
            st.error(f"Error membaca Test CSV: {e}")
    else:
        st.info("Belum ada file diupload untuk test")

st.markdown("---")

st.markdown("### Training Options")
batch_size = st.radio("Batch size", options=[16,32,64,128], index=1)

any_uploaded = any([train_file, val_file, test_file])
all_uploaded_valid = True
if train_file and not train_valid:
    all_uploaded_valid = False
if val_file and not val_valid:
    all_uploaded_valid = False
if test_file and not test_valid:
    all_uploaded_valid = False

disabled = False
reasons = []
if not any_uploaded:
    disabled = True
    reasons.append("Belum ada CSV yang diupload")
if not all_uploaded_valid:
    disabled = True
    reasons.append("Salah satu atau lebih file tidak valid (cek pesan di tiap kolom)")
if disabled:
    st.info("Tidak dapat memulai training: " + "; ".join(reasons))

if st.button("Mulai Training", type="primary", disabled=disabled):
    with st.spinner("Memulai training (dummy)..."):
        import time
        time.sleep(1)
    st.success("Training started (dummy)")
    uploaded = [getattr(f,'name','') for f in [train_file,val_file,test_file] if f]
    st.write("Uploaded files:", uploaded)
    st.write("Batch size:", batch_size)
