import shutil
import subprocess

import pandas as pd
import streamlit as st

from src.config import Config

DATA_DIR = Config.ROOT_DIR / "data" / "raw"
DATA_FILE = DATA_DIR / "ai_jobs_market_2025_2026.csv"
BACKUP_FILE = DATA_DIR / "ai_jobs_market_2025_2026_ORIGINAL.csv"


def backup_original():
    if DATA_FILE.exists() and not BACKUP_FILE.exists():
        shutil.copy2(DATA_FILE, BACKUP_FILE)


def render_sidebar_upload():
    # Tính năng này chỉ hiển thị cho admin
    if st.session_state.get("role") != "admin":
        return

    backup_original()
    
    with st.expander("📥 Dataset - upload / restore", expanded=False):
        st.caption(f"Active analytical dataset: `{DATA_FILE.name}`")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV with the same/similar 25-column schema", 
            type=["csv"], 
            help="Limit 200MB per file • CSV"
        )
        
        is_locked_test = st.checkbox("Require document-standard locked test 2026-03", value=True, help="Enforce the exact temporal cutoffs for the 12-stage pipeline.")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                rows = len(df)
                cols = len(df.columns)
                schema_pass = "PASS" if cols == 25 else "FAIL"
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{rows:,}")
                c2.metric("Columns", f"{cols}")
                c3.metric("Schema", schema_pass)
                
                with st.expander("Upload preview"):
                    st.dataframe(df.head(20), use_container_width=True)
                
                if st.button("▶ Analyze uploaded dataset - run full 12-stage pipeline", use_container_width=True):
                    if schema_pass == "FAIL":
                        st.error("Schema must have exactly 25 columns to proceed.")
                    else:
                        with st.spinner("Running data quality, preprocessing, temporal CV, model comparison and deployable bundle..."):
                            uploaded_file.seek(0)
                            with open(DATA_FILE, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            res = subprocess.run(["python", "pipeline.py"], capture_output=True, text=True, cwd=str(Config.ROOT_DIR))
                            
                            if res.returncode == 0:
                                st.session_state["upload_success"] = True
                                st.rerun()
                            else:
                                st.error("Pipeline failed!")
                                st.code(res.stderr)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
        disabled_restore = not BACKUP_FILE.exists()
        if st.button("↩ Restore assignment dataset and rerun", disabled=disabled_restore, use_container_width=True):
            with st.spinner("Restoring original dataset and running pipeline..."):
                shutil.copy2(BACKUP_FILE, DATA_FILE)
                res = subprocess.run(["python", "pipeline.py"], capture_output=True, text=True, cwd=str(Config.ROOT_DIR))
                if res.returncode == 0:
                    st.session_state["restore_success"] = True
                    st.rerun()
                else:
                    st.error("Pipeline failed!")
                    st.code(res.stderr)
