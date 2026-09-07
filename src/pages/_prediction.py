from __future__ import annotations

import io
import datetime

import joblib
import pandas as pd
import streamlit as st

from src.components._header import page_header, stage_intro
from src.config import Config
from src.pages._common import evidence, fmt_money, read_csv, read_json, show_image

ART = Config.ARTIFACT_DIR
OUT = Config.OUTPUT_DIR / "05_salary_prediction"
DATA_PATH = Config.ROOT_DIR / "data" / "raw" / "ai_jobs_market_2025_2026.csv"

# Các cột đầy đủ của bộ dữ liệu gốc (25 cột) để export CSV khớp format
_ALL_COLUMNS = [
    "job_id", "job_title", "job_category", "years_of_experience",
    "education_required", "city", "country", "remote_work", "company_size",
    "industry", "demand_score", "benefits_score_10", "required_skills",
    "skill_count", "salary_min_usd", "salary_max_usd", "annual_salary_usd",
    "salary_tier", "experience_level", "posting_year", "posting_month",
    "is_senior", "is_remote_friendly", "is_llm_role", "ai_salary_premium_pct",
    "demand_growth_yoy_pct",
]

# Các city hợp lệ theo quốc gia (để Deployment Gate kiểm tra logic)
_CITY_COUNTRY_MAP: dict[str, set[str]] = {
    "Vietnam": {"Ho Chi Minh City", "Hanoi", "Da Nang"},
    "United States": {"New York", "San Francisco", "Seattle", "Austin", "Chicago", "Boston", "Los Angeles"},
    "United Kingdom": {"London", "Manchester", "Edinburgh"},
    "Germany": {"Berlin", "Munich", "Hamburg"},
    "Canada": {"Toronto", "Vancouver", "Montreal"},
    "Australia": {"Sydney", "Melbourne", "Brisbane"},
    "Singapore": {"Singapore"},
    "Japan": {"Tokyo", "Osaka"},
    "India": {"Bangalore", "Mumbai", "Delhi"},
    "France": {"Paris", "Lyon"},
    "Netherlands": {"Amsterdam", "Rotterdam"},
    "Sweden": {"Stockholm", "Gothenburg"},
    "Switzerland": {"Zurich", "Geneva"},
    "Remote": {"Remote"},
}


@st.cache_resource
def _load_bundle():
    path = ART / "model_bundle.joblib"
    return joblib.load(path) if path.exists() else None


def _get_next_job_id() -> int:
    """Đọc file data gốc, lấy max job_id rồi +1 (auto-increment)."""
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
        if "job_id" in df.columns:
            return int(df["job_id"].dropna().astype(int).max()) + 1
    except Exception:
        pass
    return 10001


def _validate_city_country(city: str, country: str) -> str | None:
    """Trả về thông báo lỗi nếu city không khớp với country, None nếu hợp lệ."""
    allowed_cities = _CITY_COUNTRY_MAP.get(country)
    if allowed_cities and city not in allowed_cities:
        return f"City '{city}' không khớp với Country '{country}'."
    return None


def render() -> None:
    page_header(
        "5. Salary prediction",
        "Stage 12 · Queue-based validation + Deployment Gate + Role-based inference",
        "💰",
    )
    stage_intro(
        "Stage 12 · Streamlit Salary Prediction Dashboard",
        "Build a valid raw feature row from training metadata, validate via Deployment Gate Queue, predict in batch with the saved bundle.",
        "User form + artifacts/metadata.json",
        "Predicted salary + empirical 90% error interval + review flags",
    )

    metadata = read_json(ART / "metadata.json")
    bundle = _load_bundle()
    if not metadata or bundle is None:
        st.error("Deployment artifacts are missing. Run `python pipeline.py` first.")
        return

    cats = metadata.get("category_options", {})
    ranges = metadata.get("numeric_ranges", {})
    skills = metadata.get("skill_vocabulary", [])
    interval = float(metadata.get("prediction_interval_abs_error_q90", 0.0))

    # Lấy role từ session (mặc định user nếu chưa có)
    role = st.session_state.get("role", "user")
    is_admin = (role == "admin")

    # ── Khởi tạo Queue trong session_state ──────────────────────────────────
    if "prediction_queue" not in st.session_state:
        st.session_state["prediction_queue"] = []

    demand_r = ranges.get("demand_score", {"min": 0, "max": 100, "median": 80})
    benefit_r = ranges.get("benefits_score_10", {"min": 0, "max": 10, "median": 8})
    yr = ranges.get("years_of_experience", {"min": 1, "max": 15, "median": 6})

    # ── Form nhập liệu ───────────────────────────────────────────────────────
    st.markdown("### ➕ Add to Validation Queue")
    with st.form("add_to_queue_form", clear_on_submit=True):
        a, b, c = st.columns(3)
        job_title  = a.selectbox("Job title",    cats.get("job_title", ["Unknown"]))
        job_category = b.selectbox("Job category", cats.get("job_category", ["Unknown"]))
        education  = c.selectbox("Education",    cats.get("education_required", ["Unknown"]))
        city       = a.selectbox("City",         cats.get("city", ["Unknown"]))
        country    = b.selectbox("Country",      cats.get("country", ["Unknown"]))
        remote     = c.selectbox("Remote work",  cats.get("remote_work", ["Unknown"]))
        company    = a.selectbox("Company size", cats.get("company_size", ["Unknown"]))
        industry   = b.selectbox("Industry",     cats.get("industry", ["Unknown"]))
        years      = c.number_input(
            "Years of experience",
            min_value=float(yr.get("min", 0)),
            max_value=float(yr.get("max", 50)),
            value=float(yr.get("median", 5)),
            step=1.0,
        )

        # Admin: hiện trường đánh giá thị trường | User: ẩn, dùng median
        if is_admin:
            demand = a.number_input(
                "Demand score",
                min_value=float(demand_r.get("min", 0)),
                max_value=float(demand_r.get("max", 100)),
                value=float(demand_r.get("median", 80)),
                step=1.0,
            )
            benefits = b.number_input(
                "Benefits score / 10",
                min_value=float(benefit_r.get("min", 0)),
                max_value=float(benefit_r.get("max", 10)),
                value=float(benefit_r.get("median", 8)),
                step=1.0,
            )
        else:
            demand   = float(demand_r.get("median", 80))
            benefits = float(benefit_r.get("median", 8))
            st.caption("ℹ️ Market evaluation fields (Demand score, Benefits score) are set automatically based on market medians.")

        selected_skills = st.multiselect("Required skills", skills, max_selections=12)
        add_btn = st.form_submit_button("📥 Add to Validation Queue", use_container_width=True)

    if add_btn:
        # Deployment Gate: kiểm tra logic city/country
        validation_error = _validate_city_country(city, country)
        if validation_error:
            st.error(f"⛔ Deployment Gate: {validation_error} Vui lòng chỉnh lại.")
        else:
            entry = {
                "job_title": job_title,
                "job_category": job_category,
                "years_of_experience": years,
                "education_required": education,
                "city": city,
                "country": country,
                "remote_work": remote,
                "company_size": company,
                "industry": industry,
                "demand_score": demand,
                "benefits_score_10": benefits,
                "required_skills": "|".join(dict.fromkeys(selected_skills)),
                "skill_count": len(set(selected_skills)),
            }
            st.session_state["prediction_queue"].append(entry)
            st.success(f"✅ Đã thêm vào hàng đợi ({len(st.session_state['prediction_queue'])} bản ghi)")

    # ── Hiển thị Queue ───────────────────────────────────────────────────────
    queue = st.session_state["prediction_queue"]
    st.markdown(f"### 📋 Validation Queue ({len(queue)} records)")
    if queue:
        queue_df = pd.DataFrame(queue)
        st.dataframe(queue_df, use_container_width=True, hide_index=True)

        col_predict, col_clear, col_download = st.columns([2, 1, 1])

        # Nút Predict hàng loạt
        if col_predict.button("🚀 Predict Annual Salaries", use_container_width=True, type="primary"):
            model_features = metadata["model_features"]
            results = []
            for rec in queue:
                row = pd.DataFrame([rec])
                pred = float(bundle.predict(row[model_features])[0])
                low  = max(0.0, pred - interval)
                high = pred + interval
                results.append({**rec, "predicted_salary": pred, "lower_bound": low, "upper_bound": high})
            result_df = pd.DataFrame(results)
            st.markdown("#### 💡 Batch Prediction Results")
            display_cols = ["job_title", "city", "country", "predicted_salary", "lower_bound", "upper_bound"]
            display_df = result_df[[c for c in display_cols if c in result_df.columns]].copy()
            for col in ["predicted_salary", "lower_bound", "upper_bound"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(fmt_money)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.success(f"Model: **{metadata.get('model_name','—')}** · Interval half-width: **{fmt_money(interval)}**")
            st.session_state["last_predictions"] = result_df

        # Nút Clear Queue
        if col_clear.button("🗑️ Clear Queue", use_container_width=True):
            st.session_state["prediction_queue"] = []
            st.rerun()

        # Nút Download CSV (khớp 25 cột dataset gốc)
        next_id = _get_next_job_id()
        now = datetime.datetime.now()
        rows_for_export = []
        for i, rec in enumerate(queue):
            export_row: dict = {col: "" for col in _ALL_COLUMNS}
            export_row["job_id"] = next_id + i
            export_row["posting_year"]  = now.year
            export_row["posting_month"] = now.month
            for k, v in rec.items():
                if k in export_row:
                    export_row[k] = v
            rows_for_export.append(export_row)
        export_df = pd.DataFrame(rows_for_export, columns=_ALL_COLUMNS)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        col_download.download_button(
            "💾 Download CSV",
            data=csv_bytes,
            file_name=f"new_job_records_{now.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Queue is empty. Add records using the form above.")

    # ── Xem kết quả predict gần nhất (nếu có) ──────────────────────────────
    if "last_predictions" in st.session_state:
        with st.expander("📊 Last batch prediction detail"):
            st.dataframe(st.session_state["last_predictions"], use_container_width=True, hide_index=True)

    st.caption(
        "Geographic cost-of-living adjustment is intentionally not implemented "
        "because the supplied project data does not include a verified cost-of-living index."
    )
    evidence(
        "Prediction interpretation",
        f"Point estimates use empirical ±{fmt_money(interval)} 90th-percentile DEV out-of-fold absolute-error band.",
        "This is a practical error band derived from historical validation residuals, not a formal probabilistic confidence guarantee.",
        "Use for academic scenario exploration; obtain external verified market data before operational compensation decisions.",
        "info",
    )
