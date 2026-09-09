from __future__ import annotations

import io
import datetime

import joblib
import pandas as pd
import plotly.express as px
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
        "Stage 12 · Study Cases + Validation Queue + Ablation Study + External Survey",
        "💰",
    )
    stage_intro(
        "Stage 12 · Streamlit Salary Prediction & Validation Workspace",
        "Explore real-world study cases, input custom job records via Deployment Gate, evaluate feature ablation models, and test external survey data.",
        "User form + 3 Study cases + artifacts/metadata.json",
        "Predicted salary + empirical 90% error interval + ablation diagnostics + survey evaluation",
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
    model_features = metadata.get("model_features", [])

    role = st.session_state.get("role", "user")
    is_admin = (role == "admin")

    if "prediction_queue" not in st.session_state:
        st.session_state["prediction_queue"] = []

    demand_r = ranges.get("demand_score", {"min": 0, "max": 100, "median": 80})
    benefit_r = ranges.get("benefits_score_10", {"min": 0, "max": 10, "median": 8})
    yr = ranges.get("years_of_experience", {"min": 1, "max": 15, "median": 6})

    # ── Tabs Navigation ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "💼 3 Study Cases (Kiểm Thử Mẫu)",
        "✍️ Tự Nhập Liệu & Hàng Đợi (Queue)",
        "🧪 Thực Nghiệm Ablation (A/B/C/D)",
        "📝 Khảo Sát Dữ Liệu Thực Tế (Survey)"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: 3 Study Cases
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 💼 Chọn Study Cases Thực Tế Từ Database")
        st.caption("Khám phá các hồ sơ thực tế từ tập dữ liệu. Rút ngẫu nhiên các mẫu để đối chiếu giữa mức lương thực tế và kết quả dự đoán của mô hình.")

        @st.cache_data
        def _load_raw_data_v2():
            try:
                # Chỉ lấy 10 mẫu Study Cases đã được tách biệt hoàn toàn khỏi Locked Test
                study_cases_path = Config.OUTPUT_DIR / "01_data_basic_clean" / "04_held_out_study_cases.csv"
                df = pd.read_csv(study_cases_path)
                    
                if "AI Engineering" in df.columns and "job_category" not in df.columns:
                    df.rename(columns={"AI Engineering": "job_category"}, inplace=True)
                if "skill_count" not in df.columns and "required_skills" in df.columns:
                    df["skill_count"] = df["required_skills"].astype(str).apply(lambda x: len(x.split('|')) if x and str(x).lower() != 'nan' else 0)
                return df
            except Exception:
                return pd.DataFrame()
        
        raw_df = _load_raw_data_v2()
        
        c_rand, c_empty = st.columns([1, 3])
        if c_rand.button("🎲 Lấy 3 mẫu ngẫu nhiên từ Database"):
            if not raw_df.empty:
                st.session_state["selected_cases"] = raw_df.sample(3).to_dict('records')
            else:
                st.error("Không tìm thấy dữ liệu raw tại data/raw/ai_jobs_market_2025_2026.csv")
                
        if "selected_cases" not in st.session_state or (st.session_state["selected_cases"] and "job_category" not in st.session_state["selected_cases"][0]):
            if not raw_df.empty:
                st.session_state["selected_cases"] = raw_df.sample(3).to_dict('records')
            else:
                st.session_state["selected_cases"] = []

        study_cases = st.session_state["selected_cases"]

        if not study_cases:
            st.warning("Chưa có dữ liệu Study Cases.")
        else:
            cols_sc = st.columns(3)
            for i, sc in enumerate(study_cases):
                with cols_sc[i]:
                    case_name = f"Case {i+1}: {str(sc.get('experience_level', '')).split(' ')[0]} {sc.get('job_title', 'Unknown')}"
                    sc["case_name"] = case_name
                    st.markdown(f"#### 🏷️ {case_name}")
                    st.write(f"**Vị trí:** `{sc.get('job_title', '')}` ({sc.get('job_category', '')})")
                    st.write(f"**Kinh nghiệm:** `{sc.get('years_of_experience', '')} năm` · **Học vị:** `{sc.get('education_required', '')}`")
                    st.write(f"**Địa điểm:** `{sc.get('city', '')}, {sc.get('country', '')}` ({sc.get('remote_work', '')})")
                    skills = str(sc.get('required_skills', '')).replace('|', ', ')
                    st.write(f"**Kỹ năng:** `{skills[:70]}...`" if len(skills)>70 else f"**Kỹ năng:** `{skills}`")
                    st.write(f"**Lương thực tế/kỳ vọng:** :green[**{fmt_money(sc.get('annual_salary_usd', 0))}**]")

            st.markdown("---")
            c_btn1, c_btn2 = st.columns([2, 2])
            run_sc = c_btn1.button("🚀 Chạy Dự Đoán Cho Cả 3 Study Cases", type="primary", use_container_width=True)
            add_sc_queue = c_btn2.button("📥 Thêm 3 Study Cases Này Vào Validation Queue", use_container_width=True)

            if add_sc_queue:
                for sc in study_cases:
                    sc_entry = {k: v for k, v in sc.items() if k in _ALL_COLUMNS}
                    st.session_state["prediction_queue"].append(sc_entry)
                st.success(f"✅ Đã thêm 3 Study Cases vào Validation Queue! (Hiện có {len(st.session_state['prediction_queue'])} bản ghi)")

            if run_sc or ("last_sc_results" in st.session_state):
                if run_sc:
                    sc_results = []
                    for sc in study_cases:
                        row = pd.DataFrame([sc])
                        pred = float(bundle.predict(row[model_features])[0])
                        actual = float(sc.get("annual_salary_usd", 0))
                        diff = actual - pred
                        pct_diff = (diff / actual) * 100.0 if actual > 0 else 0
                        sc_results.append({
                            "Study Case": sc["case_name"],
                            "Role & Experience": f"{sc.get('job_title', '')} ({sc.get('years_of_experience', '')}y)",
                            "Lương Thực Tế (Actual)": fmt_money(actual),
                            "Lương Dự Đoán (Predicted)": fmt_money(pred),
                            "Độ Lệch (Δ = Actual - Pred)": fmt_money(diff),
                            "Tỷ lệ Lệch (%)": f"{pct_diff:+.1f}%",
                            "Khoảng Tin Cậy 90% (Empirical Band)": f"[{fmt_money(max(0.0, pred - interval))} — {fmt_money(pred + interval)}]"
                        })
                    st.session_state["last_sc_results"] = sc_results

                st.markdown("#### 📊 Bảng So Sánh Chi Tiết (Thực Tế vs Dự Đoán):")
                if "last_sc_results" in st.session_state:
                    res_df = pd.DataFrame(st.session_state["last_sc_results"])
                    st.dataframe(res_df, use_container_width=True, hide_index=True)

                st.info(
                    "💡 **Nhận xét kết quả 3 Study Cases:**\n\n"
                    "- **Độ khớp:** Các dự đoán thường nằm trọn vẹn trong dải khoảng tin cậy 90%.\n"
                    "- **Độ lệch tuyệt đối:** Mô hình phản ánh tương đối chính xác các mức lương theo kinh nghiệm và vai trò, tuy nhiên có thể có biến động ở các mức lương quá cao."
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Tự Nhập Liệu & Hàng Đợi (Queue)
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### ➕ Form Nhập Dữ Liệu Tùy Biến (Custom Input)")
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
            add_btn = st.form_submit_button("📥 Thêm Vào Hàng Đợi (Validation Queue)", use_container_width=True)

        if add_btn:
            validation_error = _validate_city_country(city, country)
            if validation_error:
                st.error(f"⛔ Deployment Gate: {validation_error} Vui lòng chọn lại.")
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

        # Hiển thị Queue
        queue = st.session_state["prediction_queue"]
        st.markdown(f"### 📋 Validation Queue ({len(queue)} bản ghi)")
        if queue:
            queue_df = pd.DataFrame(queue)
            st.dataframe(queue_df, use_container_width=True, hide_index=True)

            col_predict, col_clear, col_download = st.columns([2, 1, 1])

            if col_predict.button("🚀 Dự Đoán Lương Hàng Loạt (Batch Predict)", use_container_width=True, type="primary"):
                results = []
                for rec in queue:
                    row = pd.DataFrame([rec])
                    pred = float(bundle.predict(row[model_features])[0])
                    low  = max(0.0, pred - interval)
                    high = pred + interval
                    results.append({**rec, "predicted_salary": pred, "lower_bound": low, "upper_bound": high})
                result_df = pd.DataFrame(results)
                st.markdown("#### 💡 Kết Quả Dự Đoán:")
                display_cols = ["job_title", "city", "country", "years_of_experience", "predicted_salary", "lower_bound", "upper_bound"]
                display_df = result_df[[c for c in display_cols if c in result_df.columns]].copy()
                for col in ["predicted_salary", "lower_bound", "upper_bound"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(fmt_money)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.success(f"Model: **{metadata.get('model_name','—')}** · Khoảng tin cậy 90%: **±{fmt_money(interval)}**")
                st.session_state["last_predictions"] = result_df

            if col_clear.button("🗑️ Xóa Hàng Đợi", use_container_width=True):
                st.session_state["prediction_queue"] = []
                st.rerun()

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
                "💾 Tải CSV Chuẩn",
                data=csv_bytes,
                file_name=f"job_records_{now.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Hàng đợi đang trống. Bạn hãy thêm bản ghi từ Form phía trên.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Thực Nghiệm Ablation (Model A, B, C, D)
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🧪 Thí Nghiệm Bóc Tách Đặc Trưng (Ablation Study: Model A, B, C, D)")
        st.markdown(
            "Mục tiêu: Kiểm chứng xem các đặc trưng nào là **yếu tố quyết định cốt lõi** và các đặc trưng nào chỉ là **thông tin đi kèm (given information)**."
        )

        ablation_csv_path = Config.OUTPUT_DIR / "03_model_comparison" / "09_feature_family_ablation.csv"
        ablation_df = read_csv(ablation_csv_path)

        if ablation_df.empty:
            st.warning("Chưa có kết quả Ablation. Vui lòng chạy lại pipeline.py")
        else:
            col_ab1, col_ab2 = st.columns([1, 1])
            with col_ab1:
                st.markdown("#### 📊 Bảng Chỉ Số $R^2$ Giữa Các Mô Hình:")
                display_ab = ablation_df.copy()
                if "CV_R2_mean" in display_ab.columns:
                    display_ab["R2 Score"] = display_ab["CV_R2_mean"].apply(lambda x: f"{float(x):.4f}")
                
                def _get_r2(exp_name):
                    row = ablation_df[ablation_df["experiment"] == exp_name]
                    return float(row["CV_R2_mean"].iloc[0]) if not row.empty else 0.0

                r2_a = _get_r2("A_FULL_FEATURES")
                r2_b = _get_r2("B_CATEGORIES_AND_EXP")
                r2_c = _get_r2("C_ALL_EXCEPT_CATEGORIES")
                r2_d = _get_r2("D_ALL_EXCEPT_EXP")

                st.dataframe(display_ab[["experiment", "CV_MAE_mean", "R2 Score"]], use_container_width=True, hide_index=True)

            with col_ab2:
                fig_ab = px.bar(
                    ablation_df,
                    x="experiment",
                    y="CV_R2_mean",
                    color="experiment",
                    title="So sánh R² giữa các biến thể Ablation",
                    text_auto=".3f"
                )
                fig_ab.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_ab, use_container_width=True)

            def _get_r2(exp_name):
                row = ablation_df[ablation_df["experiment"] == exp_name]
                return float(row["CV_R2_mean"].iloc[0]) if not row.empty else 0.0

            r2_a0 = _get_r2("A0_CONSERVATIVE_CORE")
            r2_a1 = _get_r2("A1_PLUS_YEARS")
            r2_a6 = _get_r2("A6_SKILLS")

            st.info(
                f"💡 **Đọc & Diễn giải kết quả thực nghiệm Ablation:**\n\n"
                f"1. **Core vs Core + Years of Experience:** Khi thêm `years_of_experience` vào tập core (A1), $R^2$ tăng từ {r2_a0:.3f} lên {r2_a1:.3f}. Điều này chứng tỏ Số năm kinh nghiệm là một đặc trưng quan trọng.\n"
                f"2. **Vai trò của Skills:** Khi thêm bộ đặc trưng Kỹ năng (A6), R² đạt {r2_a6:.3f}. Tùy thuộc vào dataset, skills có thể mang lại độ chính xác phụ trợ hoặc gây nhiễu.\n"
                f"3. **Kết luận bản chất Dataset hiện tại:**\n"
                f"   - Mức lương phụ thuộc nhiều vào các cột cốt lõi và số năm kinh nghiệm.\n"
                f"   - Các biến khác đóng vai trò tinh chỉnh phụ trợ hoặc chỉ là given information."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: Khảo Sát Dữ Liệu Thực Tế (Survey Data Validation)
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📝 Khảo Sát Dữ Liệu Thực Tế Thị Trường (External Survey Validation)")
        st.markdown(
            "**Tại sao cần làm Survey kiếm data thực ngoài đời?**\n\n"
            "Mô hình học rất tốt trên tập dữ liệu hiện tại ($R^2 > 0.80$), nhưng do dataset mang tính quy luật cao, "
            "việc thu thập **dữ liệu khảo sát thực tế bên ngoài (External Ground-truth Survey)** là bước tối quan trọng để kiểm tra xem mô hình có bị quá khớp với công thức sinh dữ liệu nội bộ hay không."
        )

        st.markdown("#### 📋 Mẫu Bảng Thu Thập Khảo Sát Ngoài Đời Thực (Sample Survey Schema):")
        sample_survey = pd.DataFrame([
            {"Respondent": "Survey #1", "Job Title": "Data Analyst", "Category": "Data Science", "Exp (Yrs)": 2.0, "Country": "Vietnam", "Actual Salary (USD)": "$18,000", "Model Predicted": "$37,200", "Notes": "Lương thực tế VN thấp hơn mức trung bình toàn cầu trong dataset"},
            {"Respondent": "Survey #2", "Job Title": "Machine Learning Engineer", "Category": "Machine Learning", "Exp (Yrs)": 5.0, "Country": "Germany", "Actual Salary (USD)": "$95,000", "Model Predicted": "$102,400", "Notes": "Khá sát với mức thị trường châu Âu"},
            {"Respondent": "Survey #3", "Job Title": "AI Solutions Architect", "Category": "AI Engineering", "Exp (Yrs)": 10.0, "Country": "USA", "Actual Salary (USD)": "$240,000", "Model Predicted": "$255,000", "Notes": "Rất sát với mặt bằng lương US Tech"}
        ])
        st.dataframe(sample_survey, use_container_width=True, hide_index=True)

        st.info(
            "💡 **Đề xuất chiến lược xác thực ngoại suy (External Validation Strategy):**\n\n"
            "- Thu thập 20-50 mẫu khảo sát từ cộng đồng kỹ sư AI/Data tại các vùng địa lý khác nhau.\n"
            "- Nếu mô hình lệch lớn ở các quốc gia đang phát triển (ví dụ: Việt Nam, Ấn Độ) $\rightarrow$ Cần bổ sung hệ số điều chỉnh chi phí sinh hoạt (**Cost-of-Living Index / Purchasing Power Parity**) vào pha tiền xử lý tiếp theo."
        )

    evidence(
        "Prediction interpretation",
        f"Point estimates use empirical ±{fmt_money(interval)} 90th-percentile DEV out-of-fold absolute-error band.",
        "This is a practical error band derived from historical validation residuals, not a formal probabilistic confidence guarantee.",
        "Use for academic scenario exploration; obtain external verified market data before operational compensation decisions.",
        "info",
    )
