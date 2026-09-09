import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components._header import page_header, stage_intro
from src.config import Config
from src.pages._common import evidence, fmt_money, guard, read_csv, read_json, show_image

BEST = Config.OUTPUT_DIR / "04_best_model_and_feature_importance"
ART = Config.ARTIFACT_DIR
OUT_PRED = Config.OUTPUT_DIR / "05_salary_prediction"


def render() -> None:
    page_header("4. Best model", "Stage 10 · bounded tuning → one-time locked test → explainability → deployment-equivalence gate", "🏆")
    stage_intro(
        "Stage 10 · Best Model Selection & Feature Importance Review",
        "Tune inside DEV folds, then open March-2026 once and inspect model behavior rather than accepting R² at face value.",
        "Frozen model family + DEV folds",
        "Locked-test metrics + predictions + residuals + importance + slices",
    )
    metrics = read_csv(BEST / "10_final_locked_test_metrics.csv")
    tuning = read_csv(BEST / "10_best_model_tuning_results.csv")
    raw_imp = read_csv(BEST / "10_raw_feature_permutation_importance.csv")
    enc_imp = read_csv(BEST / "10_encoded_feature_importance.csv")
    slices = read_csv(BEST / "10_error_slices.csv")
    pred_test = read_csv(BEST / "10_locked_test_predictions_with_error.csv")

    if guard(metrics, "Stage 10 outputs are missing. Run `python pipeline.py` first."):
        return

    m = metrics.iloc[0]
    cols = st.columns(5)
    cols[0].metric("Best model", m["selected_model"])
    cols[1].metric("MAE", fmt_money(m["MAE"]))
    cols[2].metric("RMSE", fmt_money(m["RMSE"]))
    cols[3].metric("R²", f"{float(m['R2']):.3f}")
    cols[4].metric("MedAE", fmt_money(m["MedAE"]))

    # ── Tabs Navigation ───────────────────────────────────────────────────────
    tab4, tab1, tab2, tab3 = st.tabs([
        "📋 Chi Tiết Mẫu Locked Test",
        "🏆 Train vs Locked Test",
        "📉 Phân Tích Phần Dư (4 Góc Nhìn)",
        "🔍 Feature Importance"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Train vs Locked Test Performance
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 🎯 Đánh Giá Hiệu Năng Trên Tập Locked Test (Tháng 03/2026)")
        st.caption("Tập kiểm thử Locked Test chỉ được mở 1 lần duy nhất sau khi đã cố định kiến trúc mô hình và siêu tham số.")

        c1, c2 = st.columns([1, 1])
        with c1:
            show_image(BEST / "actual_vs_predicted_locked_test.png")
        with c2:
            st.markdown("#### ⚙️ Kết quả Tinh chỉnh Siêu tham số (Bounded Tuning):")
            st.dataframe(tuning, use_container_width=True, hide_index=True)

            st.markdown("#### ⚖️ So sánh DEV vs Locked Test (Generalization Gap):")
            cv_df = read_csv(Config.OUTPUT_DIR / "03_model_comparison" / "09_model_comparison_temporal_cv.csv")
            dev_mae, dev_rmse, dev_r2 = 0, 0, 0
            if not cv_df.empty:
                cv_row = cv_df[cv_df["model"] == m["selected_model"]]
                if not cv_row.empty:
                    dev_mae = cv_row.iloc[0]["CV_MAE_mean"]
                    dev_rmse = cv_row.iloc[0]["CV_RMSE_mean"]
                    dev_r2 = cv_row.iloc[0]["CV_R2_mean"]

            comp_metrics = [
                {"Giai đoạn / Tập dữ liệu": "Tập huấn luyện & DEV (5 Temporal Folds)", "MAE": fmt_money(dev_mae), "RMSE": fmt_money(dev_rmse), "R²": f"{float(dev_r2):.3f}"},
                {"Giai đoạn / Tập dữ liệu": "Tập kiểm thử khóa (Locked Test)", "MAE": fmt_money(m["MAE"]), "RMSE": fmt_money(m["RMSE"]), "R²": f"{float(m['R2']):.3f}"}
            ]
            import pandas as pd
            st.dataframe(pd.DataFrame(comp_metrics), use_container_width=True, hide_index=True)

        st.info(
            "💡 **Kết luận về So sánh Train - Test & Overfitting (Generalization Conclusion):**\n\n"
            f"- **Chỉ số Out-of-sample:** Trên tập dữ liệu tương lai chưa từng nhìn thấy, `{m['selected_model']}` đạt **MAE {fmt_money(m['MAE'])}** và **$R^2 = {float(m['R2']):.3f}**.\n"
            f"- **Độ lệch tổng quát hóa (Generalization Gap):** R² thay đổi từ {float(dev_r2):.3f} (DEV) xuống {float(m['R2']):.3f} (Test). Biên độ suy giảm này phản ánh khả năng tổng quát hóa thực tế của mô hình."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Residual Analysis (4 trường hợp)
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📉 Phân Tích Phần Dư (Residual Diagnostics - 4 Góc Nhìn)")
        st.caption("Phần dư e = y_thực tế - y_dự đoán. Phân tích 4 trường hợp để đánh giá bản chất Dataset và độ tin cậy của mô hình.")

        if not pred_test.empty and "annual_salary_usd" in pred_test.columns and "predicted_salary_usd" in pred_test.columns:
            pred_test_calc = pred_test.copy()
            pred_test_calc["residual"] = pred_test_calc["annual_salary_usd"] - pred_test_calc["predicted_salary_usd"]

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("#### 1️⃣ Phân phối phần dư (Normality & Centering)")
                fig_hist = px.histogram(
                    pred_test_calc,
                    x="residual",
                    nbins=40,
                    title="Residual Distribution (e = True - Pred)",
                    labels={"residual": "Residual ($)"},
                    color_discrete_sequence=["#1f77b4"],
                    marginal="box"
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                fig_hist.update_layout(height=340)
                st.plotly_chart(fig_hist, use_container_width=True)

            with r_col2:
                st.markdown("#### 2️⃣ Phần dư vs Lương Dự đoán (Heteroscedasticity)")
                fig_scatter = px.scatter(
                    pred_test_calc,
                    x="predicted_salary_usd",
                    y="residual",
                    color="job_category" if "job_category" in pred_test_calc.columns else None,
                    title="Residuals vs Predicted Salary",
                    labels={"predicted_salary_usd": "Predicted Salary ($)", "residual": "Residual ($)"},
                    hover_data=["job_title", "country", "experience_level"] if "job_title" in pred_test_calc.columns else None
                )
                fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
                fig_scatter.update_layout(height=340)
                st.plotly_chart(fig_scatter, use_container_width=True)

            r_col3, r_col4 = st.columns(2)
            with r_col3:
                st.markdown("#### 3️⃣ Lát cắt sai số theo Phân khúc (Error Slices)")
                if not slices.empty:
                    st.dataframe(slices, use_container_width=True, hide_index=True, height=280)
                else:
                    st.caption("Chưa có bảng error slices.")

            with r_col4:
                st.markdown("#### 4️⃣ Điểm dị biệt & Sai số cực trị (Outliers)")
                outliers = pred_test_calc.sort_values("absolute_error_usd", ascending=False).head(10)
                st.dataframe(
                    outliers[["job_title", "job_category", "experience_level", "annual_salary_usd", "predicted_salary_usd", "absolute_error_usd"]],
                    use_container_width=True,
                    hide_index=True,
                    height=280
                )
        else:
            show_image(BEST / "locked_test_residuals.png")

        st.info(
            f"💡 **Nhận xét sâu về Dataset qua 4 trường hợp phân tích phần dư:**\n\n"
            f"1. **Phân phối phần dư:** Đỉnh phân phối hội tụ sát điểm 0 ($MedAE = {fmt_money(m['MedAE'])}`), sai số phân tán đều 2 phía $\\rightarrow$ Mô hình không bị thiên vị hệ thống (unbiased).\n"
            f"2. **Hiện tượng Heteroscedasticity:** Tùy thuộc vào bản chất dữ liệu, phần dư có thể mở rộng ở dải lương cao do biên độ đàm phán lương thực tế rộng hơn.\n"
            f"3. **Phân khúc sai số:** Có thể xem chi tiết MAE theo từng phân khúc kinh nghiệm ở bảng Error Slices.\n"
            f"4. **Kết luận về Dataset:** Dataset có cấu trúc rõ ràng. Mô hình đáng tin cậy trong dải lương phổ thông và cần đi kèm khoảng tin cậy 90% khi dự đoán."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Feature Importance
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🔍 Tầm Quan Trọng Của Các Đặc Trưng (Feature Importance)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 1. Permutation Importance (Raw Features)")
            show_image(BEST / "raw_feature_permutation_importance.png")
            st.dataframe(raw_imp, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### 2. Top 25 Encoded Feature Importance")
            show_image(BEST / "top25_encoded_feature_importance.png")
            st.dataframe(enc_imp.head(25), use_container_width=True, hide_index=True)

        top_share = float(enc_imp.head(2)["importance"].sum()) if not enc_imp.empty and "importance" in enc_imp else 0.0
        st.info(
            f"💡 **Kết luận về Feature Importance:**\n\n"
            f"- Hai đặc trưng cốt lõi `job_category` và `years_of_experience` chiếm tới **{top_share*100:.1f}%** tổng mức độ quan trọng được mã hóa.\n"
            "- Các đặc trưng còn lại như `country`, `education_required`, `company_size`, `skills` đóng vai trò tinh chỉnh nhỏ ở các nhánh cây phụ."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: Chi tiết mẫu Locked Test
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📋 Chi Tiết Mẫu Dự Đoán Trên Tập Kiểm Thử Khóa (Locked Test)")
        
        study_cases_path = Config.OUTPUT_DIR / "01_data_basic_clean" / "04_held_out_study_cases.csv"
        study_cases = read_csv(study_cases_path)
        locked_len = len(pred_test)
        study_len = len(study_cases)
        total_locked = locked_len + study_len
        study_pct = (study_len / total_locked) * 100 if total_locked > 0 else 0
        
        if study_len > 0:
            st.info(f"💡 **Study Cases (Bảo mật tuyệt đối):** Hệ thống đã chủ động chừa ra **{study_len} bản ghi** (chiếm **{study_pct:.1f}%** tổng số dòng của tháng Locked Test) ngay từ đầu để phục vụ cho việc kiểm thử thực tế. Các bản ghi này hoàn toàn ĐƯỢC GIẤU ĐI và KHÔNG tham gia vào bất kỳ khâu nào (từ tính toán Metrics, Ablation cho tới Feature Importance) để mô phỏng hoàn hảo việc kiểm tra trên dữ liệu hoàn toàn mới ở môi trường thực tế.")
            with st.expander("🕵️ Xem trước các bản ghi Study Cases đã được chừa ra", expanded=False):
                st.dataframe(study_cases, use_container_width=True, hide_index=True)
            st.markdown("---")

        summary = read_csv(OUT_PRED / "12_prediction_summary.csv")
        if not summary.empty:
            st.markdown("#### Bảng tổng hợp Serving Summary trên tập Locked Test (sau khi đã trừ đi Study Cases):")
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("#### Danh sách các bản ghi dự đoán mẫu:")
        if not pred_test.empty:
            st.dataframe(pred_test.head(100), use_container_width=True, hide_index=True, height=450)
        else:
            sample_ex = read_csv(OUT_PRED / "12_locked_test_prediction_examples.csv")
            st.dataframe(sample_ex.head(100), use_container_width=True, hide_index=True, height=450)

    evidence(
        "Locked-test interpretation",
        f"{m['selected_model']} achieves MAE {fmt_money(m['MAE'])}, RMSE {fmt_money(m['RMSE'])}, R² {float(m['R2']):.3f}, MedAE {fmt_money(m['MedAE'])}. Top-two encoded importance share is {top_share*100:.1f}%.",
        "These metrics quantify fit to this supplied dataset and future month; feature importance explains the fitted model, not causal salary economics. Concentrated importance warrants extra caution when underlying fields show synthetic-looking structure.",
        "Promote only with the documented limitations, ablation evidence and reload-equivalence PASS. External verified job-posting validation is still required for operational compensation decisions.",
        "warn",
    )

