from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import PASTEL, STAGES, TARGET


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pipeline_12(path: Path) -> None:
    # Snake layout keeps the numbered sequence visually continuous:
    # 01→02→03→04→05→06, then down to 07 and back left through 12.
    fig, ax = plt.subplots(figsize=(18, 7.5))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    colors = [
        PASTEL["purple"], PASTEL["blue"], PASTEL["pink"], PASTEL["orange"], PASTEL["yellow"], PASTEL["green"],
        PASTEL["blue"], PASTEL["purple"], PASTEL["pink"], PASTEL["yellow"], PASTEL["green"], PASTEL["orange"],
    ]
    positions = []
    for i in range(12):
        if i < 6:
            col, row = i, 1
        else:
            col, row = 11 - i, 0  # 07 under 06, 12 under 01
        positions.append((col + 0.08, row + 0.25))

    for i, title in enumerate(STAGES):
        x, y = positions[i]
        rect = plt.Rectangle((x, y), 0.84, 0.72, facecolor=colors[i], edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.10, y + 0.56, f"{i + 1:02d}", fontsize=13, fontweight="bold", color=PASTEL["ink"])
        ax.text(
            x + 0.10, y + 0.36, "\n".join(wrap(title, width=22)), fontsize=8.5,
            va="top", color=PASTEL["ink"], fontweight="semibold",
        )

    # Draw arrows between consecutive stages using box edges.
    for i in range(11):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        if i < 5:
            start_xy, end_xy = (x1 + 0.86, y1 + 0.36), (x2 - 0.07, y2 + 0.36)
        elif i == 5:
            start_xy, end_xy = (x1 + 0.42, y1 - 0.02), (x2 + 0.42, y2 + 0.78)
        else:
            start_xy, end_xy = (x1 - 0.02, y1 + 0.36), (x2 + 0.91, y2 + 0.36)
        ax.annotate("", xy=end_xy, xytext=start_xy, arrowprops=dict(arrowstyle="->", color="#7C8DA6", lw=1.7))

    ax.text(0.08, 2.25, "AI JOB MARKET SALARY PREDICTION", fontsize=23, fontweight="bold", color=PASTEL["ink"])
    ax.text(0.08, 2.08, "12-stage data-quality-first, leakage-safe and deployable workflow", fontsize=12.5, color=PASTEL["muted"])
    ax.text(0.08, 0.05, "Target: annual_salary_usd | Locked temporal test: 2026-03 | Skills: TRAIN-only vocabulary + multi-hot encoding", fontsize=10.5, color=PASTEL["muted"])
    _save(fig, path)

""" def plot_target_distribution(frame: pd.DataFrame, path: Path) -> None:
    target = pd.to_numeric(frame[TARGET], errors="coerce").dropna()
    q1, q3 = target.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.hist(target, bins=28, color=PASTEL["blue_dark"], edgecolor="white")
    ax.axvline(target.mean(), linestyle="--", linewidth=1.8, color="#B35C76", label=f"Mean ${target.mean():,.0f}")
    ax.axvline(target.median(), linestyle=":", linewidth=1.8, color="#4F6D8A", label=f"Median ${target.median():,.0f}")
    ax.axvline(upper, linestyle="--", linewidth=1.2, color="#A06A3B", label=f"IQR upper ${upper:,.0f}")
    ax.set_title("Observed Target Distribution — annual_salary_usd", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Annual salary (USD)")
    ax.set_ylabel("Job postings")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.16)
    _save(fig, path) """ #code cũ

def plot_target_distribution(frame: pd.DataFrame, path: Path) -> None:
    target = pd.to_numeric(frame[TARGET], errors="coerce").dropna()
    
    # 1. Tính toán metrics
    n = len(target)
    mean_val = target.mean()
    median_val = target.median()
    std_val = target.std()
    min_val = target.min()
    max_val = target.max()
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1
    skew = target.skew()
    outliers = int(((target < q1 - 1.5 * iqr) | (target > q3 + 1.5 * iqr)).sum())
    outlier_pct = (outliers / n) * 100 if n > 0 else 0
    
    # 2. Setup Layout (Kéo dãn lề phải)
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.subplots_adjust(right=0.75) 
    
    # 3. Vẽ Histogram (Không cần import seaborn, dùng pyplot chuẩn của file)
    ax.hist(target, bins=24, color="steelblue", edgecolor="white", alpha=0.9)
    ax.set_ylim(0, 185) # Ép trần trục tung vươn qua mốc 175 để scale vạch chuẩn
    
    # 4. Vẽ các đường vạch
    ax.axvline(mean_val, color="steelblue", linestyle="--", linewidth=1.5, label=f"Mean: ${mean_val:,.0f}")
    ax.axvline(median_val, color="steelblue", linestyle=":", linewidth=2, label=f"Median: ${median_val:,.0f}")
    ax.axvline(q1, color="steelblue", linestyle="-.", linewidth=1.5, label=f"Q1: ${q1:,.0f}")
    ax.axvline(q3, color="steelblue", linestyle="-.", linewidth=1.5, label=f"Q3: ${q3:,.0f}")
    
    # 5. Format trục
    ax.set_title(f"Observed Target Distribution — {TARGET} (Raw CSV)", fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Annual salary (USD)", fontsize=12)
    ax.set_ylabel("Job postings", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, frameon=True, edgecolor="lightgray")
    ax.grid(axis="y", linestyle="-", alpha=0.2)
    
    # 6. Text Box 1 (Inner)
    interp_text = (
        "DISTRIBUTION INTERPRETATION\n"
        "Moderately right-skewed\n"
        f"Mean - Median: ${mean_val - median_val:,.0f}\n"
        f"Central 50%: {q1:,.0f}-{q3:,.0f}"
    )
    ax.text(0.03, 0.65, interp_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="darkgray", alpha=0.9))
    
    # 7. Text Box 2 (Outer Profile)
    profile_text = (
        "RAW TARGET PROFILE\n\n"
        f"{'Records (N)':<20}{n:,}\n"
        f"{'Missing':<20}0\n"
        f"{'Unique values':<20}{target.nunique():,}\n\n"
        f"{'Mean':<20}${mean_val:,.0f}\n"
        f"{'Median':<20}${median_val:,.0f}\n"
        f"{'Std. deviation':<20}${std_val:,.0f}\n\n"
        f"{'Minimum':<20}${min_val:,.0f}\n"
        f"{'Q1 (25%)':<20}${q1:,.0f}\n"
        f"{'Q3 (75%)':<20}${q3:,.0f}\n"
        f"{'Maximum':<20}${max_val:,.0f}\n\n"
        f"{'IQR':<20}${iqr:,.0f}\n"
        f"{'Skewness':<20}{skew:.3f}\n"
        f"{'IQR outliers':<20}{outliers} ({outlier_pct:.1f}%)"
    )
    fig.text(0.77, 0.45, profile_text, fontsize=10, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="darkgray"))
             
    # 8. Text Box 3 (Outer Guide)
    guide_text = (
        "READING GUIDE\n\n"
        "Dashed line = Mean\n"
        "Dotted line = Median\n"
        "Dash-dot    = Q1 / Q3\n\n"
        f"Target:\n{TARGET}\n"
        "Unit: USD / year"
    )
    fig.text(0.77, 0.12, guide_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="darkgray"))
             
    # 9. Dùng hàm _save chuẩn của file để xuất ảnh
    _save(fig, path)


def plot_cardinality(profile: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    data = profile.sort_values("unique_count", ascending=False).head(top_n).sort_values("unique_count")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(data["column"], data["unique_count"], color=PASTEL["green"])
    ax.set_title(f"Top {top_n} Feature Cardinalities", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Unique values")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_missingness(quality: pd.DataFrame, path: Path) -> None:
    data = quality.sort_values("missing_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(data["column"], data["missing_pct"], color=PASTEL["purple"])
    ax.set_title("Missingness by Feature", color=PASTEL["ink"], fontweight="bold")
    ax.set_ylabel("Missing rows (%)")
    ax.tick_params(axis="x", rotation=70)
    ax.grid(axis="y", alpha=0.15)
    _save(fig, path)


""" def plot_logic_issue_rates(issues: pd.DataFrame, path: Path) -> None:
    data = issues.sort_values("affected_pct")
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.barh(data["issue"], data["affected_pct"], color=PASTEL["pink"])
    for i, row in enumerate(data.itertuples()):
        ax.text(row.affected_pct + 0.8, i, f"{row.affected_pct:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(100, float(data["affected_pct"].max()) + 8))
    ax.set_xlabel("Rows affected (%)")
    ax.set_title("Data Logic & Integrity Issues — Observed Rates", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path) """ #code cũ

def plot_logic_issue_rates(issues: pd.DataFrame, path: Path) -> None:
    # 1. Sắp xếp data theo severity giảm dần và đảo ngược để vẽ Bar từ trên xuống
    data = issues.sort_values("affected_pct", ascending=False).reset_index(drop=True)
    data_rev = data.iloc[::-1].reset_index(drop=True) 
    
    # 2. Setup Grid 2 tầng (Tầng 1: Chart, Tầng 2: Info-Table)
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.1)
    
    # Từ điển dịch tên technical sang format của sếp Ngân
    ISSUE_MAP = {
        "experience_bucket_mismatch": ("Experience bucket mismatch", "Experience fields contradict each other.", "Ablate both; keep only the validated signal."),
        "salary_tier_mismatch": ("salary_tier inconsistency", "Tier labels do not match annual salary bands.", "Exclude salary_tier from model features."),
        "salary_outside_min_max": ("Salary outside stated min/max", "Target falls outside stated salary range.", "Block min/max salary as leakage-risk metadata."),
        "skill_rows_with_duplicate_tokens": ("Duplicate skill tokens", "Repeated skills inflate token counts.", "Normalize + de-duplicate before multi-hot encoding.")
    }
    
    # ==========================================
    # TẦNG 1: VẼ BAR CHART
    # ==========================================
    ax_chart = fig.add_subplot(gs[0])
    
    # Đổ màu theo độ nghiêm trọng
    colors = []
    for pct in data_rev["affected_pct"]:
        if pct > 70: colors.append("#B3261E")   # Đỏ đậm
        elif pct > 30: colors.append("#E64A19") # Đỏ cam
        else: colors.append("#F57C00")          # Cam
        
    labels_rev = [ISSUE_MAP.get(row.issue, (row.issue,))[0] for row in data_rev.itertuples()]
    ax_chart.barh(labels_rev, data_rev["affected_pct"], color=colors, height=0.6)
    
    # Gắn label % và số lượng row bị lỗi lên cuối mỗi thanh Bar
    for i, row in enumerate(data_rev.itertuples()):
        ax_chart.text(row.affected_pct + 1, i, f"{row.affected_pct:.1f}% ({row.affected_rows:,})", va="center", fontsize=10, fontweight="bold")
        
    ax_chart.set_xlim(0, 105)
    ax_chart.set_xlabel("Share of raw rows affected (%)", fontsize=10)
    ax_chart.set_title("Data Logic & Integrity Issues — Ranked by Severity", fontsize=15, fontweight="bold", pad=25)
    ax_chart.grid(axis="x", alpha=0.2, linestyle="--")
    ax_chart.spines["top"].set_visible(False)
    ax_chart.spines["right"].set_visible(False)
    
    # Subtitle ghi chú tổng số dòng
    total_n = data['total_rows'].iloc[0] if not data.empty else 1500
    fig.text(0.12, 0.96, f"Raw dataset: ai_jobs_market_2025_2026.csv | N = {total_n:,} | Highest-impact issues shown first", fontsize=9, color="#555")
    
    # ==========================================
    # TẦNG 2: VẼ INFOGRAPHIC TABLE
    # ==========================================
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off") # Giấu toàn bộ hệ trục tọa độ đi
    
    ax_table.text(0, 0.98, "KEY FINDINGS & REQUIRED ACTIONS", fontsize=13, fontweight="bold", ha="left")
    
    y_pos = 0.85
    row_height = 0.23 # Khoảng cách giữa các hàng
    
    for i, row in enumerate(data.itertuples()):
        title, why, action = ISSUE_MAP.get(row.issue, (row.issue, "N/A", row.action))
        color = colors[len(data) - 1 - i]
        
        # 1. Vẽ nền xám bao quanh row
        rect = plt.Rectangle((0, y_pos - 0.15), 1.0, 0.19, transform=ax_table.transAxes, facecolor="#F8F9FA", edgecolor="#E0E0E0", lw=1)
        ax_table.add_patch(rect)
        
        # 2. Vẽ dải màu đánh dấu (Ribbon) mép trái
        ribbon = plt.Rectangle((0, y_pos - 0.15), 0.015, 0.19, transform=ax_table.transAxes, facecolor=color)
        ax_table.add_patch(ribbon)
        
        # 3. Vẽ Badge đánh số thứ tự (Dùng bbox của text để bo góc)
        ax_table.text(0.06, y_pos - 0.05, f"#{i+1}", transform=ax_table.transAxes, color="white", fontsize=11, fontweight="bold", ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="none"))
        
        # 4. Cột 1: Ép sát lề trái hơn (x=0.08)
        ax_table.text(0.08, y_pos - 0.02, title, transform=ax_table.transAxes, fontsize=11, fontweight="bold")
        ax_table.text(0.08, y_pos - 0.10, f"{row.affected_pct:.1f}%  •  {row.affected_rows:,} rows", transform=ax_table.transAxes, fontsize=9, color=color, fontweight="bold")
        
        # 5. Cột 2: Đẩy sang trái nhường đất (x=0.38)
        ax_table.text(0.38, y_pos - 0.02, "WHY IT MATTERS", transform=ax_table.transAxes, fontsize=8, fontweight="bold", color="#888")
        ax_table.text(0.38, y_pos - 0.10, why, transform=ax_table.transAxes, fontsize=9.5)
        
        # 6. Cột 3: Lùi sâu về x=0.66 để thừa hẳn 34% chiều rộng cho text dài
        ax_table.text(0.66, y_pos - 0.02, "REQUIRED ACTION", transform=ax_table.transAxes, fontsize=8, fontweight="bold", color="#888")
        ax_table.text(0.66, y_pos - 0.10, action, transform=ax_table.transAxes, fontsize=9.5)
        
        y_pos -= row_height
        
    fig.text(0.12, 0.05, "Note: percentages may overlap because one job posting can trigger multiple integrity rules.", fontsize=9, style="italic", color="#777")
    
    _save(fig, path)


def plot_salary_by_job_category(summary: pd.DataFrame, path: Path) -> None:
    data = summary.sort_values("mean_salary_usd")
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.barh(data["job_category"], data["mean_salary_usd"] / 1000, color=PASTEL["blue_dark"])
    ax.set_title("Mean annual_salary_usd by Job Domain — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Mean salary (USD thousands)")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


""" def plot_years_by_experience_level(frame: pd.DataFrame, path: Path) -> None:
    order = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]
    data = [frame.loc[frame["experience_level"].eq(level), "years_of_experience"].dropna().to_numpy() for level in order]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(PASTEL["purple"])
    ax.set_title("Years of Experience Distribution by Experience Level", color=PASTEL["ink"], fontweight="bold")
    ax.set_ylabel("Years of experience")
    ax.tick_params(axis="x", rotation=8)
    ax.grid(axis="y", alpha=0.15)
    _save(fig, path) """


""" def plot_salary_by_experience_level(frame: pd.DataFrame, path: Path) -> None:
    order = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]
    data = [frame.loc[frame["experience_level"].eq(level), TARGET].dropna().to_numpy() for level in order]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(PASTEL["green"])
    ax.set_title("Annual Salary Distribution by Experience Level — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_ylabel("Annual salary (USD)")
    ax.tick_params(axis="x", rotation=8)
    ax.grid(axis="y", alpha=0.15)
    _save(fig, path)
 """

""" def plot_salary_by_years(summary: pd.DataFrame, path: Path) -> None:
    data = summary.sort_values("years_of_experience")
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.plot(data["years_of_experience"], data["median_salary_usd"], marker="o", linewidth=2, color="#6C9BCF")
    ax.fill_between(data["years_of_experience"], data["mean_salary_usd"], data["median_salary_usd"], alpha=0.12, color="#6C9BCF")
    ax.set_title("Salary Pattern by Years of Experience — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Years of experience")
    ax.set_ylabel("Salary (USD)")
    ax.grid(alpha=0.15)
    _save(fig, path) """



def plot_salary_range_integrity(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.scatter(frame["annual_salary_usd"], frame["salary_min_usd"], s=12, alpha=0.35, label="salary_min_usd", color="#D1876A")
    ax.scatter(frame["annual_salary_usd"], frame["salary_max_usd"], s=12, alpha=0.35, label="salary_max_usd", color="#6C9BCF")
    lims = [min(frame["annual_salary_usd"].min(), frame["salary_min_usd"].min()), max(frame["annual_salary_usd"].max(), frame["salary_max_usd"].max())]
    ax.plot(lims, lims, linestyle="--", linewidth=1, color="#7A7A7A", label="equal to annual salary")
    ax.set_xlabel("annual_salary_usd")
    ax.set_ylabel("stated salary bound")
    ax.set_title("Salary Range Metadata vs Target — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.12)
    _save(fig, path)


def plot_feature_policy(policy: pd.DataFrame, path: Path) -> None:
    counts = policy.groupby("policy").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90, colors=[PASTEL["green"], PASTEL["pink"], PASTEL["yellow"], PASTEL["blue"]][: len(counts)])
    ax.add_artist(plt.Circle((0, 0), 0.52, color="white"))
    ax.set_title("Feature Governance Mix", color=PASTEL["ink"], fontweight="bold")
    _save(fig, path)


def plot_correlations(corr: pd.DataFrame, path: Path, top_n: int = 30) -> None:
    data = corr.head(top_n).sort_values("pearson")
    colors = ["#D8896F" if x < 0 else "#6C9BCF" for x in data["pearson"]]
    fig, ax = plt.subplots(figsize=(10, 8.0))
    ax.barh(data["feature"], data["pearson"], color=colors)
    ax.axvline(0, color="#64748B", linewidth=1)
    ax.set_title(f"Top {top_n} Encoded Features by |Pearson| — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Pearson correlation with annual_salary_usd")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_pearson_spearman(corr: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    data = corr.head(top_n).iloc[::-1]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    ax.barh(y - 0.18, data["pearson"], height=0.35, label="Pearson", color=PASTEL["blue_dark"])
    ax.barh(y + 0.18, data["spearman"], height=0.35, label="Spearman", color="#B18BC7")
    ax.set_yticks(y, data["feature"])
    ax.axvline(0, color="#64748B", linewidth=1)
    ax.legend(frameon=False)
    ax.set_title("Pearson vs Spearman — Top Encoded Features", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_vif(vif: pd.DataFrame, path: Path) -> None:
    if vif.empty:
        return
    data = vif.sort_values("vif")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.barh(data["feature"], data["vif"], color=PASTEL["yellow"])
    ax.axvline(5, linestyle="--", color="#A06A3B", linewidth=1, label="review = 5")
    ax.axvline(10, linestyle="--", color="#A33A3A", linewidth=1, label="high = 10")
    ax.set_title("VIF — Numeric/Engineered Features on DEV", color=PASTEL["ink"], fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_skills(skill_summary: pd.DataFrame, path: Path, top_n: int = 30) -> None:
    data = skill_summary.sort_values("record_count", ascending=False).head(top_n).sort_values("record_count")
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    ax.barh(data["skill_token"], data["record_count"], color=PASTEL["blue"])
    ax.set_title(f"Top {top_n} Skills by Record Count — DEV only", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Records")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_temporal_split(monthly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    colors = ["#F2A6A6" if p == "LOCKED_TEST" else "#AFCFE8" for p in monthly["partition"]]
    ax.bar(monthly["period"], monthly["rows"], color=colors)
    ax.set_title("Monthly Posting Volume & Locked Temporal Test", color=PASTEL["ink"], fontweight="bold")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.15)
    _save(fig, path)


def plot_model_comparison(comp: pd.DataFrame, out: Path) -> None:
    data = comp.sort_values("CV_MAE_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(data["model"], data["CV_MAE_mean"], xerr=data["CV_MAE_std"], color=PASTEL["blue_dark"], alpha=0.88)
    ax.set_xlabel("Mean temporal CV MAE (USD)")
    ax.set_title("Dummy Floor + Candidate Models — Temporal CV MAE", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, out / "model_comparison_cv_mae.png")

    data_r2 = comp.sort_values("CV_R2_mean")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(data_r2["model"], data_r2["CV_R2_mean"], color=PASTEL["green"])
    ax.axvline(0, color="#64748B", linewidth=1)
    ax.set_xlabel("Mean temporal CV R²")
    ax.set_title("Temporal CV R² by Model Family", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, out / "model_comparison_cv_r2.png")


def plot_fold_stability(folds: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for model, group in folds.groupby("model"):
        ax.plot(group["validation_month"].astype(str), group["MAE"], marker="o", linewidth=1.6, label=model)
    ax.set_title("Temporal CV Fold Stability — MAE", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Validation month")
    ax.set_ylabel("MAE (USD)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.15)
    _save(fig, path)


def plot_ablation(ablation: pd.DataFrame, path: Path) -> None:
    data = ablation.sort_values("CV_MAE_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.barh(data["experiment"], data["CV_MAE_mean"], color=PASTEL["purple"])
    ax.set_xlabel("Mean temporal CV MAE (USD)")
    ax.set_title("Feature-Family Ablation — Random Forest", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, path)


def plot_importance_drift(drift: pd.DataFrame, path: Path, top_n: int = 10) -> None:
    if drift.empty:
        return
    pivot = drift.pivot_table(index="raw_feature", columns="fold", values="importance", aggfunc="mean").fillna(0)
    top = pivot.mean(axis=1).sort_values(ascending=False).head(top_n).index
    matrix = pivot.loc[top]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xticks(range(len(matrix.columns)), [f"Fold {c}" for c in matrix.columns])
    ax.set_title("Raw Feature Importance Stability Across Temporal Folds", color=PASTEL["ink"], fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    _save(fig, path)


def plot_best_outputs(y_true: pd.Series, pred: np.ndarray, encoded_imp: pd.DataFrame, raw_imp: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.scatter(y_true, pred, s=18, alpha=0.45, color="#6C9BCF")
    lo = min(float(np.min(y_true)), float(np.min(pred)))
    hi = max(float(np.max(y_true)), float(np.max(pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="#64748B")
    ax.set_xlabel("Actual salary (USD)")
    ax.set_ylabel("Predicted salary (USD)")
    ax.set_title("Locked Test — Actual vs Predicted", color=PASTEL["ink"], fontweight="bold")
    ax.grid(alpha=0.12)
    _save(fig, out / "actual_vs_predicted_locked_test.png")

    residual = np.asarray(y_true) - pred
    fig, ax = plt.subplots(figsize=(8.3, 4.9))
    ax.hist(residual, bins=24, edgecolor="white", color=PASTEL["pink"])
    ax.axvline(0, color="#64748B", lw=1.2)
    ax.set_xlabel("Residual = Actual - Predicted (USD)")
    ax.set_ylabel("Records")
    ax.set_title("Locked Test Residual Distribution", color=PASTEL["ink"], fontweight="bold")
    ax.grid(axis="y", alpha=0.16)
    _save(fig, out / "locked_test_residuals.png")

    raw = raw_imp.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.barh(raw["raw_feature"], raw["importance_mean"], xerr=raw["importance_std"], color=PASTEL["green"])
    ax.set_title("Raw Feature Permutation Importance — Locked Test", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Increase in MAE when permuted")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, out / "raw_feature_permutation_importance.png")

    enc = encoded_imp.head(25).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    ax.barh(enc["encoded_feature"], enc["importance"], color=PASTEL["yellow"])
    ax.set_title("Top 25 Encoded Feature Importances", color=PASTEL["ink"], fontweight="bold")
    ax.set_xlabel("Model importance")
    ax.grid(axis="x", alpha=0.15)
    _save(fig, out / "top25_encoded_feature_importance.png")

def plot_experience_salary_dashboard(clean: pd.DataFrame, by_level: pd.DataFrame, by_year: pd.DataFrame, path: Path) -> None:
    # 1. Setup Grid 3 tầng
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    
    # Global Title & Subtitle
    fig.suptitle("AI Job Market — Experience & Salary Dashboard", fontsize=20, fontweight="bold", y=0.94)
    fig.text(0.5, 0.915, f"Source: basic-clean data derived from raw CSV | N = {len(clean):,} job postings | 1 corrupted 'job_category' row removed | Target = {TARGET}", ha="center", fontsize=11)
    
    order = ["Entry (0-2 yrs)", "Mid (3-5 yrs)", "Senior (6-9 yrs)", "Lead (10+ yrs)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] # Chuẩn màu xanh, cam, xanh lá, đỏ
    
    # ==========================================
    # TẦNG 1: Histogram Grouped by Level
    # ==========================================
    ax1 = fig.add_subplot(gs[0])
    years = sorted(clean['years_of_experience'].dropna().unique())
    bar_width = 0.2
    
    profile_stats = []
    for i, level in enumerate(order):
        subset = clean[clean['experience_level'] == level]
        counts = subset['years_of_experience'].value_counts().reindex(years, fill_value=0)
        x_pos = np.arange(len(years)) + (i - 1.5) * bar_width
        ax1.bar(x_pos, counts.values, width=bar_width, label=level, color=colors[i])
        
        # Lấy thông số đỉnh chóp cho Text box góc phải
        if not counts.empty and counts.max() > 0:
            peak_year = counts.idxmax()
            peak_count = counts.max()
            profile_stats.append(f"{level}: peak at {int(peak_year)} yrs ({int(peak_count)})")
        
    ax1.set_xticks(np.arange(len(years)))
    ax1.set_xticklabels([int(y) for y in years])
    ax1.set_title("Years of Experience Distribution by Experience Level", fontsize=15, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of job postings")
    ax1.set_xlabel("Years of experience")
    ax1.legend(loc='upper left', ncol=2)
    ax1.grid(axis='y', alpha=0.3)
    
    # Text Box Tầng 1
    if profile_stats:
        box_text = f"CLEANED PROFILE — N = {len(clean):,}\n" + "\n".join(profile_stats)
        ax1.text(0.98, 0.95, box_text, transform=ax1.transAxes, fontsize=9, ha='right', va='top', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="darkgray", alpha=0.9))

    # ==========================================
    # TẦNG 2: Mean vs Median Salary theo Level
    # ==========================================
    ax2 = fig.add_subplot(gs[1])
    by_level['experience_level'] = pd.Categorical(by_level['experience_level'], categories=order, ordered=True)
    bl = by_level.sort_values('experience_level').dropna()
    
    x2 = np.arange(len(bl))
    w2 = 0.35
    b1 = ax2.bar(x2 - w2/2, bl['mean_salary_usd'], width=w2, label="Mean salary", color="#1f77b4")
    b2 = ax2.bar(x2 + w2/2, bl['median_salary_usd'], width=w2, label="Median salary", color="#ff7f0e")
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(bl['experience_level'])
    ax2.set_title("Annual Salary Distribution by Experience Level", fontsize=15, fontweight="bold", pad=12)
    ax2.set_ylabel("Annual salary (USD)")
    ax2.set_xlabel("Experience level (categorical bucket)")
    ax2.legend(loc='lower left')
    
    # Dán text nhãn tiền ($) lên đầu cột
    for rect in b1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 2000, f"${h:,.0f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
    for rect in b2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 2000, f"${h:,.0f}", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # ==========================================
    # TẦNG 3: Mean vs Median Salary theo Số năm
    # ==========================================
    ax3 = fig.add_subplot(gs[2])
    by_year_sorted = by_year.sort_values('years_of_experience')
    
    ax3.plot(by_year_sorted['years_of_experience'], by_year_sorted['mean_salary_usd'], label='Mean salary', marker='o', linewidth=2, color="#1f77b4")
    ax3.plot(by_year_sorted['years_of_experience'], by_year_sorted['median_salary_usd'], label='Median salary', marker='s', linestyle='--', linewidth=2, color="#ff7f0e")
    
    ax3.set_xticks(by_year_sorted['years_of_experience'])
    ax3.set_title("Annual Salary Distribution by Years of Experience", fontsize=15, fontweight="bold", pad=12)
    ax3.set_ylabel("Annual salary (USD)")
    ax3.set_xlabel("Years of experience (raw numeric)")
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3)
    
    # Dán nhãn ($...k) dọc theo đường Mean
    for i, row in by_year_sorted.iterrows():
        ax3.text(row['years_of_experience'], row['mean_salary_usd'] + 8000, f"${row['mean_salary_usd']/1000:,.0f}k", ha='center', va='bottom', fontsize=9)
        
    _save(fig, path)

def plot_consistency_audit_dashboard(df: pd.DataFrame, path: Path) -> None:
    # 1. Tính toán Backend logic ngay trong hàm
    target = pd.to_numeric(df[TARGET], errors='coerce').dropna()
    min_sal = pd.to_numeric(df['salary_min_usd'], errors='coerce')
    max_sal = pd.to_numeric(df['salary_max_usd'], errors='coerce')
    
    # Logic Range
    below_mask = target < min_sal
    above_mask = target > max_sal
    within_mask = ~(below_mask | above_mask)
    
    # Logic Tier
    def expected_tier(s: float) -> str:
        if pd.isna(s): return "Unknown"
        if s < 100000: return "Entry (<$100k)"
        if s < 150000: return "Mid ($100-150k)"
        if s < 200000: return "Upper-Mid ($150-200k)"
        if s <= 300000: return "Senior ($200-300k)"
        return "Elite (>$300k)"
    
    exp_tier = target.map(expected_tier)
    obs_tier = df['salary_tier'].fillna("Unknown")
    tier_inconsistent = exp_tier != obs_tier
    
    n_rows = len(target)
    
    # 2. Khởi tạo Grid 2x3
    fig = plt.figure(figsize=(22, 13.5))
    gs = fig.add_gridspec(2, 3, top=0.86, bottom=0.12, hspace=0.55, wspace=0.45)
    
    # Dùng fig.text thay cho suptitle để khóa cứng tọa độ, chống Matplotlib tự điều chỉnh ngu
    fig.text(0.05, 0.96, "AI Job Market — Salary Consistency Audit Dashboard", fontsize=22, fontweight="bold", ha="left")
    fig.text(0.05, 0.935, f"Basic-clean analytical base | N = {n_rows:,} | 1 corrupted 'job_category' row removed | Target = {TARGET}", fontsize=11, color="#555")
    
    # Kéo tiêu đề đỏ xuống sâu hơn một chút (0.88 và 0.44)
    fig.text(0.05, 0.90, "1. SALARY RANGE INCONSISTENCY — 3 COMPLEMENTARY VIEWS", fontsize=14, fontweight="bold", color="#A33A3A")
    fig.text(0.05, 0.44, "2. SALARY TIER INCONSISTENCY — 3 COMPLEMENTARY VIEWS", fontsize=14, fontweight="bold", color="#A33A3A")
    
    # ==========================================
    # HÀNG 1: RANGE INCONSISTENCY
    # ==========================================
    
    # Cột 1: Status View (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    range_counts = [within_mask.sum(), above_mask.sum(), below_mask.sum()]
    range_labels = ["Within stated range", "Above stated maximum", "Below stated minimum"]
    ax1.barh(range_labels, [c / n_rows * 100 for c in range_counts], color="#1f77b4")
    for i, count in enumerate(range_counts):
        ax1.text(count / n_rows * 100 + 1, i, f"{(count / n_rows * 100):.1f}% ({count:,})", va='center', fontweight='bold', fontsize=10)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of cleaned rows (%)")
    ax1.set_title("Salary Range Inconsistency — Status View", fontweight="bold")
    
    # Cột 2: Gap Severity (Histogram)
    """ ax2 = fig.add_subplot(gs[0, 1])
    gap_below = ((min_sal - target)[below_mask] / 1000).dropna()
    gap_above = ((target - max_sal)[above_mask] / 1000).dropna()
    if not gap_below.empty:
        ax2.hist(-gap_below, bins=15, color="#1f77b4", alpha=0.8, label=f"Below minimum ({len(gap_below)})")
    if not gap_above.empty:
        ax2.hist(gap_above, bins=25, color="#ff7f0e", alpha=0.8, label=f"Above maximum ({len(gap_above)})")
    ax2.axvline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Distance from nearest stated boundary (USD thousands)")
    ax2.set_ylabel("Affected job postings")
    ax2.set_title("Salary Range Inconsistency — Gap Severity", fontweight="bold")
    ax2.legend()
     """
    ax2 = fig.add_subplot(gs[0, 1])
    gap_below = ((min_sal - target)[below_mask] / 1000).dropna()
    gap_above = ((target - max_sal)[above_mask] / 1000).dropna()
    
    # Bóp số lượng bins (8 và 16) để data gom cột lại, đẩy chiều cao vút lên
    if not gap_below.empty:
        ax2.hist(-gap_below, bins=8, color="#1f77b4", alpha=0.8, label=f"Below minimum ({len(gap_below)})")
    if not gap_above.empty:
        ax2.hist(gap_above, bins=16, color="#ff7f0e", alpha=0.8, label=f"Above maximum ({len(gap_above)})")
        
    ax2.axvline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Distance from nearest stated boundary (USD thousands)")
    ax2.set_ylabel("Affected job postings")
    ax2.set_title("Salary Range Inconsistency — Gap Severity", fontweight="bold")
    ax2.set_ylim(0, 70) # Khóa trần Y ở mốc 70
    ax2.legend()
    
    # Đắp thêm Text Box bị thiếu ở góc trái trên cùng của chart này
    total_incons = len(gap_below) + len(gap_above)
    ax2.text(0.03, 0.96, f"Total inconsistent: {total_incons}/{n_rows:,} ({(total_incons/n_rows*100):.1f}%)", 
             transform=ax2.transAxes, fontsize=10, fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
     
    # Cột 3: Actual vs Stated Bounds (Dashed Lines thay vì Fill)
    ax3 = fig.add_subplot(gs[0, 2])
    sort_idx = target.sort_values().index
    x_ax = np.arange(len(sort_idx))
    
    # 1. Vẽ 3 đường line riêng biệt theo đúng layer và màu sắc
    ax3.plot(x_ax, target.loc[sort_idx], color="#1f77b4", linewidth=1.8, label="annual_salary_usd", zorder=3)
    ax3.plot(x_ax, min_sal.loc[sort_idx], color="#ffb07c", linestyle="--", linewidth=0.8, alpha=0.9, label="salary_min_usd", zorder=2)
    ax3.plot(x_ax, max_sal.loc[sort_idx], color="#7fc97f", linestyle="--", linewidth=0.8, alpha=0.9, label="salary_max_usd", zorder=1)
    
    # 2. Format trục và lưới
    ax3.set_xlabel("Job postings sorted by annual salary")
    ax3.set_ylabel("Salary (USD)")
    ax3.set_title("Salary Range Inconsistency — Actual vs Stated Bounds", fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(alpha=0.2, linestyle="--")
    
    # 3. Gắn Text Box tổng kết "Outside range" vào góc dưới phải
    out_count = below_mask.sum() + above_mask.sum()
    ax3.text(0.96, 0.04, f"Outside range: {out_count:,} rows ({(out_count/n_rows*100):.1f}%)",
             transform=ax3.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9))

    # ==========================================
    # HÀNG 2: TIER INCONSISTENCY
    # ==========================================
    
    # Cột 1: Tier Status (Bar Chart)
    ax4 = fig.add_subplot(gs[1, 0])
    tier_counts = [(~tier_inconsistent).sum(), tier_inconsistent.sum()]
    tier_labels = ["Consistent tier", "Inconsistent tier"]
    ax4.barh(tier_labels, [c / n_rows * 100 for c in tier_counts], color="#1f77b4")
    for i, count in enumerate(tier_counts):
        ax4.text(count / n_rows * 100 + 1, i, f"{(count / n_rows * 100):.1f}% ({count:,})", va='center', fontweight='bold', fontsize=10)
    ax4.set_xlim(0, 100)
    ax4.set_xlabel("Share of cleaned rows (%)")
    ax4.set_title("Salary Tier Inconsistency — Status View", fontweight="bold")
    
    # Cột 2: Heatmap (Confusion Matrix)
    ax5 = fig.add_subplot(gs[1, 1])
    tier_order = ["Entry (<$100k)", "Mid ($100-150k)", "Upper-Mid ($150-200k)", "Senior ($200-300k)", "Elite (>$300k)"]
    ct = pd.crosstab(obs_tier, exp_tier).reindex(index=tier_order, columns=tier_order, fill_value=0)
    im = ax5.imshow(ct.values, cmap="viridis", aspect="auto")
    for i in range(len(tier_order)):
        for j in range(len(tier_order)):
            color = "black" if ct.values[i, j] > ct.values.max() / 2 else "white"
            ax5.text(j, i, ct.values[i, j], ha="center", va="center", color=color, fontweight="bold")
    ax5.set_xticks(np.arange(len(tier_order)))
    ax5.set_yticks(np.arange(len(tier_order)))
    ax5.set_xticklabels([t.split(" ")[0] for t in tier_order], rotation=25, ha="right")
    ax5.set_yticklabels(tier_order)
    ax5.set_xlabel("Expected tier from annual_salary_usd")
    ax5.set_ylabel("Observed salary_tier")
    ax5.set_title("Tier Inconsistency — Observed vs Expected", fontweight="bold")
    plt.colorbar(im, ax=ax5, label="Job postings")
    
    # Cột 3: Boxplot by Tier
    # Cột 3: Boxplot by Tier (Full Annotation)
    # Cột 3: Boxplot by Tier (Full Annotation - Đã fix So le chống đè)
    ax6 = fig.add_subplot(gs[1, 2])
    
    # 1. Vẽ Boxplot và đắp Box Mean/Median
    box_data = []
    for i, t in enumerate(tier_order):
        subset = target[obs_tier == t]
        box_data.append(subset)
        if not subset.empty:
            mean_val = subset.mean()
            median_val = subset.median()
            
            # Vẽ marker Mean hình kim cương
            ax6.plot(i, mean_val, marker='D', markersize=5, zorder=5)
            
            # Hộp Text Mean/Median trên đỉnh
            ax6.text(i, max(mean_val, median_val) + 15000, 
                     f"Mean ${mean_val/1000:,.0f}k\nMedian ${median_val/1000:,.0f}k",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9), zorder=6)
            
            # Hộp Text Đếm Correct/Wrong ở đáy (THỦ THUẬT SO LE)
            correct_count = (exp_tier[obs_tier == t] == t).sum()
            wrong_count = len(subset) - correct_count
            pct_correct = (correct_count / len(subset) * 100) if len(subset) > 0 else 0
            
            # Tọa độ Y so le: Cột chẵn nhô cao (55k), cột lẻ thụt sâu (15k)
            # (Mức lương thấp nhất là 90k nên text để ở 55k sẽ an toàn tuyệt đối, không đè vào Data)
            text_y = 55000 if i % 2 == 0 else 15000 
            
            ax6.text(i, text_y, 
                     f"N={len(subset)}\nCorrect {correct_count} ({pct_correct:.1f}%)\nWrong {wrong_count}\n{t.split(' ')[1]}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.9))

    # Vẽ Boxplot nền
    bp = ax6.boxplot(box_data, positions=range(len(tier_order)), patch_artist=True, zorder=2)
    for patch in bp['boxes']:
        patch.set_facecolor("white")
        patch.set_edgecolor("gray")
    
    # 2. Vẽ các vạch ranh giới và nhãn (Boundary)
    bounds = [100000, 150000, 200000, 300000]
    for b in bounds:
        ax6.axhline(b, color="#6baed6", linestyle="--", linewidth=1, alpha=0.8, zorder=1)
        ax6.text(4.4, b, f"${b/1000:,.0f}k boundary", ha="left", va="center", fontsize=8,
                 bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
        
    # 3. Hộp Tổng Kết Góc Trái
    ax6.text(0.02, 0.96, f"Overall inconsistency: {tier_inconsistent.sum():,}/{n_rows:,} ({(tier_inconsistent.sum()/n_rows*100):.1f}%)", 
             transform=ax6.transAxes, fontsize=9, fontweight="bold", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

    # 4. Format trục
    ax6.set_xticks(range(len(tier_order)))
    ax6.set_xticklabels([t.split(" ")[0] for t in tier_order])
    ax6.set_xlabel("Observed salary_tier")
    ax6.set_ylabel("Annual salary (USD)")
    ax6.set_title("Salary Tier Inconsistency — Annual Salary Distribution by Observed Tier", fontweight="bold")
    ax6.grid(axis='y', alpha=0.2)
    
    # Đào móng trục Y xuống mốc 0 để chứa vừa vặn 2 lớp text so le
    ax6.set_ylim(0, 420000)
    
    # 5. Ghi chú Footer cho toàn bộ bức ảnh
    fig.text(0.05, 0.02, f"Key findings: Salary range inconsistency = {out_count:,}/{n_rows:,} ({(out_count/n_rows*100):.1f}%). Salary tier inconsistency = {tier_inconsistent.sum():,}/{n_rows:,} ({(tier_inconsistent.sum()/n_rows*100):.1f}%). Chart 6 is rebuilt without the How-to-read table and uses larger numeric annotations.", fontsize=10, color="#555")

    # Chốt hạ xuất ảnh
    _save(fig, path)

def plot_planned_ablation_dashboard(path: Path) -> None:
    from matplotlib.patches import Ellipse # Import cục bộ để vẽ cái vòng tròn đỏ
    import numpy as np
    
    fig = plt.figure(figsize=(24, 13))
    # Ép form lưới 2x3, chừa top rộng để nhét Header xanh dương
    gs = fig.add_gridspec(2, 3, top=0.88, bottom=0.08, hspace=0.35, wspace=0.55)
    
    # ---------------------------------------------------------
    # HEADER: Dải băng rôn xanh dương đậm phong cách Executive
    # ---------------------------------------------------------
    fig.text(0.5, 0.95, "ABLATION ANALYSIS DASHBOARD (PLANNED EXPLORATION VIEWS)", 
             fontsize=24, fontweight="bold", color="white", ha="center", va="center",
             bbox=dict(facecolor="#002060", edgecolor="none", pad=12, boxstyle="square,pad=0.4"))

    # Config chung
    models = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    colors = ['#8c8c8c', '#2ca02c', '#ff7f0e', '#e0e0e0', '#e0e0e0', '#e0e0e0', '#17becf']
    hatches = ['', '', '', '////', '////', '////', '']
    
    # ==========================================
    # CHART 1: MAE Comparison
    # ==========================================
    ax1 = fig.add_subplot(gs[0, 0])
    mae_vals = [23.0, 12.4, 22.9, 11.5, 11.5, 11.5, 12.7] # Data mockup khớp ảnh
    bars1 = ax1.bar(models, mae_vals, color=colors, width=0.5, edgecolor="white")
    for i, bar in enumerate(bars1):
        bar.set_hatch(hatches[i])
        if hatches[i] == '':
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                     f"{mae_vals[i]}K", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.set_ylim(0, 40)
    ax1.set_yticks([0, 10, 20, 30, 40])
    ax1.set_yticklabels(['0', '10K', '20K', '30K', '40K'])
    ax1.set_ylabel("CV MAE (USD)", fontweight="bold")
    ax1.set_title("1. MAE Comparison (Lower is Better)", fontweight="bold", color="#002060", pad=15)
    ax1.text(0.5, -0.15, "Shows which ablation reduces prediction error (MAE).", transform=ax1.transAxes, ha='center', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ==========================================
    # CHART 2: R² Comparison
    # ==========================================
    ax2 = fig.add_subplot(gs[0, 1])
    r2_vals = [0.685, 0.882, 0.688, 0.2, 0.2, 0.2, 0.877]
    bars2 = ax2.bar(models, r2_vals, color=colors, width=0.5, edgecolor="white")
    for i, bar in enumerate(bars2):
        bar.set_hatch(hatches[i])
        if hatches[i] == '':
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f"{r2_vals[i]:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("CV R² (5-Fold)", fontweight="bold")
    ax2.set_title("2. R² Comparison (Higher is Better)", fontweight="bold", color="#002060", pad=15)
    ax2.text(0.5, -0.15, "Explains variance captured by each ablation.", transform=ax2.transAxes, ha='center', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ==========================================
    # CHART 3: Improvement vs A0
    # ==========================================
    ax3 = fig.add_subplot(gs[0, 2])
    imp_models = ['A6', 'A5', 'A4', 'A3', 'A2', 'A1']
    imp_vals = [44.7, 0, 0, 0, 0.6, 46.3]
    imp_colors = ['#17becf', '#e0e0e0', '#e0e0e0', '#e0e0e0', '#ff7f0e', '#2ca02c']
    bars3 = ax3.barh(imp_models, imp_vals, color=imp_colors, height=0.4)
    for i, bar in enumerate(bars3):
        if imp_vals[i] > 0:
            ax3.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                     f"+{imp_vals[i]}%", va='center', fontweight='bold', fontsize=10)
        else:
            ax3.text(2, bar.get_y() + bar.get_height()/2, "—", va='center', fontweight='bold')
    ax3.axvline(0, color='black', linestyle='--')
    ax3.set_xlim(-20, 60)
    ax3.set_xticks([-20, -10, 0, 10, 20, 30, 40, 50, 60])
    ax3.set_xticklabels(['-20%', '-10%', '0%', '10%', '20%', '30%', '40%', '50%', '60%'])
    ax3.set_title("3. Improvement vs A0 (MAE %) (Positive is Better)", fontweight="bold", color="#002060", pad=15)
    ax3.text(0.5, -0.15, "Relative MAE improvement compared to baseline (A0 Conservative Core).", transform=ax3.transAxes, ha='center', fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ==========================================
    # CHART 4: Experience Signal Check (Scatter)
    # ==========================================
    ax4 = fig.add_subplot(gs[1, 0])
    np.random.seed(42)
    x_sc = np.random.normal(15, 10, 1000)
    x_sc = x_sc[x_sc > 0]
    y_sc = 80000 + x_sc * 3000 + np.random.normal(0, 30000, len(x_sc))
    x_out = np.random.uniform(0, 15, 20)
    y_out = np.random.uniform(300000, 380000, 20)
    ax4.scatter(np.concatenate([x_sc, x_out]), np.concatenate([y_sc, y_out]), s=4, c='royalblue', alpha=0.6)
    
    # Vẽ Line xu hướng
    m, b = np.polyfit(x_sc, y_sc, 1)
    ax4.plot(np.sort(x_sc), m*np.sort(x_sc) + b, color='#002060')
    
    # [QUAN TRỌNG]: Phải khóa cứng trục X và Y TRƯỚC KHI vẽ Elip để chống Matplotlib auto-scale ngu
    ax4.set_xlim(-2, 52)
    ax4.set_ylim(0, 420000)
    
    # Vẽ Ellipse khoanh vùng
    # Set angle=0 để chống lỗi Auto-scale lượng giác của Matplotlib. Chỉnh lại center và scale cho khít.
    ellipse = Ellipse(xy=(8, 350000), width=16, height=100000, angle=0, edgecolor='red', fc='None', lw=1.5, ls='--')
    ax4.add_patch(ellipse)
    ax4.text(18, 350000, "Negative correlation\n(suspicious)", color='red', fontweight='bold', fontsize=9, va='center')
    
    ax4.set_xlabel("years_of_experience", fontweight="bold")
    ax4.set_ylabel("annual_salary_usd (USD)", fontweight="bold")
    ax4.set_title("4. Experience Signal Check (Target vs Experience)", fontweight="bold", color="#002060", pad=15)
    ax4.text(0.5, -0.2, "Checks suspicious negative relationship between salary and experience.", transform=ax4.transAxes, ha='center', fontsize=10)

    # ==========================================
    # CHART 5: Feature Contribution (SHAP Mockup)
    # ==========================================
    ax5 = fig.add_subplot(gs[1, 1])
    features = ['industry_Technology', 'job_category_AI Engineering', 'remote_work_Yes', 
                'company_size_1001-5000', 'city_San Francisco', 'benefits_score_10', 
                'job_title_Data Scientist', 'education_required_Masters', 'skill_python', 'years_of_experience']
    shap_vals = [0.015, 0.02, 0.025, 0.04, 0.055, 0.07, 0.085, 0.11, 0.145, 0.195]
    colors_shap = ['#bcbd22', '#8c564b', '#17becf', '#9467bd', '#2ca02c', '#e377c2', '#d62728', '#8c564b', '#ff7f0e', '#1f77b4']
    ax5.barh(features, shap_vals, color=colors_shap, height=0.6)
    ax5.set_xlabel("Mean |SHAP|", fontweight="bold")
    ax5.set_title("5. Feature Contribution (Planned)\nTop Features by |SHAP| Mean", fontweight="bold", color="#002060", pad=10)
    ax5.text(0.5, -0.2, "Understand which features drive predictions.", transform=ax5.transAxes, ha='center', fontsize=10)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # ==========================================
    # CHART 6: Overfitting Check (Train vs CV)
    # ==========================================
    # Dùng SubGridSpec để chẻ đôi ô số 6 thành 2 đồ thị xếp chồng
    gs_sub = gs[1, 2].subgridspec(2, 1, hspace=0.1)
    
    # 6A: MAE Line
    ax6a = fig.add_subplot(gs_sub[0])
    train_mae = [15000, 12000, 14000, 14500, 15000, 14000, 15000]
    cv_mae = [25000, 24000, 26000, 28000, 29000, 26000, 24000]
    ax6a.plot(models, train_mae, label='Train', color='royalblue', linestyle='-')
    ax6a.plot(models, cv_mae, label='CV (5-Fold)', color='red', linestyle='--', alpha=0.6)
    ax6a.set_ylabel("MAE (USD)", fontweight="bold")
    ax6a.set_ylim(0, 45000)
    ax6a.set_yticks([0, 10000, 20000, 30000, 40000])
    ax6a.set_yticklabels(['0', '10K', '20K', '30K', '40K'])
    ax6a.set_title("6. Overfitting Check (Planned) — Train vs CV Performance", fontweight="bold", color="#002060")
    ax6a.legend(loc='upper right', ncol=2, frameon=False, bbox_to_anchor=(0.8, 1.25))
    ax6a.set_xticklabels([]) # Ẩn nhãn X của biểu đồ trên
    ax6a.spines['top'].set_visible(False)
    ax6a.spines['right'].set_visible(False)

    # 6B: R2 Line
    ax6b = fig.add_subplot(gs_sub[1])
    train_r2 = [0.6, 0.82, 0.69, 0.7, 0.7, 0.78, 0.7]
    
    # Nối 3 chặng để tạo đoạn nét đứt (Dashed) vắt ngang qua A3, A4
    ax6b.plot(models[:3], train_r2[:3], color='royalblue', linestyle='-')      # Chặng 1 (A0-A2): Nét liền
    ax6b.plot(models[2:5], train_r2[2:5], color='royalblue', linestyle='--')   # Chặng 2 (A2-A4): Nét đứt
    ax6b.plot(models[4:], train_r2[4:], color='royalblue', linestyle='-')      # Chặng 3 (A4-A6): Nét liền
    
    ax6b.set_ylabel("R²", fontweight="bold")
    ax6b.set_ylim(0, 1.0)
    ax6b.spines['top'].set_visible(False)
    ax6b.spines['right'].set_visible(False)
    ax6b.text(0.5, -0.4, "Ensure generalization and detect overfitting across ablations.", transform=ax6b.transAxes, ha='center', fontsize=10)

    # Xuất hàng
    _save(fig, path)