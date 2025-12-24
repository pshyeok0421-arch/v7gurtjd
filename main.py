# main.py
from __future__ import annotations

import io
import math
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------
# Page / Font (Korean safe)
# ----------------------------
st.set_page_config(
    page_title="🌱 극지식물 최적 EC 농도 연구",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"

# ----------------------------
# Constants / Metadata
# ----------------------------
SCHOOLS = ["전체", "송도고", "하늘고", "아라고", "동산고"]

SCHOOL_META = pd.DataFrame(
    [
        {"학교명": "송도고", "EC 목표": 1.0, "개체수(시트)": 29, "색상": "#1f77b4"},
        {"학교명": "하늘고", "EC 목표": 2.0, "개체수(시트)": 45, "색상": "#2ca02c"},  # 최적
        {"학교명": "아라고", "EC 목표": 4.0, "개체수(시트)": 106, "색상": "#ff7f0e"},
        {"학교명": "동산고", "EC 목표": 8.0, "개체수(시트)": 58, "색상": "#d62728"},
    ]
)

OPTIMAL_EC = 2.0
DATA_DIR = Path(__file__).resolve().parent / "data"

ENV_REQUIRED_COLS = ["time", "temperature", "humidity", "ph", "ec"]

# Growth columns (Korean)
GROWTH_REQUIRED_COLS = ["개체번호", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]


# ----------------------------
# Unicode-safe file matching
# ----------------------------
def _norm_all(s: str) -> Tuple[str, str]:
    """Return (NFC, NFD) normalized versions."""
    return (unicodedata.normalize("NFC", s), unicodedata.normalize("NFD", s))


def _path_name_norms(p: Path) -> Tuple[str, str]:
    return _norm_all(p.name)


def _equals_unicode(a: str, b: str) -> bool:
    a_nfc, a_nfd = _norm_all(a)
    b_nfc, b_nfd = _norm_all(b)
    return (a_nfc == b_nfc) or (a_nfd == b_nfd) or (a_nfc == b_nfd) or (a_nfd == b_nfc)


def find_file_by_exact_name(data_dir: Path, target_name: str) -> Path | None:
    """Iterate files and match target_name using NFC/NFD bidirectional comparison."""
    for p in data_dir.iterdir():
        if p.is_file():
            if _equals_unicode(p.name, target_name):
                return p
    return None


def find_first_xlsx(data_dir: Path) -> Path | None:
    """Iterate files and return first .xlsx (unicode-safe, no glob)."""
    for p in data_dir.iterdir():
        if p.is_file():
            nfc, nfd = _path_name_norms(p)
            if nfc.lower().endswith(".xlsx") or nfd.lower().endswith(".xlsx"):
                return p
    return None


# ----------------------------
# Cached data loaders
# ----------------------------
@st.cache_data(show_spinner=False)
def load_environment_csvs(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all school environment CSVs into dict by school name (without hardcoding path joins)."""
    wanted = {
        "송도고": "송도고_환경데이터.csv",
        "하늘고": "하늘고_환경데이터.csv",
        "아라고": "아라고_환경데이터.csv",
        "동산고": "동산고_환경데이터.csv",
    }

    out: Dict[str, pd.DataFrame] = {}

    for school, fname in wanted.items():
        p = find_file_by_exact_name(data_dir, fname)
        if p is None:
            continue
        df = pd.read_csv(p)
        # Ensure columns exist
        missing = [c for c in ENV_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"[{school}] 환경 데이터에 필요한 컬럼이 없습니다: {missing}")

        # Parse time
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        # numeric coercion
        for col in ["temperature", "humidity", "ph", "ec"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["temperature", "humidity", "ph", "ec"])

        df["학교"] = school
        out[school] = df

    return out


@st.cache_data(show_spinner=False)
def load_growth_xlsx(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load growth results xlsx (all sheets) without hardcoding sheet names."""
    xlsx_path = find_file_by_exact_name(data_dir, "4개교_생육결과데이터.xlsx")
    if xlsx_path is None:
        # fallback: any xlsx
        xlsx_path = find_first_xlsx(data_dir)
    if xlsx_path is None:
        return {}

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheets = xls.sheet_names

    out: Dict[str, pd.DataFrame] = {}

    # map sheet -> school by unicode-normalized containment, not hardcoded exact list
    school_candidates = ["동산고", "송도고", "아라고", "하늘고"]

    for sh in sheets:
        sh_nfc, sh_nfd = _norm_all(sh)
        matched_school = None
        for s in school_candidates:
            s_nfc, s_nfd = _norm_all(s)
            if (s_nfc in sh_nfc) or (s_nfd in sh_nfd) or (s_nfc in sh_nfd) or (s_nfd in sh_nfc):
                matched_school = s
                break

        df = pd.read_excel(xls, sheet_name=sh, engine="openpyxl")
        if df is None or df.empty:
            continue

        # If it doesn't match any known school name, still keep it under sheet name
        key = matched_school if matched_school is not None else sh

        # Validate required columns (allow minor whitespace)
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        missing = [c for c in GROWTH_REQUIRED_COLS if c not in df.columns]
        if missing:
            # keep but warn later, don’t crash whole app
            df["_missing_cols"] = ", ".join(missing)

        # numeric columns
        for col in ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["학교"] = key
        out[key] = df

    return out


# ----------------------------
# Helpers
# ----------------------------
def get_target_ec(school: str) -> float | None:
    row = SCHOOL_META[SCHOOL_META["학교명"] == school]
    if row.empty:
        return None
    return float(row.iloc[0]["EC 목표"])


def get_color(school: str) -> str:
    row = SCHOOL_META[SCHOOL_META["학교명"] == school]
    if row.empty:
        return "#888888"
    return str(row.iloc[0]["색상"])


def combine_env(env_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not env_dict:
        return pd.DataFrame(columns=ENV_REQUIRED_COLS + ["학교"])
    return pd.concat(env_dict.values(), ignore_index=True)


def combine_growth(growth_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not growth_dict:
        return pd.DataFrame(columns=GROWTH_REQUIRED_COLS + ["학교"])
    return pd.concat(growth_dict.values(), ignore_index=True)


def safe_mean(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def format_num(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.{digits}f}"


def linear_fit_line(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[float, float] | None:
    """
    Fit y = a*x + b using least squares without numpy/statsmodels.
    Return (a, b) or None.
    """
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 2:
        return None

    x = tmp["x"]
    y = tmp["y"]
    x_mean = x.mean()
    y_mean = y.mean()

    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return None

    a = (((x - x_mean) * (y - y_mean)).sum()) / denom
    b = y_mean - a * x_mean
    return float(a), float(b)


def scatter_with_fit(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, hover_data=["학교"], title=title)
    fig.update_layout(font=dict(family=PLOTLY_FONT))

    fit = linear_fit_line(df, x, y)
    if fit is not None:
        a, b = fit
        x_min = float(pd.to_numeric(df[x], errors="coerce").min())
        x_max = float(pd.to_numeric(df[x], errors="coerce").max())
        xs = [x_min, x_max]
        ys = [a * x_min + b, a * x_max + b]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="회귀선"))

    # correlation (Pearson)
    corr = pd.to_numeric(df[x], errors="coerce").corr(pd.to_numeric(df[y], errors="coerce"))
    if corr is not None and not (isinstance(corr, float) and math.isnan(corr)):
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            text=f"상관계수 r = {corr:.3f}",
            borderpad=6,
        )
    return fig


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return buffer.getvalue()


def multi_sheet_xlsx_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in dfs.items():
            sheet = str(name)[:31]  # Excel sheet limit
            df.to_excel(writer, index=False, sheet_name=sheet)
    buffer.seek(0)
    return buffer.getvalue()


# ----------------------------
# Sidebar
# ----------------------------
st.title("🌱 극지식물 최적 EC 농도 연구")

selected_school = st.sidebar.selectbox("학교 선택", SCHOOLS, index=0)

# ----------------------------
# Load data with safety
# ----------------------------
with st.spinner("데이터 로딩 중..."):
    try:
        env_dict = load_environment_csvs(DATA_DIR)
    except Exception as e:
        st.error(f"환경 데이터 로딩 오류: {e}")
        env_dict = {}

    try:
        growth_dict = load_growth_xlsx(DATA_DIR)
    except Exception as e:
        st.error(f"생육 데이터 로딩 오류: {e}")
        growth_dict = {}

env_all = combine_env(env_dict)
growth_all = combine_growth(growth_dict)

if env_all.empty:
    st.error("환경 데이터가 없습니다. data/ 폴더에 CSV 4개가 있는지 확인하세요.")
if growth_all.empty:
    st.error("생육 결과 데이터가 없습니다. data/ 폴더에 XLSX가 있는지 확인하세요.")

# Filtered
if selected_school != "전체":
    env_view = env_all[env_all["학교"] == selected_school].copy()
    growth_view = growth_all[growth_all["학교"] == selected_school].copy()
else:
    env_view = env_all.copy()
    growth_view = growth_all.copy()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# =========================================================
# Tab 1: Overview
# =========================================================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
극지식물은 온도·습도·pH 같은 환경 조건뿐 아니라 **양액의 EC(전기전도도)** 변화에 따라 생육이 크게 달라질 수 있습니다.  
본 연구는 **학교별로 서로 다른 EC 조건(1.0 / 2.0 / 4.0 / 8.0)**에서 재배한 극지식물의 생육 결과를 비교하여  
**최적 EC 농도(생중량 중심)**를 도출하는 것을 목표로 합니다.
"""
    )

    st.subheader("학교별 EC 조건")
    meta_show = SCHOOL_META.copy()
    meta_show["EC 목표"] = meta_show["EC 목표"].map(lambda v: f"{v:.1f}")
    st.dataframe(meta_show, use_container_width=True, hide_index=True)

    # KPI cards
    total_n = None
    if not growth_all.empty and "개체번호" in growth_all.columns:
        total_n = int(growth_all["개체번호"].dropna().nunique())
    else:
        total_n = int(len(growth_all)) if not growth_all.empty else 0

    avg_temp = safe_mean(env_view["temperature"]) if not env_view.empty else None
    avg_hum = safe_mean(env_view["humidity"]) if not env_view.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_n:,}")
    c2.metric("평균 온도(°C)", format_num(avg_temp, 2))
    c3.metric("평균 습도(%)", format_num(avg_hum, 2))
    c4.metric("최적 EC(가정)", f"{OPTIMAL_EC:.1f} (하늘고)")

# =========================================================
# Tab 2: Environment
# =========================================================
with tab2:
    st.subheader("학교별 환경 평균 비교")

    if env_all.empty:
        st.stop()

    # Summary by school (all schools)
    env_summary = (
        env_all.groupby("학교", as_index=False)
        .agg(
            평균온도=("temperature", "mean"),
            평균습도=("humidity", "mean"),
            평균pH=("ph", "mean"),
            실측EC평균=("ec", "mean"),
        )
        .copy()
    )
    # Add target EC
    env_summary["목표EC"] = env_summary["학교"].map(lambda s: get_target_ec(s))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC"),
    )

    # Top-left: temp
    fig.add_trace(
        go.Bar(
            x=env_summary["학교"],
            y=env_summary["평균온도"],
            name="평균 온도",
        ),
        row=1,
        col=1,
    )
    # Top-right: humidity
    fig.add_trace(
        go.Bar(
            x=env_summary["학교"],
            y=env_summary["평균습도"],
            name="평균 습도",
        ),
        row=1,
        col=2,
    )
    # Bottom-left: pH
    fig.add_trace(
        go.Bar(
            x=env_summary["학교"],
            y=env_summary["평균pH"],
            name="평균 pH",
        ),
        row=2,
        col=1,
    )
    # Bottom-right: target vs measured EC (dual bar)
    fig.add_trace(
        go.Bar(
            x=env_summary["학교"],
            y=env_summary["목표EC"],
            name="목표 EC",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=env_summary["학교"],
            y=env_summary["실측EC평균"],
            name="실측 EC 평균",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=650,
        barmode="group",
        font=dict(family=PLOTLY_FONT),
        margin=dict(l=40, r=20, t=80, b=40),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("선택한 학교 시계열")

    if env_view.empty:
        st.info("선택한 조건에 해당하는 환경 데이터가 없습니다.")
    else:
        # Temperature
        fig_t = px.line(env_view.sort_values("time"), x="time", y="temperature", title="온도 변화")
        fig_t.update_layout(font=dict(family=PLOTLY_FONT))
        st.plotly_chart(fig_t, use_container_width=True)

        # Humidity
        fig_h = px.line(env_view.sort_values("time"), x="time", y="humidity", title="습도 변화")
        fig_h.update_layout(font=dict(family=PLOTLY_FONT))
        st.plotly_chart(fig_h, use_container_width=True)

        # EC with target line
        fig_ec = px.line(env_view.sort_values("time"), x="time", y="ec", title="EC 변화")
        target = None
        if selected_school != "전체":
            target = get_target_ec(selected_school)
        fig_ec.update_layout(font=dict(family=PLOTLY_FONT))
        if target is not None:
            fig_ec.add_hline(y=target, line_dash="dash", annotation_text=f"목표 EC {target:.1f}", annotation_position="top left")
        st.plotly_chart(fig_ec, use_container_width=True)

    with st.expander("환경 데이터 원본 테이블 / CSV 다운로드"):
        st.dataframe(env_view, use_container_width=True, hide_index=True)
        st.download_button(
            label="CSV 다운로드",
            data=df_to_csv_bytes(env_view),
            file_name="환경데이터_필터링.csv" if selected_school == "전체" else f"환경데이터_{selected_school}.csv",
            mime="text/csv",
        )

# =========================================================
# Tab 3: Growth results
# =========================================================
with tab3:
    st.subheader("🥇 핵심 결과: EC별 평균 생중량")

    if growth_all.empty:
        st.stop()

    # Attach target EC by school
    growth_all_ec = growth_all.copy()
    growth_all_ec["EC 목표"] = growth_all_ec["학교"].map(lambda s: get_target_ec(s))

    # If user filtered to one school, keep analysis consistent (still show EC buckets on available data)
    growth_use = growth_all_ec if selected_school == "전체" else growth_all_ec[growth_all_ec["학교"] == selected_school].copy()

    if "생중량(g)" not in growth_use.columns or growth_use["생중량(g)"].dropna().empty:
        st.error("생중량(g) 데이터가 없거나 비어 있습니다.")
        st.stop()

    # Group by EC target (1,2,4,8)
    ec_group = (
        growth_use.dropna(subset=["EC 목표"])
        .groupby("EC 목표", as_index=False)
        .agg(
            평균생중량=("생중량(g)", "mean"),
            평균잎수=("잎 수(장)", "mean"),
            평균지상부길이=("지상부 길이(mm)", "mean"),
            개체수=("개체번호", "count"),
        )
        .sort_values("EC 목표")
    )

    if ec_group.empty:
        st.error("EC 목표 값으로 묶을 수 있는 생육 데이터가 없습니다.")
        st.stop()

    # Highlight max mean weight
    max_row = ec_group.loc[ec_group["평균생중량"].idxmax()]
    best_ec = float(max_row["EC 목표"])
    best_weight = float(max_row["평균생중량"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("최대 평균 생중량", f"{best_weight:.3f} g")
    k2.metric("해당 EC", f"{best_ec:.1f}")
    # emphasize optimal assumption
    k3.metric("가정 최적 EC(하늘고)", f"{OPTIMAL_EC:.1f}")
    k4.metric("선택 범위", "전체" if selected_school == "전체" else selected_school)

    st.subheader("EC별 생육 비교 (2x2)")

    fig2 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 생중량(g) ⭐", "평균 잎 수(장)", "평균 지상부 길이(mm)", "개체수"),
    )

    # Weight
    fig2.add_trace(go.Bar(x=ec_group["EC 목표"], y=ec_group["평균생중량"], name="평균 생중량"), row=1, col=1)
    # Leaves
    fig2.add_trace(go.Bar(x=ec_group["EC 목표"], y=ec_group["평균잎수"], name="평균 잎 수"), row=1, col=2)
    # Shoot length
    fig2.add_trace(go.Bar(x=ec_group["EC 목표"], y=ec_group["평균지상부길이"], name="평균 지상부 길이"), row=2, col=1)
    # Count
    fig2.add_trace(go.Bar(x=ec_group["EC 목표"], y=ec_group["개체수"], name="개체수"), row=2, col=2)

    fig2.update_layout(
        height=650,
        barmode="group",
        font=dict(family=PLOTLY_FONT),
        margin=dict(l=40, r=20, t=80, b=40),
        showlegend=False,
    )

    # Mark EC 2.0 as optimal (vertical line on first subplot feel via annotation)
    fig2.add_vline(x=OPTIMAL_EC, line_dash="dash", annotation_text="최적(하늘고 EC 2.0)", annotation_position="top")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("학교별 생중량 분포")

    if selected_school == "전체":
        fig_box = px.box(
            growth_all_ec.dropna(subset=["생중량(g)"]),
            x="학교",
            y="생중량(g)",
            points="outliers",
            title="학교별 생중량 분포 (Box Plot)",
            color="학교",
            color_discrete_map={r["학교명"]: r["색상"] for _, r in SCHOOL_META.iterrows()},
        )
    else:
        fig_box = px.box(
            growth_use.dropna(subset=["생중량(g)"]),
            x="학교",
            y="생중량(g)",
            points="outliers",
            title="선택한 학교 생중량 분포 (Box Plot)",
            color="학교",
            color_discrete_map={selected_school: get_color(selected_school)},
        )
    fig_box.update_layout(font=dict(family=PLOTLY_FONT))
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("상관관계 분석 (회귀선은 statsmodels 없이 직접 계산)")

    # Scatter 1: Leaves vs Weight
    s1 = growth_use.dropna(subset=["잎 수(장)", "생중량(g)"])
    if len(s1) >= 2:
        fig_s1 = scatter_with_fit(s1, "잎 수(장)", "생중량(g)", "잎 수 vs 생중량")
        st.plotly_chart(fig_s1, use_container_width=True)
    else:
        st.info("잎 수 vs 생중량 산점도를 그리기 위한 데이터가 부족합니다.")

    # Scatter 2: Shoot length vs Weight
    s2 = growth_use.dropna(subset=["지상부 길이(mm)", "생중량(g)"])
    if len(s2) >= 2:
        fig_s2 = scatter_with_fit(s2, "지상부 길이(mm)", "생중량(g)", "지상부 길이 vs 생중량")
        st.plotly_chart(fig_s2, use_container_width=True)
    else:
        st.info("지상부 길이 vs 생중량 산점도를 그리기 위한 데이터가 부족합니다.")

    with st.expander("학교별 생육 데이터 원본 / XLSX 다운로드"):
        st.dataframe(growth_view, use_container_width=True, hide_index=True)

        # Download: if 전체 -> multi sheet, else single sheet
        if selected_school == "전체":
            # Keep only the 4 known schools if present; otherwise include all loaded keys
            dfs = {}
            for school in ["동산고", "송도고", "아라고", "하늘고"]:
                if school in growth_dict:
                    dfs[school] = growth_dict[school]
            if not dfs:
                dfs = growth_dict
            xlsx_bytes = multi_sheet_xlsx_bytes(dfs)
            st.download_button(
                label="XLSX 다운로드 (시트 포함)",
                data=xlsx_bytes,
                file_name="생육결과_전체.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            xlsx_bytes = df_to_xlsx_bytes(growth_view)
            st.download_button(
                label="XLSX 다운로드 (선택 학교)",
                data=xlsx_bytes,
                file_name=f"생육결과_{selected_school}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
 
