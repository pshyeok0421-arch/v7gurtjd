import io
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="🌱 극지식물 최적 EC 농도 연구", layout="wide")

# Korean font (Streamlit UI)
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

PLOTLY_FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"

st.title("🌱 극지식물 최적 EC 농도 연구")

DATA_DIR = Path(__file__).parent / "data"

SCHOOLS: List[str] = ["송도고", "하늘고", "아라고", "동산고"]
EC_TARGET: Dict[str, float] = {"송도고": 1.0, "하늘고": 2.0, "아라고": 4.0, "동산고": 8.0}
SCHOOL_COLOR: Dict[str, str] = {
    "송도고": "#3b82f6",
    "하늘고": "#22c55e",
    "아라고": "#f59e0b",
    "동산고": "#ef4444",
}


# ----------------------------
# Utilities: NFC/NFD-safe matching
# ----------------------------
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def norm_equal(a: str, b: str) -> bool:
    """Bidirectional NFC/NFD equality."""
    return (_nfc(a) == _nfc(b)) or (_nfd(a) == _nfd(b))


def contains_norm(haystack: str, needle: str) -> bool:
    """Check if needle is contained in haystack under NFC/NFD."""
    h_nfc, n_nfc = _nfc(haystack), _nfc(needle)
    h_nfd, n_nfd = _nfd(haystack), _nfd(needle)
    return (n_nfc in h_nfc) or (n_nfd in h_nfd)


def find_file_by_predicate(directory: Path, predicate) -> Path | None:
    """Must use iterdir(); no glob; NFC/NFD safe."""
    if not directory.exists():
        return None
    for p in directory.iterdir():
        if p.is_file() and predicate(p):
            return p
    return None


def find_all_files_by_predicate(directory: Path, predicate) -> List[Path]:
    out: List[Path] = []
    if not directory.exists():
        return out
    for p in directory.iterdir():
        if p.is_file() and predicate(p):
            out.append(p)
    return out


# ----------------------------
# Data Loading
# ----------------------------
@st.cache_data(show_spinner=False)
def load_environment_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load 4 CSVs:
    columns: time, temperature, humidity, ph, ec
    School name is inferred from filename (before first underscore), NFC/NFD safe.
    """
    env_files = find_all_files_by_predicate(
        data_dir,
        lambda p: p.suffix.lower() == ".csv" and (contains_norm(p.name, "환경데이터") or contains_norm(p.stem, "환경데이터")),
    )

    env_by_school: Dict[str, pd.DataFrame] = {}

    for fp in env_files:
        # infer school from filename without f-string composing
        # expected: "{학교}_환경데이터.csv"
        stem = fp.stem  # e.g., "송도고_환경데이터"
        school_guess = stem.split("_")[0].strip()

        # map to known schools using NFC/NFD comparison (no hard dependency on exact normalization)
        matched_school = None
        for s in SCHOOLS:
            if norm_equal(school_guess, s) or contains_norm(stem, s):
                matched_school = s
                break
        if matched_school is None:
            # keep it, but under the guessed name
            matched_school = school_guess

        try:
            df = pd.read_csv(fp)
        except Exception:
            # try utf-8-sig fallback
            df = pd.read_csv(fp, encoding="utf-8-sig")

        # normalize column names
        df.columns = [c.strip() for c in df.columns]

        # parse time
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.sort_values("time")

        # ensure numeric
        for col in ["temperature", "humidity", "ph", "ec"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["school"] = matched_school
        env_by_school[matched_school] = df

    return env_by_school


@st.cache_data(show_spinner=False)
def load_growth_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load XLSX with 4 sheets (school sheets).
    IMPORTANT: Sheet names are not hardcoded. We read all sheets, then match to schools via NFC/NFD.
    """
    xlsx_path = find_file_by_predicate(
        data_dir,
        lambda p: p.suffix.lower() in [".xlsx", ".xlsm"]
        and (contains_norm(p.name, "생육결과") or contains_norm(p.stem, "생육결과")),
    )
    if xlsx_path is None:
        # fallback: pick first xlsx if exists
        xlsx_path = find_file_by_predicate(data_dir, lambda p: p.suffix.lower() in [".xlsx", ".xlsm"])

    if xlsx_path is None:
        return {}

    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheets = xl.sheet_names

    growth_by_school: Dict[str, pd.DataFrame] = {}

    for sh in sheets:
        df = pd.read_excel(xlsx_path, sheet_name=sh, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]

        # match sheet name to known schools without hardcoding sheet names
        matched_school = None
        for s in SCHOOLS:
            if norm_equal(sh, s) or contains_norm(sh, s):
                matched_school = s
                break
        if matched_school is None:
            # keep sheet as its own "school" label
            matched_school = sh

        # normalize numeric columns (best-effort; Korean headers expected)
        num_candidates = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
        for col in num_candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["school"] = matched_school
        growth_by_school[matched_school] = df

    return growth_by_school


def concat_dict_dfs(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not d:
        return pd.DataFrame()
    return pd.concat(list(d.values()), ignore_index=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "data") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] if sheet_name else "data")
    buffer.seek(0)
    return buffer.getvalue()


# ----------------------------
# Load data
# ----------------------------
with st.spinner("데이터 로딩 중..."):
    env_by_school = load_environment_data(DATA_DIR)
    growth_by_school = load_growth_data(DATA_DIR)

if not env_by_school:
    st.error("환경 데이터(CSV)를 찾지 못했습니다. data 폴더와 파일명을 확인하세요.")
if not growth_by_school:
    st.error("생육 결과 데이터(XLSX)를 찾지 못했습니다. data 폴더와 파일명을 확인하세요.")

env_all = concat_dict_dfs(env_by_school)
growth_all = concat_dict_dfs(growth_by_school)

# attach EC target to growth
if not growth_all.empty:
    growth_all["EC_목표"] = growth_all["school"].map(EC_TARGET)


# ----------------------------
# Sidebar
# ----------------------------
school_option = st.sidebar.selectbox("학교 선택", ["전체"] + SCHOOLS, index=0)

if school_option == "전체":
    env_filtered = env_all.copy()
    growth_filtered = growth_all.copy()
else:
    env_filtered = env_by_school.get(school_option, pd.DataFrame()).copy()
    growth_filtered = growth_by_school.get(school_option, pd.DataFrame()).copy()


# ----------------------------
# Compute summary metrics
# ----------------------------
def safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if df is None or df.empty or col not in df.columns:
        return None
    val = df[col].mean()
    return None if pd.isna(val) else float(val)


total_individuals = None
if not growth_all.empty:
    total_individuals = int(growth_all.shape[0])

avg_temp = safe_mean(env_filtered if not env_filtered.empty else env_all, "temperature")
avg_hum = safe_mean(env_filtered if not env_filtered.empty else env_all, "humidity")

optimal_ec_value = None
optimal_ec_school = None
optimal_ec_weight = None
if not growth_all.empty and "생중량(g)" in growth_all.columns:
    tmp = growth_all.copy()
    tmp["EC_목표"] = tmp["school"].map(EC_TARGET)
    g = tmp.dropna(subset=["EC_목표", "생중량(g)"]).groupby("EC_목표", as_index=False)["생중량(g)"].mean()
    if not g.empty:
        best_row = g.loc[g["생중량(g)"].idxmax()]
        optimal_ec_value = float(best_row["EC_목표"])
        optimal_ec_weight = float(best_row["생중량(g)"])
        # find which school corresponds (if unique mapping)
        inv = {v: k for k, v in EC_TARGET.items()}
        optimal_ec_school = inv.get(optimal_ec_value)


# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ============================
# Tab 1: Overview
# ============================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
- 서로 다른 **EC 농도 조건(1.0 / 2.0 / 4.0 / 8.0)** 에서 극지식물의 생육 결과를 비교하여
  **최적 EC 농도(생중량 중심)** 를 도출합니다.
- 동시에 학교별 **환경(온도/습도/pH/EC)** 기록을 비교해, 생육 차이가 환경과 어떻게 연결되는지 확인합니다.
        """.strip()
    )

    # EC condition table
    rows = []
    for s in SCHOOLS:
        n = growth_by_school.get(s, pd.DataFrame()).shape[0] if growth_by_school else 0
        rows.append(
            {
                "학교명": s,
                "EC 목표": EC_TARGET.get(s),
                "개체수": n,
                "색상": SCHOOL_COLOR.get(s, ""),
            }
        )
    cond_df = pd.DataFrame(rows)

    st.markdown("#### 학교별 EC 조건")
    st.dataframe(cond_df, use_container_width=True, hide_index=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", "-" if total_individuals is None else f"{total_individuals:,}")
    c2.metric("평균 온도", "-" if avg_temp is None else f"{avg_temp:.2f} °C")
    c3.metric("평균 습도", "-" if avg_hum is None else f"{avg_hum:.2f} %")

    if optimal_ec_value is None:
        c4.metric("최적 EC", "-")
    else:
        label = f"{optimal_ec_value:.1f}"
        if optimal_ec_school:
            label += f" ({optimal_ec_school})"
        c4.metric("최적 EC", label)

    if optimal_ec_value is not None:
        st.info(
            f"생중량 평균 기준 최적 EC는 **{optimal_ec_value:.1f}**"
            + (f" (**{optimal_ec_school}**) " if optimal_ec_school else " ")
            + f"이며, 평균 생중량은 **{optimal_ec_weight:.3f} g** 입니다."
        )
        if abs(optimal_ec_value - 2.0) < 1e-9:
            st.success("✅ 하늘고(EC 2.0)가 최적 조건으로 도출되었습니다.")
        else:
            st.warning("참고: 제공된 설정상 하늘고는 EC 2.0(최적)로 알려져 있으나, 실제 데이터 평균 결과는 위 계산을 따릅니다.")


# ============================
# Tab 2: Environment
# ============================
with tab2:
    st.subheader("학교별 환경 평균 비교")

    if env_all.empty:
        st.error("환경 데이터가 비어 있어 그래프를 표시할 수 없습니다.")
    else:
        # 평균 테이블
        means = []
        for s in SCHOOLS:
            df = env_by_school.get(s, pd.DataFrame())
            if df.empty:
                continue
            means.append(
                {
                    "school": s,
                    "temperature": df["temperature"].mean(),
                    "humidity": df["humidity"].mean(),
                    "ph": df["ph"].mean(),
                    "ec_measured": df["ec"].mean(),
                    "ec_target": EC_TARGET.get(s),
                }
            )
        mean_df = pd.DataFrame(means)

        # 2x2 subplot bars
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 온도(°C)", "평균 습도(%)", "평균 pH", "목표 EC vs 실측 EC"),
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )

        if not mean_df.empty:
            fig.add_trace(
                go.Bar(x=mean_df["school"], y=mean_df["temperature"], name="온도"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(x=mean_df["school"], y=mean_df["humidity"], name="습도"),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Bar(x=mean_df["school"], y=mean_df["ph"], name="pH"),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Bar(x=mean_df["school"], y=mean_df["ec_target"], name="목표 EC"),
                row=2,
                col=2,
            )
            fig.add_trace(
                go.Bar(x=mean_df["school"], y=mean_df["ec_measured"], name="실측 EC"),
                row=2,
                col=2,
            )

        fig.update_layout(
            height=650,
            barmode="group",
            legend_title_text="지표",
            font=dict(family=PLOTLY_FONT_FAMILY),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("선택한 학교 시계열")

    # if "전체" choose a school for time series
    ts_school = school_option
    if school_option == "전체":
        ts_school = st.selectbox("시계열로 볼 학교 선택", SCHOOLS, index=1)

    ts_df = env_by_school.get(ts_school, pd.DataFrame())
    if ts_df.empty:
        st.error("선택한 학교의 환경 데이터가 없습니다.")
    else:
        target_ec = EC_TARGET.get(ts_school)

        # Temperature
        fig_t = px.line(ts_df, x="time", y="temperature", title="온도 변화")
        fig_t.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_t, use_container_width=True)

        # Humidity
        fig_h = px.line(ts_df, x="time", y="humidity", title="습도 변화")
        fig_h.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_h, use_container_width=True)

        # EC with target line
        fig_ec = px.line(ts_df, x="time", y="ec", title="EC 변화 (목표선 포함)")
        if target_ec is not None:
            fig_ec.add_hline(y=target_ec, line_dash="dash", annotation_text=f"목표 EC {target_ec:.1f}")
        fig_ec.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_ec, use_container_width=True)

    with st.expander("환경 데이터 원본 테이블 + 다운로드"):
        show_env = env_filtered if school_option != "전체" else env_all
        if show_env.empty:
            st.error("표시할 환경 데이터가 없습니다.")
        else:
            st.dataframe(show_env, use_container_width=True)
            st.download_button(
                "CSV 다운로드",
                data=to_csv_bytes(show_env),
                file_name="환경데이터_필터링.csv",
                mime="text/csv",
            )


# ============================
# Tab 3: Growth Results
# ============================
with tab3:
    st.subheader("🥇 핵심 결과: EC별 평균 생중량")

    if growth_all.empty:
        st.error("생육 데이터가 비어 있어 그래프를 표시할 수 없습니다.")
    else:
        if "생중량(g)" not in growth_all.columns:
            st.error("생육 데이터에 '생중량(g)' 컬럼이 없습니다.")
        else:
            g_all = growth_all.copy()
            g_all["EC_목표"] = g_all["school"].map(EC_TARGET)
            ec_weight = (
                g_all.dropna(subset=["EC_목표", "생중량(g)"])
                .groupby("EC_목표", as_index=False)["생중량(g)"]
                .mean()
                .sort_values("EC_목표")
            )

            if ec_weight.empty:
                st.error("EC별 평균 생중량을 계산할 수 없습니다(결측치 확인).")
            else:
                best_idx = ec_weight["생중량(g)"].idxmax()
                best_ec = float(ec_weight.loc[best_idx, "EC_목표"])
                best_w = float(ec_weight.loc[best_idx, "생중량(g)"])

                # Card-style highlight
                inv = {v: k for k, v in EC_TARGET.items()}
                best_school = inv.get(best_ec, "")
                st.metric("최대 평균 생중량(EC)", f"{best_w:.3f} g", delta=f"EC {best_ec:.1f} ({best_school})")

                if abs(best_ec - 2.0) < 1e-9:
                    st.success("✅ 하늘고(EC 2.0)가 최적 EC로 도출되었습니다.")
                else:
                    st.warning("참고: 설정상 하늘고(EC 2.0)가 최적이라고 알려져 있으나, 실제 평균 생중량 최대값은 위 계산을 따릅니다.")

    st.divider()

    st.subheader("EC별 생육 비교 (2x2)")

    if growth_all.empty:
        st.stop()

    g = growth_all.copy()
    g["EC_목표"] = g["school"].map(EC_TARGET)

    # aggregates
    agg = g.groupby("EC_목표", as_index=False).agg(
        평균_생중량=("생중량(g)", "mean") if "생중량(g)" in g.columns else ("EC_목표", "size"),
        평균_잎수=("잎 수(장)", "mean") if "잎 수(장)" in g.columns else ("EC_목표", "size"),
        평균_지상부길이=("지상부 길이(mm)", "mean") if "지상부 길이(mm)" in g.columns else ("EC_목표", "size"),
        개체수=("school", "size"),
    ).sort_values("EC_목표")

    # 2x2 charts
    fig2 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 생중량(g) ⭐", "평균 잎 수(장)", "평균 지상부 길이(mm)", "개체수"),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    fig2.add_trace(go.Bar(x=agg["EC_목표"], y=agg["평균_생중량"], name="평균 생중량"), row=1, col=1)
    fig2.add_trace(go.Bar(x=agg["EC_목표"], y=agg["평균_잎수"], name="평균 잎 수"), row=1, col=2)
    fig2.add_trace(go.Bar(x=agg["EC_목표"], y=agg["평균_지상부길이"], name="평균 지상부 길이"), row=2, col=1)
    fig2.add_trace(go.Bar(x=agg["EC_목표"], y=agg["개체수"], name="개체수"), row=2, col=2)

    fig2.update_layout(
        height=700,
        showlegend=False,
        font=dict(family=PLOTLY_FONT_FAMILY),
        margin=dict(l=30, r=30, t=60, b=30),
    )

    # Emphasize EC=2.0 in title annotation (visual hint)
    # (No hard styling required; Streamlit/Plotly default colors okay)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("학교별 생중량 분포")

    if "생중량(g)" in g.columns and not g["생중량(g)"].dropna().empty:
        fig_dist = px.violin(
            g.dropna(subset=["생중량(g)"]),
            x="school",
            y="생중량(g)",
            box=True,
            points="outliers",
            title="학교별 생중량 분포 (Violin + Box)",
        )
        fig_dist.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.error("생중량(g) 데이터가 부족해 분포 그래프를 표시할 수 없습니다.")

    st.divider()

    st.subheader("상관관계 분석")

    c1, c2 = st.columns(2)

    with c1:
        if "잎 수(장)" in g.columns and "생중량(g)" in g.columns:
            scatter1 = g.dropna(subset=["잎 수(장)", "생중량(g)"])
            if scatter1.empty:
                st.error("잎 수 vs 생중량 산점도를 그릴 데이터가 없습니다.")
            else:
                fig_sc1 = px.scatter(
                    scatter1,
                    x="잎 수(장)",
                    y="생중량(g)",
                    color="school",
                    title="잎 수(장) vs 생중량(g)",
                    trendline="ols",
                )
                fig_sc1.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_sc1, use_container_width=True)
        else:
            st.error("'잎 수(장)' 또는 '생중량(g)' 컬럼이 없습니다.")

    with c2:
        if "지상부 길이(mm)" in g.columns and "생중량(g)" in g.columns:
            scatter2 = g.dropna(subset=["지상부 길이(mm)", "생중량(g)"])
            if scatter2.empty:
                st.error("지상부 길이 vs 생중량 산점도를 그릴 데이터가 없습니다.")
            else:
                fig_sc2 = px.scatter(
                    scatter2,
                    x="지상부 길이(mm)",
                    y="생중량(g)",
                    color="school",
                    title="지상부 길이(mm) vs 생중량(g)",
                    trendline="ols",
                )
                fig_sc2.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_sc2, use_container_width=True)
        else:
            st.error("'지상부 길이(mm)' 또는 '생중량(g)' 컬럼이 없습니다.")

    with st.expander("학교별 생육 데이터 원본 + XLSX 다운로드"):
        show_growth = growth_filtered if school_option != "전체" else growth_all
        if show_growth.empty:
            st.error("표시할 생육 데이터가 없습니다.")
        else:
            st.dataframe(show_growth, use_container_width=True)
            xlsx_bytes = to_xlsx_bytes(show_growth, sheet_name="생육데이터")
            st.download_button(
                "XLSX 다운로드",
                data=xlsx_bytes,
                file_name="생육데이터_필터링.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
