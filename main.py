import io
import unicodedata
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(page_title="극지식물 최적 EC 농도 연구", layout="wide")

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

PLOTLY_FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, sans-serif"

SCHOOLS = ["동산고", "송도고", "아라고", "하늘고"]
EC_TARGETS = {"송도고": 1.0, "하늘고": 2.0, "아라고": 4.0, "동산고": 8.0}
SCHOOL_COLORS = {"동산고": "#636EFA", "송도고": "#EF553B", "아라고": "#00CC96", "하늘고": "#AB63FA"}

DATA_DIR = Path(__file__).resolve().parent / "data"


# ----------------------------
# 유틸: 한글 NFC/NFD 안전 비교
# ----------------------------
def _norm(s: str, form: str) -> str:
    return unicodedata.normalize(form, s)


def _match_name(file_name: str, keyword: str) -> bool:
    """
    파일명과 키워드를 NFC/NFD 양방향으로 비교해서 포함 여부를 판단
    (확장자 .csv.csv / .xlsx.xlsx 같은 경우도 이름 포함 비교라서 안전)
    """
    a_nfc = _norm(file_name, "NFC")
    a_nfd = _norm(file_name, "NFD")
    k_nfc = _norm(keyword, "NFC")
    k_nfd = _norm(keyword, "NFD")

    return (k_nfc in a_nfc) or (k_nfd in a_nfd) or (k_nfc in a_nfd) or (k_nfd in a_nfc)


@st.cache_data(show_spinner=False)
def discover_files(data_dir: Path) -> Tuple[Dict[str, Path], Optional[Path]]:
    """
    iterdir()로 data 폴더를 훑고,
    - 환경 CSV: 각 학교명 + '환경데이터' + '.csv' 포함 파일
    - 생육 XLSX: '생육결과데이터' + '.xlsx' 포함 파일
    를 찾아 반환
    """
    env_files: Dict[str, Path] = {}
    growth_xlsx: Optional[Path] = None

    if not data_dir.exists():
        return env_files, growth_xlsx

    for p in data_dir.iterdir():
        if not p.is_file():
            continue

        name = p.name  # 원본 파일명 그대로

        # 환경 CSV 탐색 (확장자 2번이어도 ".csv"가 들어있으면 OK)
        if _match_name(name.lower(), ".csv") and _match_name(name, "환경데이터"):
            for sch in SCHOOLS:
                if _match_name(name, sch) and sch not in env_files:
                    env_files[sch] = p

        # 생육 XLSX 탐색 (".xlsx.xlsx"도 name에 ".xlsx" 포함)
        if _match_name(name.lower(), ".xlsx") and _match_name(name, "생육결과데이터"):
            # 여러 개가 있으면 "가장 먼저 발견된 것" 사용
            if growth_xlsx is None:
                growth_xlsx = p

    return env_files, growth_xlsx


@st.cache_data(show_spinner=False)
def load_env_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 컬럼 표준화
    df.columns = [str(c).strip() for c in df.columns]
    required = {"time", "temperature", "humidity", "ph", "ec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"환경 데이터 컬럼이 부족합니다: {sorted(missing)} / 실제 컬럼: {list(df.columns)}")

    # time 파싱
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    # 수치형 변환
    for c in ["temperature", "humidity", "ph", "ec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_growth_xlsx(path: Path) -> Dict[str, pd.DataFrame]:
    """
    sheet_name=None으로 시트명 하드코딩 없이 전부 로드.
    반환: {학교명: df}
    """
    all_sheets: Dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None, engine="openpyxl")

    # 시트명 -> 학교명 매핑 (NFC/NFD 안전)
    out: Dict[str, pd.DataFrame] = {}
    for sheet, df in all_sheets.items():
        # 컬럼 표준화
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # 학교명 판별
        matched_school = None
        for sch in SCHOOLS:
            if _match_name(str(sheet), sch):
                matched_school = sch
                break

        if matched_school is None:
            # 학교명을 못 찾으면 건너뜀 (다른 설명 시트가 있을 수도 있음)
            continue

        # 기대 컬럼 정리 (없어도 에러는 안 내되, 핵심은 숫자 변환)
        numeric_cols = ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        out[matched_school] = df

    return out


def filter_by_school(selected: str, env_map: Dict[str, pd.DataFrame], growth_map: Dict[str, pd.DataFrame]):
    if selected == "전체":
        return env_map, growth_map
    env_one = {selected: env_map[selected]} if selected in env_map else {}
    growth_one = {selected: growth_map[selected]} if selected in growth_map else {}
    return env_one, growth_one


def safe_download_csv(df: pd.DataFrame, file_name: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="CSV 다운로드",
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
    )


def safe_download_xlsx(df: pd.DataFrame, file_name: str):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    st.download_button(
        label="XLSX 다운로드",
        data=buffer,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def plotly_apply_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(font=dict(family=PLOTLY_FONT_FAMILY))
    return fig


def mean_or_nan(series: pd.Series) -> float:
    try:
        return float(series.mean())
    except Exception:
        return float("nan")


# ----------------------------
# 앱 시작
# ----------------------------
st.title("🌱 극지식물 최적 EC 농도 연구")

# 사이드바
st.sidebar.header("설정")
selected_school = st.sidebar.selectbox("학교 선택", ["전체"] + SCHOOLS, index=0)

# 파일 탐색 + 로딩
with st.spinner("데이터 파일을 탐색하고 불러오는 중..."):
    env_paths, growth_xlsx = discover_files(DATA_DIR)

    env_data: Dict[str, pd.DataFrame] = {}
    env_errors = []
    for sch, p in env_paths.items():
        try:
            env_data[sch] = load_env_csv(p)
        except Exception as e:
            env_errors.append(f"- {sch}: {p.name} 로딩 실패 → {e}")

    growth_data: Dict[str, pd.DataFrame] = {}
    growth_error = None
    if growth_xlsx is not None:
        try:
            growth_data = load_growth_xlsx(growth_xlsx)
        except Exception as e:
            growth_error = str(e)

# 에러 안내 (명확하게)
if not DATA_DIR.exists():
    st.error(f"`data/` 폴더를 찾을 수 없습니다: {DATA_DIR}")
    st.stop()

if env_errors:
    st.error("환경 데이터(CSV) 로딩 중 오류가 발생했습니다:\n" + "\n".join(env_errors))

if growth_xlsx is None:
    st.error("생육 결과 XLSX 파일을 찾지 못했습니다. 파일명에 '생육결과데이터' 와 '.xlsx' 가 포함되어야 합니다.")
elif growth_error:
    st.error(f"생육 결과 XLSX 로딩 중 오류: {growth_error}")

if not env_data:
    st.error("환경 데이터(CSV)를 하나도 불러오지 못했습니다. data/ 폴더에 파일이 있는지 확인하세요.")
    st.stop()

if not growth_data:
    st.error("생육 결과 데이터(XLSX)를 하나도 불러오지 못했습니다. 시트명에 학교명이 포함되어 있는지 확인하세요.")
    st.stop()

# 선택 필터
env_view, growth_view = filter_by_school(selected_school, env_data, growth_data)

tabs = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ----------------------------
# Tab 1: 실험 개요
# ----------------------------
with tabs[0]:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
극지식물의 생육에 영향을 주는 핵심 요인 중 하나는 **양액의 EC(전기전도도)** 입니다.  
본 연구는 4개 학교가 서로 다른 EC 조건(1.0 / 2.0 / 4.0 / 8.0)에서 재배한 생육 결과를 비교하여  
**최적 EC 농도(생중량 중심)** 를 도출하는 것을 목표로 합니다.
"""
    )

    # 학교별 EC 조건 표 (개체수는 생육 데이터에서 계산)
    rows = []
    for sch in SCHOOLS:
        n = int(growth_data.get(sch, pd.DataFrame()).shape[0])
        rows.append(
            {
                "학교명": sch,
                "EC 목표": EC_TARGETS.get(sch, None),
                "개체수": n,
                "색상": SCHOOL_COLORS.get(sch, ""),
            }
        )
    cond_df = pd.DataFrame(rows)

    st.subheader("학교별 EC 조건")
    st.dataframe(cond_df, use_container_width=True)

    # 주요 지표 카드 4개
    total_n = int(sum(df.shape[0] for df in growth_data.values()))
    all_env_concat = pd.concat(env_data.values(), ignore_index=True)
    avg_temp = mean_or_nan(all_env_concat["temperature"])
    avg_hum = mean_or_nan(all_env_concat["humidity"])

    # 최적 EC (생중량 평균 최대)
    growth_long = []
    for sch, df in growth_data.items():
        if "생중량(g)" in df.columns:
            tmp = df[["생중량(g)"]].copy()
            tmp["학교"] = sch
            tmp["EC"] = EC_TARGETS.get(sch, None)
            growth_long.append(tmp)
    growth_long_df = pd.concat(growth_long, ignore_index=True) if growth_long else pd.DataFrame()

    best_ec = None
    if not growth_long_df.empty:
        ec_means = (
            growth_long_df.dropna(subset=["EC", "생중량(g)"])
            .groupby("EC", as_index=False)["생중량(g)"]
            .mean()
            .sort_values("생중량(g)", ascending=False)
        )
        if not ec_means.empty:
            best_ec = float(ec_means.iloc[0]["EC"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_n}개")
    c2.metric("평균 온도", f"{avg_temp:.2f} ℃" if pd.notna(avg_temp) else "N/A")
    c3.metric("평균 습도", f"{avg_hum:.2f} %" if pd.notna(avg_hum) else "N/A")
    c4.metric("최적 EC(생중량 기준)", f"{best_ec:.1f}" if best_ec is not None else "N/A")

    st.info("참고: 본 대시보드는 **생중량 평균이 가장 높은 EC를 ‘최적’** 으로 표시합니다.")


# ----------------------------
# Tab 2: 환경 데이터
# ----------------------------
with tabs[1]:
    st.subheader("학교별 환경 평균 비교")

    # 평균 요약
    env_summary_rows = []
    for sch, df in env_view.items():
        env_summary_rows.append(
            {
                "학교": sch,
                "평균 온도": mean_or_nan(df["temperature"]),
                "평균 습도": mean_or_nan(df["humidity"]),
                "평균 pH": mean_or_nan(df["ph"]),
                "실측 EC 평균": mean_or_nan(df["ec"]),
                "EC 목표": EC_TARGETS.get(sch, None),
            }
        )
    env_summary = pd.DataFrame(env_summary_rows)

    # 2x2 서브플롯
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC(평균)"),
    )

    # (1,1) 평균 온도
    fig.add_trace(
        go.Bar(
            x=env_summary["학교"],
            y=env_summary["평균 온도"],
            name="평균 온도",
        ),
        row=1,
        col=1,
    )

    # (1,2) 평균 습도
    fig.add_trace(
        go.Bar(
            x=env_summary["학교"],
            y=env_summary["평균 습도"],
            name="평균 습도",
        ),
        row=1,
        col=2,
    )

    # (2,1) 평균 pH
    fig.add_trace(
        go.Bar(
            x=env_summary["학교"],
            y=env_summary["평균 pH"],
            name="평균 pH",
        ),
        row=2,
        col=1,
    )

    # (2,2) 목표 vs 실측 EC(평균) 이중 막대
    fig.add_trace(
        go.Bar(x=env_summary["학교"], y=env_summary["EC 목표"], name="EC 목표"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=env_summary["학교"], y=env_summary["실측 EC 평균"], name="실측 EC 평균"),
        row=2,
        col=2,
    )

    fig.update_layout(barmode="group", height=650, margin=dict(t=70))
    fig = plotly_apply_font(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열")

    if selected_school == "전체":
        st.info("사이드바에서 특정 학교를 선택하면 해당 학교의 시계열 그래프가 표시됩니다.")
    else:
        if selected_school not in env_data:
            st.error(f"{selected_school}의 환경 데이터를 찾지 못했습니다.")
        else:
            df = env_data[selected_school].copy()
            target_ec = EC_TARGETS.get(selected_school, None)

            # 온도
            fig_t = px.line(df, x="time", y="temperature", title="온도 변화")
            fig_t = plotly_apply_font(fig_t)
            st.plotly_chart(fig_t, use_container_width=True)

            # 습도
            fig_h = px.line(df, x="time", y="humidity", title="습도 변화")
            fig_h = plotly_apply_font(fig_h)
            st.plotly_chart(fig_h, use_container_width=True)

            # EC + 목표선
            fig_ec = px.line(df, x="time", y="ec", title="EC 변화 (목표 EC 수평선 포함)")
            if target_ec is not None:
                fig_ec.add_hline(y=target_ec, line_dash="dash", annotation_text=f"목표 EC={target_ec}")
            fig_ec = plotly_apply_font(fig_ec)
            st.plotly_chart(fig_ec, use_container_width=True)

            with st.expander("환경 데이터 원본 테이블 / 다운로드"):
                st.dataframe(df, use_container_width=True)
                safe_download_csv(df, f"{selected_school}_환경데이터.csv")


# ----------------------------
# Tab 3: 생육 결과
# ----------------------------
with tabs[2]:
    st.subheader("🥇 핵심 결과: EC별 평균 생중량")

    # long-form 생성
    growth_rows = []
    for sch, df in growth_view.items():
        if df.empty:
            continue
        if "생중량(g)" not in df.columns:
            continue

        tmp = df.copy()
        tmp["학교"] = sch
        tmp["EC"] = EC_TARGETS.get(sch, None)
        growth_rows.append(tmp)

    if not growth_rows:
        st.error("표시할 생육 데이터가 없습니다. '생중량(g)' 컬럼을 확인하세요.")
        st.stop()

    gdf = pd.concat(growth_rows, ignore_index=True)

    # EC별 요약
    ec_summary = (
        gdf.dropna(subset=["EC"])
        .groupby("EC", as_index=False)
        .agg(
            평균_생중량=("생중량(g)", "mean"),
            평균_잎수=("잎 수(장)", "mean") if "잎 수(장)" in gdf.columns else ("생중량(g)", "size"),
            평균_지상부길이=("지상부 길이(mm)", "mean") if "지상부 길이(mm)" in gdf.columns else ("생중량(g)", "size"),
            개체수=("생중량(g)", "count"),
        )
        .sort_values("EC")
    )

    if ec_summary.empty:
        st.error("EC 요약을 만들 수 없습니다. EC 목표 매핑 또는 데이터 값을 확인하세요.")
        st.stop()

    # 최댓값(평균 생중량) 표시
    best_row = ec_summary.sort_values("평균_생중량", ascending=False).iloc[0]
    best_ec = float(best_row["EC"])
    best_weight = float(best_row["평균_생중량"])

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("최적 EC(평균 생중량 최대)", f"{best_ec:.1f}")
    c2.metric("최대 평균 생중량", f"{best_weight:.3f} g")
    # 요구사항: 하늘고(EC 2.0) 최적값 강조
    if abs(best_ec - 2.0) < 1e-9:
        c3.success("⭐ 최적 EC가 **2.0(하늘고)** 로 확인되었습니다!")
    else:
        c3.info("참고: 데이터 기준 최적 EC가 2.0이 아닐 수도 있습니다. (생중량 평균 최대 기준)")

    st.dataframe(ec_summary, use_container_width=True)

    st.divider()
    st.subheader("EC별 생육 비교 (2x2)")

    # 2x2 막대그래프
    fig2 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 생중량(⭐)", "평균 잎 수", "평균 지상부 길이", "개체수 비교"),
    )

    fig2.add_trace(go.Bar(x=ec_summary["EC"], y=ec_summary["평균_생중량"], name="평균 생중량"), row=1, col=1)
    fig2.add_trace(go.Bar(x=ec_summary["EC"], y=ec_summary["평균_잎수"], name="평균 잎 수"), row=1, col=2)
    fig2.add_trace(go.Bar(x=ec_summary["EC"], y=ec_summary["평균_지상부길이"], name="평균 지상부 길이"), row=2, col=1)
    fig2.add_trace(go.Bar(x=ec_summary["EC"], y=ec_summary["개체수"], name="개체수"), row=2, col=2)

    fig2.update_layout(height=650, margin=dict(t=70))
    fig2 = plotly_apply_font(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("학교별 생중량 분포")

    if selected_school == "전체":
        # 전체면 학교별
        fig_box = px.box(
            gdf.dropna(subset=["생중량(g)"]),
            x="학교",
            y="생중량(g)",
            color="학교",
            color_discrete_map=SCHOOL_COLORS,
            title="학교별 생중량 분포 (Box Plot)",
        )
    else:
        # 특정 학교면 개체번호 기준 분포
        fig_box = px.box(
            gdf.dropna(subset=["생중량(g)"]),
            y="생중량(g)",
            title=f"{selected_school} 생중량 분포 (Box Plot)",
        )

    fig_box = plotly_apply_font(fig_box)
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    st.subheader("상관관계 분석 (산점도 2개)")

    # statsmodels 없이: trendline 제거 + 간단 회귀선(선택) 추가는 생략(안전 우선)
    colA, colB = st.columns(2)

    with colA:
        if "잎 수(장)" in gdf.columns:
            scatter1 = gdf.dropna(subset=["잎 수(장)", "생중량(g)"]).copy()
            fig_sc1 = px.scatter(
                scatter1,
                x="잎 수(장)",
                y="생중량(g)",
                color="학교" if selected_school == "전체" else None,
                color_discrete_map=SCHOOL_COLORS,
                title="잎 수 vs 생중량",
            )
            fig_sc1 = plotly_apply_font(fig_sc1)
            st.plotly_chart(fig_sc1, use_container_width=True)
        else:
            st.warning("컬럼 '잎 수(장)' 이 없어 산점도를 표시할 수 없습니다.")

    with colB:
        if "지상부 길이(mm)" in gdf.columns:
            scatter2 = gdf.dropna(subset=["지상부 길이(mm)", "생중량(g)"]).copy()
            fig_sc2 = px.scatter(
                scatter2,
                x="지상부 길이(mm)",
                y="생중량(g)",
                color="학교" if selected_school == "전체" else None,
                color_discrete_map=SCHOOL_COLORS,
                title="지상부 길이 vs 생중량",
            )
            fig_sc2 = plotly_apply_font(fig_sc2)
            st.plotly_chart(fig_sc2, use_container_width=True)
        else:
            st.warning("컬럼 '지상부 길이(mm)' 이 없어 산점도를 표시할 수 없습니다.")

    with st.expander("학교별 생육 데이터 원본 / 다운로드"):
        # 보여주기용(선택 기준)
        if selected_school == "전체":
            for sch in SCHOOLS:
                if sch in growth_data:
                    st.markdown(f"**{sch}**")
                    st.dataframe(growth_data[sch], use_container_width=True)
        else:
            if selected_school in growth_data:
                st.dataframe(growth_data[selected_school], use_container_width=True)

        # 다운로드: 현재 필터된 gdf를 xlsx로 제공
        safe_download_xlsx(gdf, f"{selected_school}_생육데이터.xlsx" if selected_school != "전체" else "전체_생육데이터.xlsx")
