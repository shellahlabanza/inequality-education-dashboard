"""
Educational Inequality Dashboard (Group 6: Labanza and Lontoc)
Foundations of Business Analytics | Final Project

Run from this folder with:   streamlit run app.py

Input: data/df_long.csv  (the country-year table saved by Section 7 of the EDA notebook)

Structure
  1. Settings, palette, styling
  2. Data loading (df_long.csv is bundled in the data/ folder)
  3. Predictive model (regression, cross-validated, out-of-fold residuals)
  4. Sidebar filters
  5. Hero + tabs: Overview, Where to act, Priority countries (model), Data notes
"""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, train_test_split

# ---------------------------------------------------------------------------
# 1. SETTINGS, PALETTE, STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Educational Inequality Dashboard",
    page_icon="🎓",
    layout="wide",
)

DATA_FILE = "df_long.csv"
TARGET_YEAR = 2021
HIGH_THRESHOLD = 30  # score at or above this counts as "high inequality"

HDG_ORDER = ["Low", "Medium", "High", "Very High"]
HDG_ALL = HDG_ORDER + ["Unclassified"]
REGION_NAMES = {
    "SSA": "Sub-Saharan Africa",
    "SA": "South Asia",
    "AS": "Arab States",
    "LAC": "Latin America & Caribbean",
    "EAP": "East Asia & Pacific",
    "ECA": "Europe & Central Asia",
    "Unclassified": "Unclassified",
}
REGION_ORDER = list(REGION_NAMES.values())

# One palette for every chart
NAVY = "#16324F"
BLUE = "#0F6E8C"
RED = "#C23B3B"
AMBER = "#E9A23B"
GREY = "#8D99AE"
INK = "#1B2A41"
HDG_COLORS = {
    "Low": RED, "Medium": AMBER, "High": BLUE, "Very High": NAVY, "Unclassified": GREY,
}
# Shared scale: pale = low inequality, deep red = high inequality
SCALE = [[0.0, "#E3EEF4"], [0.45, "#F1C75B"], [1.0, "#B3202F"]]
SCORE_MAX = 50

BASELINE = "Baseline (group mean)"
LINEAR = "Linear Regression"
FOREST = "Random Forest"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&display=swap');
    .block-container {{ padding-top: 2rem; max-width: 1280px; }}
    h1, h2, h3, .hero h1, .card-title {{ font-family: 'Source Serif 4', Georgia, serif; }}
    .hero {{ background: {NAVY}; color: #fff; border-radius: 8px; padding: 28px 34px; margin-bottom: 18px; }}
    .hero .kicker {{ color: #9CC9DA; font-size: 0.95rem; margin-bottom: 6px; }}
    .hero h1 {{ color: #fff; font-size: 2.1rem; line-height: 1.2; margin: 0 0 10px 0; padding: 0; }}
    .hero p {{ color: #DCE6EE; font-size: 1.05rem; margin: 0; max-width: 62rem; }}
    .hero .audience {{ margin-top: 12px; font-size: 0.9rem; color: #9CC9DA; }}
    .kpi {{ background: #fff; border: 1px solid #DDE3EA; border-left: 6px solid {BLUE};
            border-radius: 8px; padding: 16px 18px; height: 100%; }}
    .kpi .label {{ font-size: 0.92rem; color: #51607A; }}
    .kpi .value {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 2.3rem; font-weight: 700;
                   color: {INK}; line-height: 1.15; margin: 4px 0; }}
    .kpi .sub {{ font-size: 0.9rem; color: #51607A; }}
    .kpi .good {{ color: #1E7B4F; font-weight: 600; }}
    .kpi .bad {{ color: {RED}; font-weight: 600; }}
    .qtitle {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.25rem; font-weight: 700;
               color: {INK}; margin: 0; }}
    .qsub {{ color: #51607A; font-size: 0.92rem; margin: 2px 0 6px 0; }}
    .note {{ background: #EAF3F7; border-left: 5px solid {BLUE}; border-radius: 4px;
             padding: 12px 16px; margin-top: 4px; color: {INK}; font-size: 0.97rem; }}
    .note b {{ color: {BLUE}; }}
    [data-testid="stSidebar"] h2 {{ font-family: 'Source Serif 4', Georgia, serif; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------
def find_data():
    here = Path(__file__).resolve().parent
    for path in [here / "data" / DATA_FILE, here / DATA_FILE, Path.cwd() / "data" / DATA_FILE]:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_data(path_str):
    """Read df_long.csv (output of the notebook) and add display labels."""
    df = pd.read_csv(path_str)
    df["HDG"] = df["Human Development Groups"].fillna("Unclassified")
    df["Region_Code"] = df["UNDP Developing Regions"].fillna("Unclassified")
    df["Region"] = df["Region_Code"].map(REGION_NAMES).fillna(df["Region_Code"])
    return df


data_path = find_data()
if data_path is None:
    st.error(
        f"The data file '{DATA_FILE}' was not found. Keep this folder structure: "
        "app.py and a data/ folder that contains df_long.csv, then run the app again."
    )
    st.stop()

long_df = load_data(str(data_path))
needed = {"ISO3", "Country", "Human Development Groups", "UNDP Developing Regions",
          "HDI Rank (2021)", "Year", "Inequality"}
if not needed.issubset(long_df.columns):
    st.error("df_long.csv does not have the expected columns. Use the file saved by the notebook (Section 7).")
    st.stop()

obs = long_df.dropna(subset=["Inequality"])
all_countries = sorted(long_df["Country"].unique())


# ---------------------------------------------------------------------------
# 3. PREDICTIVE MODEL
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def fit_predict(name, x_train, y_train, g_train, x_test, g_test):
    """Fit one model; return (train predictions, test predictions)."""
    if name == BASELINE:
        means = y_train.groupby(g_train).mean()
        overall = y_train.mean()

        def predict(groups):
            return groups.map(means).fillna(overall).to_numpy()

        return predict(g_train), predict(g_test)
    if name == LINEAR:
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1
        )
    model.fit(x_train, y_train)
    return model.predict(x_train), model.predict(x_test)


@st.cache_data(show_spinner="Training and validating the models...")
def run_model(df):
    """
    Target: Inequality in TARGET_YEAR (one row per country).
    Features: HDI rank, Human Development Group (ordinal), UNDP region (one-hot).
    Previous-year inequality is NOT used (r = 0.997 with the current value).
    """
    d = (
        df[df["Year"] == TARGET_YEAR]
        .dropna(subset=["Inequality", "HDI Rank (2021)", "Human Development Groups"])
        .reset_index(drop=True)
    )
    x = pd.DataFrame(
        {
            "HDI_Rank": d["HDI Rank (2021)"].astype(float),
            "HDG_Code": d["Human Development Groups"]
            .map({g: i for i, g in enumerate(HDG_ORDER)})
            .astype(float),
        }
    )
    x = pd.concat([x, pd.get_dummies(d["Region_Code"], prefix="Region", dtype=int)], axis=1)
    y = d["Inequality"].astype(float)
    groups = d["Human Development Groups"].astype(str)
    names = [BASELINE, LINEAR, FOREST]

    # (a) 80/20 hold-out split: train vs test error shows overfitting
    idx_train, idx_test = train_test_split(np.arange(len(d)), test_size=0.2, random_state=42)
    rows = []
    for name in names:
        p_tr, p_te = fit_predict(
            name, x.iloc[idx_train], y.iloc[idx_train], groups.iloc[idx_train],
            x.iloc[idx_test], groups.iloc[idx_test],
        )
        rmse_tr, _, r2_tr = regression_metrics(y.iloc[idx_train], p_tr)
        rmse_te, mae_te, r2_te = regression_metrics(y.iloc[idx_test], p_te)
        rows.append({"Model": name, "Train RMSE": rmse_tr, "Test RMSE": rmse_te,
                     "Test MAE": mae_te, "Train R²": r2_tr, "Test R²": r2_te})
    holdout = pd.DataFrame(rows)

    # (b) Repeated 5-fold cross-validation: stable estimate for a small sample
    splitter = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores = {n: {"rmse": [], "mae": [], "r2": []} for n in names}
    oof_sum = {n: np.zeros(len(d)) for n in names}
    oof_cnt = np.zeros(len(d))
    for tr, te in splitter.split(x):
        oof_cnt[te] += 1
        for name in names:
            _, p_te = fit_predict(
                name, x.iloc[tr], y.iloc[tr], groups.iloc[tr], x.iloc[te], groups.iloc[te]
            )
            rmse, mae, r2 = regression_metrics(y.iloc[te], p_te)
            scores[name]["rmse"].append(rmse)
            scores[name]["mae"].append(mae)
            scores[name]["r2"].append(r2)
            oof_sum[name][te] += p_te
    cv = pd.DataFrame(
        [
            {"Model": n, "CV RMSE": np.mean(scores[n]["rmse"]), "RMSE std": np.std(scores[n]["rmse"]),
             "CV MAE": np.mean(scores[n]["mae"]), "CV R²": np.mean(scores[n]["r2"])}
            for n in names
        ]
    )

    best = cv[cv["Model"] != BASELINE].sort_values("CV RMSE").iloc[0]
    best_name, best_rmse = best["Model"], float(best["CV RMSE"])

    # Out-of-fold predictions: each country is predicted by models that never saw it
    oof = oof_sum[best_name] / oof_cnt
    results = pd.DataFrame(
        {
            "Country": d["Country"],
            "Human Development Group": d["HDG"],
            "UNDP Region": d["Region"],
            "HDI Rank (2021)": d["HDI Rank (2021)"].astype(int),
            "Actual": y.round(2),
            "Predicted": np.round(oof, 2),
        }
    )
    results["Gap (Actual - Predicted)"] = (results["Actual"] - results["Predicted"]).round(2)

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(x, y)
    importance = (
        pd.DataFrame({"Feature": x.columns, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )
    return {"holdout": holdout, "cv": cv, "best_name": best_name, "best_rmse": best_rmse,
            "results": results, "importance": importance, "n_countries": len(d)}


# ---------------------------------------------------------------------------
# Helpers for layout
# ---------------------------------------------------------------------------
def style_fig(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
    )
    fig.update_xaxes(gridcolor="#E6EAF0", zeroline=False)
    fig.update_yaxes(gridcolor="#E6EAF0", zeroline=False)
    return fig


def card_header(title, question):
    st.markdown(
        f'<p class="qtitle">{escape(title)}</p><p class="qsub">Business question: {escape(question)}</p>',
        unsafe_allow_html=True,
    )


def note(text):
    st.markdown(f'<div class="note"><b>What this means.</b> {escape(text)}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", tone=""):
    sub_html = f'<div class="sub"><span class="{tone}">{escape(sub)}</span></div>' if sub else ""
    col.markdown(
        f'<div class="kpi"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def score_colors(values):
    return sample_colorscale(SCALE, [min(max(v / SCORE_MAX, 0), 1) for v in values])


# ---------------------------------------------------------------------------
# 4. SIDEBAR FILTERS
# ---------------------------------------------------------------------------
DEFAULTS = {"f_year": 2021, "f_hdg": HDG_ALL, "f_region": REGION_ORDER, "f_country": []}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_filters():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


st.sidebar.header("Filters")
st.sidebar.caption("Narrow the view to the countries you are responsible for.")
year = st.sidebar.slider("Year", 2010, 2021, key="f_year")
hdg_sel = st.sidebar.multiselect(
    "Human Development Group", HDG_ALL, key="f_hdg",
    help="UNDP groups countries by Human Development Index level.",
)
region_sel = st.sidebar.multiselect(
    "UNDP region", REGION_ORDER, key="f_region",
    help="Unclassified = countries that have no UNDP region in the source data.",
)
country_sel = st.sidebar.multiselect(
    "Country (optional)", all_countries, key="f_country",
    help="Leave empty to include every country.",
)
st.sidebar.button("Reset filters", on_click=reset_filters)


def filter_frame(frame, use_hdg=True, use_region=True, use_country=True):
    mask = pd.Series(True, index=frame.index)
    if use_hdg:
        mask &= frame["HDG"].isin(hdg_sel)
    if use_region:
        mask &= frame["Region"].isin(region_sel)
    if use_country and country_sel:
        mask &= frame["Country"].isin(country_sel)
    return frame[mask]


filtered = filter_frame(obs)
current = filtered[filtered["Year"] == year]


# ---------------------------------------------------------------------------
# 5. HERO
# ---------------------------------------------------------------------------
if current.empty:
    hero_line = "No countries match the current filters. Widen the filters in the sidebar."
else:
    n_high = int((current["Inequality"] >= HIGH_THRESHOLD).sum())
    hero_line = (
        f"In {year}, {n_high} of the {len(current)} countries in view still score "
        f"{HIGH_THRESHOLD} or more on the education inequality index. "
        "Explore where inequality is highest, whether it is falling, and which countries deserve a funding review."
    )
st.markdown(
    f"""
    <div class="hero">
      <div class="kicker">Education equity monitor | UNDP data, 2010 to 2021</div>
      <h1>Where is educational inequality still holding countries back?</h1>
      <p>{escape(hero_line)}</p>
      <div class="audience">Built for an education program manager at an international development
      agency who decides which countries receive education-equity funding.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_over, tab_act, tab_model, tab_notes = st.tabs(
    ["Overview", "Where to act", "Priority countries (model)", "Data notes"]
)


# ---------------------------------------------------------------------------
# TAB 1: OVERVIEW (KPIs, map, trend, trend by group)
# ---------------------------------------------------------------------------
def render_overview():
    if current.empty:
        st.warning("No countries match the current filters for this year. Widen the filters.")
        return

    avg_now = current["Inequality"].mean()
    base_rows = filtered[filtered["Year"] == 2010]["Inequality"]
    high_count = int((current["Inequality"] >= HIGH_THRESHOLD).sum())
    gap_year = filter_frame(obs, use_hdg=False, use_country=False)
    gap_year = gap_year[gap_year["Year"] == year]
    low_mean = gap_year.loc[gap_year["HDG"] == "Low", "Inequality"].mean()
    vh_mean = gap_year.loc[gap_year["HDG"] == "Very High", "Inequality"].mean()

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"Average inequality, {year}", f"{avg_now:.1f}", "Higher score = more unequal")
    if len(base_rows) > 0 and year != 2010:
        change = avg_now - base_rows.mean()
        pct = change / base_rows.mean() * 100
        kpi(c2, "Change since 2010", f"{change:+.1f} pts",
            f"{pct:+.1f}% vs 2010 average", "good" if change < 0 else "bad")
    else:
        kpi(c2, "Change since 2010", "n/a", "Pick a later year")
    kpi(c3, f"Countries scoring {HIGH_THRESHOLD}+", f"{high_count}",
        f"of {len(current)} countries with data", "bad" if high_count else "good")
    if pd.notna(low_mean) and pd.notna(vh_mean):
        kpi(c4, "Gap: Low vs Very High development", f"{low_mean - vh_mean:.1f} pts",
            "Difference in group averages")
    else:
        kpi(c4, "Gap: Low vs Very High development", "n/a", "Needs both groups in view")

    st.write("")

    # ---- Visual 1: world map -------------------------------------------------
    with st.container(border=True):
        card_header(f"Where is inequality highest in {year}?",
                    "Which parts of the world need equity programs most, and where is data missing?")
        fig = px.choropleth(
            current, locations="ISO3", color="Inequality", hover_name="Country",
            hover_data={"ISO3": False, "HDG": True, "Region": True, "Inequality": ":.1f"},
            color_continuous_scale=SCALE, range_color=(0, SCORE_MAX),
        )
        fig.update_geos(showframe=False, showcoastlines=False, landcolor="#E9ECEF",
                        bgcolor="rgba(0,0,0,0)", projection_type="natural earth",
                        showcountries=True, countrycolor="white")
        fig.update_layout(coloraxis_colorbar=dict(title="Score", thickness=12, len=0.7),
                          height=430, margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        by_region = current.groupby("Region")["Inequality"].mean().sort_values(ascending=False)
        top_region = by_region.index[0]
        note(
            f"{high_count} of {len(current)} countries score {HIGH_THRESHOLD} or more, and the "
            f"highest regional average is {top_region} ({by_region.iloc[0]:.1f}). Grey countries "
            f"have no value in {year}. Use the map to see where equity programs are most needed "
            "and where the data itself is missing."
        )

    # ---- Visuals 2 and 3: trends ----------------------------------------------
    trend = (filtered.groupby("Year")["Inequality"]
             .agg(Average="mean", Countries="count").reset_index())
    by_group = filtered.groupby(["Year", "HDG"])["Inequality"].mean().reset_index()

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            card_header("Is inequality falling over time?",
                        "Is the average moving in the right direction?")
            fig = go.Figure(go.Scatter(
                x=trend["Year"], y=trend["Average"], mode="lines+markers",
                line=dict(color=BLUE, width=3), marker=dict(size=8),
                customdata=trend["Countries"],
                hovertemplate="Year %{x}<br>Average %{y:.2f}<br>%{customdata} countries<extra></extra>",
            ))
            fig.add_vline(x=year, line_dash="dot", line_color=GREY)
            fig.update_xaxes(dtick=1, title="")
            fig.update_yaxes(title="Average score")
            st.plotly_chart(style_fig(fig, 360), width="stretch")
            if len(trend) > 1:
                first, last = trend.iloc[0], trend.iloc[-1]
                note(
                    f"The average moved from {first['Average']:.1f} in {int(first['Year'])} to "
                    f"{last['Average']:.1f} in {int(last['Year'])}, while countries with data went "
                    f"from {int(first['Countries'])} to {int(last['Countries'])}. Treat this as a "
                    "direction, not proof that programs worked, because part of the change comes "
                    "from countries entering the data."
                )

    with right:
        with st.container(border=True):
            card_header("Is the gap between development groups closing?",
                        "Are the least developed countries catching up?")
            fig = go.Figure()
            for group in HDG_ALL:
                g = by_group[by_group["HDG"] == group]
                if g.empty:
                    continue
                labels = [""] * (len(g) - 1) + [group]
                fig.add_trace(go.Scatter(
                    x=g["Year"], y=g["Inequality"], mode="lines+markers+text", name=group,
                    line=dict(color=HDG_COLORS[group], width=3), marker=dict(size=6),
                    text=labels, textposition="middle right", showlegend=False,
                    hovertemplate=f"{group}<br>Year %{{x}}<br>Average %{{y:.1f}}<extra></extra>",
                ))
            fig.update_xaxes(dtick=1, title="", range=[2009.6, 2023.2])
            fig.update_yaxes(title="Average score")
            st.plotly_chart(style_fig(fig, 360), width="stretch")
            pivot = by_group.pivot(index="Year", columns="HDG", values="Inequality")
            if {"Low", "Very High"}.issubset(pivot.columns):
                gaps = (pivot["Low"] - pivot["Very High"]).dropna()
                if len(gaps) > 1:
                    note(
                        f"The Low vs Very High gap was {gaps.iloc[0]:.1f} points in {gaps.index[0]} and "
                        f"{gaps.iloc[-1]:.1f} in {gaps.index[-1]}. A falling global average hides a "
                        "lasting disadvantage in the least developed countries, so funding should not "
                        "follow the global average."
                    )
                else:
                    note("Not enough years in view to compare the gap.")
            else:
                note("Keep both the Low and Very High groups (and their regions) in view to compare how the gap changes.")


# ---------------------------------------------------------------------------
# TAB 2: WHERE TO ACT (region bars, country ranking)
# ---------------------------------------------------------------------------
def render_act():
    if current.empty:
        st.warning("No countries match the current filters for this year. Widen the filters.")
        return

    # ---- Visual 4: regions ---------------------------------------------------
    with st.container(border=True):
        card_header(f"Which regions have the highest inequality in {year}?",
                    "Where should regional programs start?")
        reg = (current.groupby("Region")["Inequality"]
               .agg(Average="mean", Countries="count").reset_index()
               .sort_values("Average", ascending=False))
        colors = [GREY if r == "Unclassified" else c
                  for r, c in zip(reg["Region"], score_colors(reg["Average"]))]
        fig = go.Figure(go.Bar(
            x=reg["Average"], y=reg["Region"], orientation="h", marker_color=colors,
            text=reg["Average"].round(1), textposition="outside", cliponaxis=False,
            customdata=reg["Countries"],
            hovertemplate="%{y}<br>Average %{x:.2f}<br>%{customdata} countries<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(title="Average score", range=[0, SCORE_MAX])
        st.plotly_chart(style_fig(fig, 340), width="stretch")
        top_r, bot_r = reg.iloc[0], reg.iloc[-1]
        text = (f"{top_r['Region']} has the highest average ({top_r['Average']:.1f}) and "
                f"{bot_r['Region']} the lowest ({bot_r['Average']:.1f}). Regional programs should "
                "start with the longest bars. ")
        if "Unclassified" in reg["Region"].values:
            n_unc = int(reg.loc[reg["Region"] == "Unclassified", "Countries"].iloc[0])
            text += (f"The grey bar covers {n_unc} countries with no UNDP region, mostly Very High "
                     "development, so a region-only view would miss them.")
        note(text)

    # ---- Visual 5: top and bottom countries -----------------------------------
    with st.container(border=True):
        card_header(f"Which countries are the most and least unequal in {year}?",
                    "Which countries should be reviewed first for funding?")
        ranked = current.sort_values("Inequality", ascending=False)
        n_show = max(1, min(10, len(ranked) // 2))
        top = ranked.head(n_show).assign(Group="Most unequal")
        bottom = ranked.tail(n_show).assign(Group="Least unequal")
        extremes = pd.concat([top, bottom])
        fig = px.bar(
            extremes, x="Inequality", y="Country", color="Group", orientation="h",
            color_discrete_map={"Most unequal": RED, "Least unequal": BLUE},
            hover_data={"HDG": True, "Region": True, "Group": False},
            text=extremes["Inequality"].round(1),
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(title="Score", range=[0, SCORE_MAX + 4])
        st.plotly_chart(style_fig(fig, max(360, 30 * len(extremes) + 100)), width="stretch")
        hi, lo = ranked.iloc[0], ranked.iloc[-1]
        ratio = f", {hi['Inequality'] / lo['Inequality']:.0f} times higher" if lo["Inequality"] > 0 else ""
        top_group = top["HDG"].value_counts().index[0]
        note(
            f"{hi['Country']} ({hi['Inequality']:.1f}) is the most unequal and {lo['Country']} "
            f"({lo['Inequality']:.1f}) the least{ratio}. Most of the top {n_show} are {top_group} "
            "development countries. This list is the starting shortlist, refined with the model "
            "on the next tab."
        )


# ---------------------------------------------------------------------------
# TAB 3: PRIORITY COUNTRIES (predictive model, visual 6)
# ---------------------------------------------------------------------------
def render_model():
    model = run_model(long_df)
    results = model["results"]
    best_name, best_rmse = model["best_name"], model["best_rmse"]
    cv = model["cv"].set_index("Model")

    st.markdown(
        '<p class="qtitle">Which countries are more unequal than their development level predicts?</p>'
        f'<p class="qsub">Model target: inequality in {TARGET_YEAR}. Features: HDI rank, Human Development '
        "Group, UNDP region. Previous-year inequality is excluded on purpose (r = 0.997 with the current "
        "value). Every country is predicted by models that never saw it (repeated 5-fold cross-validation).</p>",
        unsafe_allow_html=True,
    )

    multiple = st.slider(
        "Priority threshold: actual is above predicted by more than this many RMSE",
        0.5, 2.0, 1.0, 0.25,
        help="1.0 flags countries whose actual score is more than one typical model error above the prediction.",
    )
    min_score = st.slider(
        "Screen by actual score: only flag countries scoring at least",
        0, 40, 0, 5,
        help="0 keeps every country above the priority line. Raise it to drop low-inequality countries, "
             "for example 20 leaves only countries that score 20 or more today.",
    )
    threshold = multiple * best_rmse
    results = results.assign(
        Priority=(results["Gap (Actual - Predicted)"] > threshold) & (results["Actual"] >= min_score),
        Caution=np.where(results["Predicted"] < 0, "Predicted below zero: check actual score", ""),
    )

    m1, m2, m3, m4 = st.columns(4)
    kpi(m1, "Best model", best_name, "Lowest cross-validated error")
    kpi(m2, "Cross-validated RMSE", f"{cv.loc[best_name, 'CV RMSE']:.2f}",
        f"Baseline: {cv.loc[BASELINE, 'CV RMSE']:.2f}")
    kpi(m3, "Cross-validated R²", f"{cv.loc[best_name, 'CV R²']:.2f}",
        f"Baseline: {cv.loc[BASELINE, 'CV R²']:.2f}")
    kpi(m4, "Priority countries", f"{int(results['Priority'].sum())}",
        f"of {len(results)} modelled", "bad")
    st.write("")

    view = results[results["Human Development Group"].isin(hdg_sel) & results["UNDP Region"].isin(region_sel)]
    if country_sel:
        view = view[view["Country"].isin(country_sel)]

    with st.container(border=True):
        card_header("Predicted vs actual inequality",
                    "Which countries sit well above what their development level predicts?")
        if view.empty:
            st.warning("No modelled countries match the current filters.")
        else:
            fig = px.scatter(
                view, x="Predicted", y="Actual", color="Human Development Group",
                category_orders={"Human Development Group": HDG_ALL}, color_discrete_map=HDG_COLORS,
                hover_data={"Country": True, "UNDP Region": True, "Gap (Actual - Predicted)": True,
                            "Predicted": ":.1f", "Actual": ":.1f"},
            )
            lim = [0, max(view["Actual"].max(), view["Predicted"].max()) + 3]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Actual = Predicted",
                                     line=dict(color=GREY, dash="dash"), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=lim, y=[v + threshold for v in lim], mode="lines",
                                     name=f"Priority line (+{threshold:.1f})",
                                     line=dict(color=RED, dash="dot"), hoverinfo="skip"))
            flagged = view[view["Priority"]]
            if not flagged.empty:
                fig.add_trace(go.Scatter(
                    x=flagged["Predicted"], y=flagged["Actual"], mode="markers", name="Priority country",
                    marker=dict(symbol="circle-open", size=15, line=dict(color=RED, width=2)),
                    text=flagged["Country"], hovertemplate="%{text}<extra>Priority</extra>"))
            fig.update_xaxes(title="Predicted score", range=lim)
            fig.update_yaxes(title="Actual score", range=lim)
            st.plotly_chart(style_fig(fig, 520), width="stretch")
            if len(flagged):
                top_flag = flagged.sort_values("Gap (Actual - Predicted)", ascending=False).iloc[0]
                common = flagged["UNDP Region"].value_counts().index[0]
                note(
                    f"{len(flagged)} countries in view sit above the red line. The largest gap is "
                    f"{top_flag['Country']} (actual {top_flag['Actual']:.1f} vs predicted "
                    f"{top_flag['Predicted']:.1f}), and {common} appears most often. These countries "
                    "are more unequal than their development level explains, so they are candidates "
                    "for a funding review beyond the development-group rule."
                )
            else:
                note("No country in view exceeds the priority line. Lower the threshold, lower the minimum score, or widen the filters.")
            caution = flagged[flagged["Caution"] != ""]
            if not caution.empty:
                names = ", ".join(caution["Country"])
                st.warning(
                    f"Caution: {names} {'are' if len(caution) > 1 else 'is'} flagged only because the linear model "
                    "predicts a score below zero. Raise the minimum score to screen out low-inequality countries."
                )

    if not view.empty:
        st.markdown("**Priority countries, largest gap first**")
        table = (view[view["Priority"]].sort_values("Gap (Actual - Predicted)", ascending=False)
                 .drop(columns="Priority").reset_index(drop=True))
        st.dataframe(table, width="stretch", hide_index=True)

    with st.expander("Model evaluation: baseline vs models, overfitting check, feature importance"):
        st.markdown("**Repeated 5-fold cross-validation (5 repeats)**")
        st.dataframe(model["cv"].round(3), width="stretch", hide_index=True)
        st.markdown("**80/20 hold-out split (compare train and test error to check overfitting)**")
        st.dataframe(model["holdout"].round(3), width="stretch", hide_index=True)
        fig = px.bar(model["importance"], x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[BLUE], title="Random Forest feature importance")
        st.plotly_chart(style_fig(fig, 360), width="stretch")
        st.caption(
            f"{model['n_countries']} countries were modelled (countries without an HDI rank or group, "
            "such as Somalia, are left out). With this sample size, cross-validation is more reliable "
            "than a single split. HDI rank and Human Development Group carry overlapping information, "
            "so importance is shared between them."
        )


# ---------------------------------------------------------------------------
# TAB 4: DATA NOTES
# ---------------------------------------------------------------------------
def render_notes():
    st.markdown("**How to read this dashboard**")
    st.markdown(
        "- Source: UNDP Inequality in Education, prepared in the EDA notebook and saved as df_long.csv "
        "(one row per country and year).\n"
        "- A higher score means education is more unequally distributed in that country.\n"
        "- 13 countries with no value in any year were removed (195 to 182 countries). Remaining gaps "
        "were kept as missing, not filled.\n"
        "- Coverage grows from 137 countries in 2010 to 176 in 2021, so yearly averages are not based "
        "on the same set of countries.\n"
        "- Many yearly values are carried forward, and 2021 repeats 2020, so year-to-year change is "
        "small by construction.\n"
        "- 42 countries have no UNDP region (almost all Very High development). They appear as "
        "'Unclassified' so they are not silently dropped.\n"
        "- The model explains inequality from development characteristics. It does not forecast future "
        "change and does not show cause and effect."
    )
    cov = obs.groupby("Year")["Inequality"].count().rename("Countries with data").reset_index()
    st.markdown("**Countries with data per year**")
    st.dataframe(cov, width="stretch", hide_index=True)
    st.download_button(
        "Download the filtered country-year data (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_inequality_data.csv",
        mime="text/csv",
    )


with tab_over:
    render_overview()
with tab_act:
    render_act()
with tab_model:
    render_model()
with tab_notes:
    render_notes()
