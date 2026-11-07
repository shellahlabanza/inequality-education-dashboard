"""
build_data.py  -  Group 6.1 (Labanza and Lontoc) | Foundations of Business Analytics

Runs ALL the Python analysis and writes public/data.js for the HTML dashboard:
  * country-year table from data/df_long.csv (saved by Section 7 of the EDA notebook)
  * EDA statistics from the notebook (frequency, central tendency, dispersion, position, correlation)
  * the predictive model (baseline vs Linear Regression vs Random Forest,
    repeated 5-fold cross-validation, 80/20 hold-out, feature importance, out-of-fold residuals)
  * the Findings Summary numbers

Run:  python build_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, train_test_split

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "df_long.csv"
OUT = ROOT / "public" / "data.js"

TARGET_YEAR = 2021
HIGH = 30
HDG_ORDER = ["Low", "Medium", "High", "Very High"]
REGION_NAMES = {
    "SSA": "Sub-Saharan Africa", "SA": "South Asia", "AS": "Arab States",
    "LAC": "Latin America & Caribbean", "EAP": "East Asia & Pacific",
    "ECA": "Europe & Central Asia", "Unclassified": "Unclassified",
}
BASELINE, LINEAR, FOREST = "Baseline (group mean)", "Linear Regression", "Random Forest"


def r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x) + 1e-9, n)  # half-up, as in the notebook


# ---------------------------------------------------------------- load
if not CSV.exists():
    raise SystemExit(f"Missing {CSV}. Put df_long.csv inside the data/ folder.")
long_df = pd.read_csv(CSV)
long_df["HDG"] = long_df["Human Development Groups"].fillna("Unclassified")
long_df["Region_Code"] = long_df["UNDP Developing Regions"].fillna("Unclassified")
long_df["Region"] = long_df["Region_Code"].map(REGION_NAMES)
obs = long_df.dropna(subset=["Inequality"]).copy()
print(f"Loaded {long_df['ISO3'].nunique()} countries, {len(long_df)} rows, {len(obs)} observed values")

# ---------------------------------------------------------------- compact tables for the browser
meta = (long_df.drop_duplicates("ISO3")[["ISO3", "Country", "HDG", "Region", "HDI Rank (2021)"]]
        .sort_values("Country").reset_index(drop=True))
iso_idx = {iso: i for i, iso in enumerate(meta["ISO3"])}
countries = [
    {"iso": a, "name": b, "hdg": c, "region": d, "hdi": (None if pd.isna(e) else int(e))}
    for a, b, c, d, e in meta.itertuples(index=False)
]
# one row = [country index, year, value]
rows = [[iso_idx[i], int(y), round(float(v), 4)] for i, y, v in zip(obs["ISO3"], obs["Year"], obs["Inequality"])]


# ---------------------------------------------------------------- EDA statistics (mirrors notebook Section 7)
def central(s):
    return {"n": int(s.count()), "mean": r(s.mean()), "median": r(s.median()),
            "mode": r(s.round().mode().iloc[0]) if len(s.dropna()) else None}


def disp(s):
    return {"n": int(s.count()), "std": r(s.std()), "var": r(s.var()), "min": r(s.min()), "max": r(s.max()),
            "range": r(s.max() - s.min()), "iqr": r(s.quantile(.75) - s.quantile(.25)),
            "cv": r(s.std() / s.mean() * 100, 1)}


by_year_c = {int(y): central(g["Inequality"]) for y, g in obs.groupby("Year")}
by_year_d = {int(y): disp(g["Inequality"]) for y, g in obs.groupby("Year")}
by_hdg = {g: {**central(d["Inequality"]), **disp(d["Inequality"]),
              "q1": r(d["Inequality"].quantile(.25)), "q3": r(d["Inequality"].quantile(.75))}
          for g, d in obs.groupby("HDG")}
by_region = {g: {**central(d["Inequality"]), **disp(d["Inequality"])} for g, d in obs.groupby("Region")}

bins = list(range(0, 55, 5))
cut = pd.cut(obs["Inequality"], bins=bins, right=False).value_counts().sort_index()
freq = [{"bin": f"{int(i.left)}-{int(i.right)}", "n": int(n), "pct": r(n / len(obs) * 100, 1)} for i, n in cut.items()]

levels = [.05, .10, .25, .50, .75, .90, .95]
quant = [{"p": f"{int(l*100)}th",
          "all": r(obs["Inequality"].quantile(l)),
          "y2010": r(obs.loc[obs.Year == 2010, "Inequality"].quantile(l)),
          "y2021": r(obs.loc[obs.Year == 2021, "Inequality"].quantile(l))} for l in levels]
deciles = [{"d": f"D{int(l*10)}",
            "y2010": r(obs.loc[obs.Year == 2010, "Inequality"].quantile(l)),
            "y2021": r(obs.loc[obs.Year == 2021, "Inequality"].quantile(l))} for l in np.arange(.1, 1.0, .1)]

corr_cols = ["Year", "Inequality", "Previous_Inequality", "Inequality_Change"]
corr = long_df[corr_cols].corr()
corr_out = {"labels": corr_cols, "z": corr.round(3).values.tolist()}

# position change 2010 -> 2021, countries observed in both years
w = obs.pivot(index="Country", columns="Year", values="Inequality")[[2010, 2021]].dropna()
w["change"] = w[2021] - w[2010]
w = w.sort_values("change")
movers = {
    "n": int(len(w)),
    "improved": [{"country": c, "y2010": r(a), "y2021": r(b), "change": r(ch)} for c, a, b, ch in w.head(8).itertuples()],
    "worsened": [{"country": c, "y2010": r(a), "y2021": r(b), "change": r(ch)} for c, a, b, ch in w.tail(8).iloc[::-1].itertuples()],
    "n_improved": int((w["change"] < 0).sum()), "n_worse": int((w["change"] > 0).sum()),
    "n_same": int((w["change"] == 0).sum()),
}

chg = long_df["Inequality_Change"].dropna()
persistence = {
    "r_prev": r(long_df[["Inequality", "Previous_Inequality"]].corr().iloc[0, 1], 3),
    "zero_change_pct": r((chg == 0).mean() * 100, 0),
    "n_changes": int(len(chg)),
    "y2021_equals_2020": bool(
        (obs.pivot(index="ISO3", columns="Year", values="Inequality")[[2020, 2021]].dropna().diff(axis=1)[2021] == 0).all()),
}

coverage = {int(y): int(n) for y, n in obs.groupby("Year")["Inequality"].count().items()}
counts_hdg = long_df.drop_duplicates("ISO3")["HDG"].value_counts().to_dict()
counts_reg = long_df.drop_duplicates("ISO3")["Region"].value_counts().to_dict()
unc = long_df.drop_duplicates("ISO3")
unc = unc[unc["Region_Code"] == "Unclassified"]
n_unc = int(len(unc))
n_unc_vh = int((unc["HDG"] == "Very High").sum())

# ---------------------------------------------------------------- predictive model (same as app.py)
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_true - y_pred
    ss = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e))), (1 - float(np.sum(e**2)) / ss if ss > 0 else float("nan"))


def fit_predict(name, xtr, ytr, gtr, xte, gte):
    if name == BASELINE:
        means, overall = ytr.groupby(gtr).mean(), ytr.mean()
        f = lambda g: g.map(means).fillna(overall).to_numpy()
        return f(gtr), f(gte)
    model = LinearRegression() if name == LINEAR else RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(xtr, ytr)
    return model.predict(xtr), model.predict(xte)


d = (long_df[long_df["Year"] == TARGET_YEAR]
     .dropna(subset=["Inequality", "HDI Rank (2021)", "Human Development Groups"]).reset_index(drop=True))
x = pd.DataFrame({
    "HDI_Rank": d["HDI Rank (2021)"].astype(float),
    "HDG_Code": d["Human Development Groups"].map({g: i for i, g in enumerate(HDG_ORDER)}).astype(float),
})
x = pd.concat([x, pd.get_dummies(d["Region_Code"], prefix="Region", dtype=int)], axis=1)
y = d["Inequality"].astype(float)
grp = d["Human Development Groups"].astype(str)
names = [BASELINE, LINEAR, FOREST]

itr, ite = train_test_split(np.arange(len(d)), test_size=0.2, random_state=42)
holdout = []
for n in names:
    ptr, pte = fit_predict(n, x.iloc[itr], y.iloc[itr], grp.iloc[itr], x.iloc[ite], grp.iloc[ite])
    rtr, _, r2tr = metrics(y.iloc[itr], ptr)
    rte, mte, r2te = metrics(y.iloc[ite], pte)
    holdout.append({"model": n, "train_rmse": r(rtr, 3), "test_rmse": r(rte, 3), "test_mae": r(mte, 3),
                    "train_r2": r(r2tr, 3), "test_r2": r(r2te, 3)})

sc = {n: {"rmse": [], "mae": [], "r2": []} for n in names}
oof_sum = {n: np.zeros(len(d)) for n in names}
oof_cnt = np.zeros(len(d))
for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=42).split(x):
    oof_cnt[te] += 1
    for n in names:
        _, p = fit_predict(n, x.iloc[tr], y.iloc[tr], grp.iloc[tr], x.iloc[te], grp.iloc[te])
        a, b, c = metrics(y.iloc[te], p)
        sc[n]["rmse"].append(a); sc[n]["mae"].append(b); sc[n]["r2"].append(c)
        oof_sum[n][te] += p
cv = [{"model": n, "cv_rmse": r(np.mean(sc[n]["rmse"]), 3), "rmse_std": r(np.std(sc[n]["rmse"]), 3),
       "cv_mae": r(np.mean(sc[n]["mae"]), 3), "cv_r2": r(np.mean(sc[n]["r2"]), 3)} for n in names]
best = sorted([c for c in cv if c["model"] != BASELINE], key=lambda c: c["cv_rmse"])[0]
oof = oof_sum[best["model"]] / oof_cnt

results = []
for i in range(len(d)):
    results.append({
        "country": d.loc[i, "Country"], "iso": d.loc[i, "ISO3"], "hdg": d.loc[i, "HDG"], "region": d.loc[i, "Region"],
        "hdi": int(d.loc[i, "HDI Rank (2021)"]), "actual": r(y[i]), "pred": r(oof[i]), "gap": r(y[i] - oof[i]),
    })
rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(x, y)
imp = sorted(({"feature": f, "importance": r(v, 4)} for f, v in zip(x.columns, rf.feature_importances_)),
             key=lambda t: t["importance"])
lin = LinearRegression().fit(x, y)
model_out = {
    "target_year": TARGET_YEAR, "n": int(len(d)), "best": best["model"], "best_rmse": best["cv_rmse"],
    "cv": cv, "holdout": holdout, "importance": imp, "results": results,
    "features": list(x.columns),
}
base_cv = next(c for c in cv if c["model"] == BASELINE)
n_prio = sum(1 for t in results if t["gap"] > best["cv_rmse"])
print("CV:", cv)
print("Best:", best["model"], "| priority countries at 1.0 RMSE:", n_prio)

# ---------------------------------------------------------------- Findings summary (numbers computed from data)
raw = obs.groupby('Year')['Inequality'].agg(['mean', 'median'])   # unrounded, so percentages match the notebook
m10, m21 = raw.loc[2010, 'mean'], raw.loc[2021, 'mean']
gh = obs.groupby(["Year", "HDG"])["Inequality"].mean().unstack()
gap10, gap21 = gh.loc[2010, "Low"] - gh.loc[2010, "Very High"], gh.loc[2021, "Low"] - gh.loc[2021, "Very High"]
o10, o21 = obs[obs.Year == 2010]["Inequality"], obs[obs.Year == 2021]["Inequality"]
reg_mean = obs.groupby("Region")["Inequality"].mean().sort_values(ascending=False)
share_high = (o21 >= HIGH).mean() * 100
top21 = obs[obs.Year == 2021].sort_values("Inequality", ascending=False).iloc[0]
low_in_top = obs[(obs.Year == 2021)].nlargest(10, "Inequality")["HDG"].value_counts()
findings = [
    {"title": "Average inequality is falling, but slowly",
     "text": f"The mean dropped from {m10:.2f} in 2010 to {m21:.2f} in 2021 (down {m10-m21:.2f} points, {(m10-m21)/m10*100:.1f}%) "
             f"and the median from {by_year_c[2010]['median']} to {by_year_c[2021]['median']}. Part of the drop comes from coverage "
             f"growing from {coverage[2010]} to {coverage[2021]} countries.", "src": "Project plan, finding 1"},
    {"title": "Human development level is the strongest separator",
     "text": f"Mean inequality is {by_hdg['Low']['mean']} (Low), {by_hdg['Medium']['mean']} (Medium), {by_hdg['High']['mean']} (High) and "
             f"{by_hdg['Very High']['mean']} (Very High). The Low-group first quartile ({by_hdg['Low']['q1']}) is above the "
             f"Very High-group third quartile ({by_hdg['Very High']['q3']}), so the groups do not overlap.", "src": "Project plan, finding 2"},
    {"title": "The gap is not closing, and the most unequal countries improve least",
     "text": f"The Low vs Very High gap was {gap10:.2f} points in 2010 and {gap21:.2f} in 2021. The 9th decile barely moved "
             f"({deciles[8]['y2010']} to {deciles[8]['y2021']}) while the median and lower deciles fell.", "src": "Project plan, finding 3"},
    {"title": "Regions differ sharply, and some countries have no region",
     "text": f"{reg_mean.index[0]} ({reg_mean.iloc[0]:.2f}) and {reg_mean.index[1]} ({reg_mean.iloc[1]:.2f}) are highest; "
             f"Europe & Central Asia is lowest among classified regions ({reg_mean['Europe & Central Asia']:.2f}). {n_unc} countries "
             f"({n_unc/182*100:.1f}%) have no UNDP region and {n_unc_vh} of them are Very High development.", "src": "Project plan, finding 4"},
    {"title": "Inequality is highly persistent, which limits what the data can show",
     "text": f"Current and previous-year values correlate at r = {persistence['r_prev']}, {persistence['zero_change_pct']:.0f}% of "
             f"year-to-year changes are exactly zero, and 2021 repeats 2020 for every country.", "src": "Project plan, finding 5"},
    {"title": "Inequality is right-skewed: a long tail of very unequal countries",
     "text": f"In 2021 the mean ({by_year_c[2021]['mean']}) sits above the median ({by_year_c[2021]['median']}), and "
             f"{share_high:.1f}% of countries score {HIGH} or more. {top21['Country']} is the highest at {top21['Inequality']:.1f}.",
     "src": "Notebook Sections 7.1 to 7.2"},
    {"title": "Most countries barely moved between 2010 and 2021",
     "text": f"Of {movers['n']} countries observed in both years, {movers['n_improved']} improved, {movers['n_worse']} worsened and "
             f"{movers['n_same']} did not change. Largest improvement: {movers['improved'][0]['country']} ({movers['improved'][0]['change']:+.1f} points).",
     "src": "Notebook Section 7.4c"},
    {"title": "The typical country improves faster than the average",
     "text": f"From 2010 to 2021 the median fell {(1-raw.loc[2021,'median']/raw.loc[2010,'median'])*100:.1f}% "
             f"({by_year_c[2010]['median']} to {by_year_c[2021]['median']}) while the mean fell {(m10-m21)/m10*100:.1f}%. The coefficient of "
             f"variation rose from {by_year_d[2010]['cv']}% to {by_year_d[2021]['cv']}% because the center falls while the spread stays.",
     "src": "Notebook Section 7.5"},
]

# ---------------------------------------------------------------- write
payload = {
    "meta": {"title": "Educational Inequality Dashboard", "years": [int(obs.Year.min()), int(obs.Year.max())],
             "high": HIGH, "n_countries": int(long_df.ISO3.nunique()), "n_rows": int(len(long_df)),
             "n_obs": int(len(obs)), "n_unclassified": n_unc, "n_unclassified_vh": n_unc_vh},
    "countries": countries, "rows": rows,
    "eda": {"persistence": persistence, "coverage": coverage},
    "model": model_out,
}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("window.DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
