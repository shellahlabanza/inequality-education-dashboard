# Educational Inequality Dashboard (HTML, Vercel-ready)

Group 6.1 (Labanza and Lontoc) | Foundations of Business Analytics | Final Project

Python does all the analysis (`build_data.py`: EDA statistics, the predictive model, cross-validation, residuals).
It writes `public/data.js`. The page `public/index.html` is plain HTML + JavaScript + Plotly, so Vercel hosts it as a static site (no server, no Python on Vercel).
Plotly and the world-map shapes are bundled in `public/`, so nothing loads from the internet.

## Folder layout
```
inequality_dashboard/
├── data/df_long.csv               <- notebook Section 7 output
├── build_data.py                  <- Python: EDA + model -> public/data.js
├── run.py                         <- one command: build + open the dashboard
├── requirements.txt               <- pandas, numpy, scikit-learn
├── notebooks/Group_6_1_Labanza_Lontoc_Final.ipynb  <- your EDA notebook + Section 8 (project plan) + Section 9 (model) (deliverable 3)
├── streamlit_version/             <- original Streamlit app (alternative tool)
├── vercel.json
└── public/                        <- this folder is what Vercel serves
    ├── index.html, data.js, plotly.min.js, topojson/world_110m.json
```

## Run it in VS Code
1. Install Python 3.9+ (`python --version`) and VS Code with the **Python** extension.
2. **File > Open Folder** > choose `inequality_dashboard`.
3. **Terminal > New Terminal**.
4. Create the virtual environment
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
5. `pip install -r requirements.txt`
6. `python run.py`
   This runs the Python analysis (you will see the model results printed), then opens **http://localhost:8000**.
7. Stop with `Ctrl + C`.

Do not double-click `index.html`: the map shapes need a web server, which `run.py` starts.
Optional: the **Live Server** VS Code extension also works if you right-click `public/index.html` > Open with Live Server (run `python build_data.py` first).

If you change `data/df_long.csv` or the model, run `python build_data.py` (or `python run.py`) again.

## What is on the dashboard
| Tab | Contents |
|---|---|
| Overview | 4 KPIs, world map, average trend, trend by development group |
| Where to act | Inequality by region, most and least unequal countries |
| Priority countries (model) | Predicted vs actual, priority table (filter-aware, CSV download, caution flag), threshold slider, minimum-score screen, justification, CV and hold-out tables, feature importance |
| Data notes | Data caveats, years of data per country (follows the filters), coverage per year, CSV download |

The dashboard has exactly 6 visuals (map, trend, group trend, region, ranking, predicted vs actual) to meet the "4 to 6 visuals" rule. Feature importance sits inside a collapsed section of the model tab. The notebook EDA views live only in the notebook and the paper, because a program manager cannot act on them.

## Streamlit version (the tool named in the brief)
The brief lists Power BI, Tableau, Looker Studio, Streamlit or Plotly Dash. If your instructor wants one of those, use `streamlit_version/` (your original app, same 6 visuals, same model):
```
cd streamlit_version
pip install -r requirements.txt
streamlit run app.py
```
Deploy on Streamlit Community Cloud (share.streamlit.io) by choosing the repository and `app.py`.
Filters: year, Human Development Group, UNDP region, country, plus the priority-threshold slider.

## Deploy to Vercel
**Option A: GitHub (recommended)**
1. Create a GitHub repository and push this folder (`git init`, `git add .`, `git commit -m "dashboard"`, `git remote add origin <url>`, `git push -u origin main`). Commit `public/data.js`.
2. On vercel.com: **Add New > Project**, import the repository.
3. Framework Preset: **Other**. Build Command: leave empty. Output Directory: `public` (vercel.json already sets this). Click **Deploy**.

**Option B: Vercel CLI**
1. `npm i -g vercel`
2. In the project folder run `vercel`, accept the defaults, then `vercel --prod`.

Your live link is the `*.vercel.app` address Vercel shows. Submit it as deliverable 2.

## Before you submit (not part of the dashboard)
- Colab notebook: upload `notebooks/Group_6_1_Labanza_Lontoc_Final.ipynb` to Colab, run all cells from the top (the raw data file is needed for Sections 1 to 7), and submit it.
- IEEE paper (PDF) and the 5 to 10 minute recorded demo are still yours to produce.
- Findings 6 to 8 on the Plan tab were added to reach the rubric's "8 to 10 findings". Read them and keep or edit them in `build_data.py`.

## Troubleshooting
| Problem | Fix |
|---|---|
| `python` not found | Try `python3`, or reinstall Python and tick "Add to PATH" |
| Map is blank | You opened the file directly. Use `python run.py` |
| "Files not loaded" message | Run `python build_data.py` to create `public/data.js` |
| Port 8000 busy | Edit the port number in `run.py` |
