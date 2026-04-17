# Spatial-Behavioural Analytics: Mapping Nutritional Vulnerability and Food Accessibility in Singapore's Aging Districts

A reproducible, end-to-end data mining pipeline that combines association rule
mining, unsupervised clustering and supervised risk modelling to identify
Singapore subzones where elderly residents are most at risk of nutritional
vulnerability. The project uses only publicly available datasets.

## 1. Project overview

The pipeline answers three concrete questions about Singapore's 332 census
subzones:

1. **What real-world combinations of demographics, food access, transit access
   and barrier-free infrastructure tend to co-occur?**
   We answer this with **Apriori association rule mining** on a transparently
   constructed binary item matrix.
2. **Are there natural groups of subzones that share a "vulnerability
   profile"?**
   We answer this with **unsupervised clustering** (K-Means, Agglomerative
   Ward and DBSCAN) and select the best partition using a sweep over three
   internal metrics plus a bootstrap stability check.
3. **Which subzones are the highest-risk priority for outreach today, and what
   features drive that risk?**
   We answer this with a **Random Forest** trained against a transparent,
   weighted z-score proxy target whose definition is fully documented in
   `outputs/tables/vulnerability_target_definition.txt`.

## 2. Problem statement

Singapore is one of the fastest-ageing societies in Asia. By 2030 roughly one
in four residents will be aged 65 or above, and elderly residents rely much
more heavily than the general population on **walkable** access to
affordable food (hawker centres, wet markets, supermarkets) and on **frequent
public transport** for daily errands. Barriers such as long walking distances,
poor pedestrian infrastructure and the absence of barrier-free buildings can
quietly turn an otherwise food-secure neighbourhood into a high-risk one for
its older residents. Existing official indicators report each of these
dimensions in isolation. The contribution of this project is to fuse them at
**subzone level** using only open data, and to flag the small set of
neighbourhoods where multiple risk factors compound.

## 3. Datasets

All raw files live in `data/raw/`. 10 files are required and can be
downloaded from data.gov.sg. Two additional files are optional and
require an LTA DataMall API key.

### 3.1 Required datasets

Most GeoJSON files are included in the repository. The three files
marked **download separately** must be placed in `data/raw/` before
running the pipeline (they are too large for GitHub or are CSV files
excluded by `.gitignore`).

| # | File | Source | Used for |
|---|------|--------|----------|
| 1 | `subzone_boundary.geojson` | data.gov.sg Master Plan Subzone Boundary | Spatial unit (332 subzones) |
| 2 | `ResidentPopulationbyPlanningAreaSubzoneofResidenceAgeGroupandSexCensusofPopulation2020.csv` | SingStat Census 2020 | Elderly count and share (**download separately** — CSV excluded by gitignore) |
| 3 | `HawkerCentresGEOJSON.geojson` | data.gov.sg | Affordable food access |
| 4 | `SupermarketsGEOJSON.geojson` | data.gov.sg | Food access |
| 5 | `NEAMarketandFoodCentre.geojson` | NEA via data.gov.sg | Wet markets (`TYPE == 'MK'`) |
| 6 | `LTABusStop.geojson` | LTA DataMall via data.gov.sg | Bus access |
| 7 | `LTAMRTStationExitGEOJSON.geojson` | LTA via data.gov.sg | MRT access |
| 8 | `FriendlyBuildings.geojson` | data.gov.sg | Barrier-free / accessibility infrastructure |
| 9 | `MasterPlan2025LandUseLayer.geojson` | URA Master Plan 2025 | Residential land-use filter (download separately, exceeds GitHub limit)(https://data.gov.sg/datasets/d_a8c3546b26712e35021f3a681d0353ae/view) |
| 10 | `SeniorActivityCentresAndActiveAgeingCentresAnnual.csv` | data.gov.sg | National context for active-ageing centres |

### 3.2 Optional datasets (LTA DataMall API required)

| # | File | Source | Used for |
|---|------|--------|----------|
| 11 | `od_bus.csv` | LTA DataMall Passenger Volume OD | Optional bus mobility context |
| 12 | `od_train.csv` | LTA DataMall Passenger Volume OD | Optional rail mobility context |

These two OD files are **optional and disabled by default**
(`features.od.enable: false` in `config/settings.yaml`). They are not
part of the final risk model.

### 3.3 How to obtain od_bus.csv and od_train.csv from LTA DataMall

1. Go to the LTA DataMall API request page:
   https://datamall.lta.gov.sg/content/datamall/en/request-for-api.html

2. Fill in the form. Choose the appropriate purpose (e.g. research or
   student project). Submit the form.

3. Check your email. After approval, LTA will send you an API Account
   Key by email.

4. Copy your API Account Key. Keep this private. Do not hardcode it
   into the code and do not upload it to GitHub.

5. Open a terminal and make sure you are in the project directory.

6. Set your API key as an environment variable:

   **Windows Command Prompt (persists across terminals):**
   ```cmd
   setx LTA_API_KEY "YOUR_API_ACCOUNT_KEY"
   ```

   **Windows Command Prompt (current session only):**
   ```cmd
   set LTA_API_KEY=YOUR_API_ACCOUNT_KEY
   ```

   **PowerShell (current session only):**
   ```powershell
   $env:LTA_API_KEY="YOUR_API_ACCOUNT_KEY"
   ```

7. If you used `setx`, close the terminal and open a new one. This is
   required because `setx` does not update the current terminal session.

8. Verify the key is available:

   **Command Prompt:**
   ```cmd
   echo %LTA_API_KEY%
   ```

   **PowerShell:**
   ```powershell
   echo $env:LTA_API_KEY
   ```

9. Download the Passenger Volume by Origin Destination datasets from the
   LTA DataMall dynamic datasets page and place the resulting CSV files
   as `data/raw/od_bus.csv` and `data/raw/od_train.csv`.

> **Note:** The pipeline runs fully without these files. They are used
> only for optional contextual analysis and are excluded from the final
> risk model.

## 4. Setup

The pipeline targets Python 3.11+ and has been validated end-to-end on
Windows 11 with both Python 3.11 and 3.12.

### Option A — virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate elderly_food_vulnerability
```

### Option C — Docker

```bash
docker compose build
docker compose run --rm pipeline
```

`outputs/` is mounted into the container so all artefacts appear on the host.

## 5. How to run

```bash
# 1. Place the required files in data/raw/ (see Section 3).
# 2. From the project root:
python run.py
```

That single command regenerates every figure, table, model and log file. To
run only part of the pipeline:

```bash
python run.py --stage features      # build the master feature matrix
python run.py --stage apriori       # Phase 1 — association rules
python run.py --stage clustering    # Phase 2 — clustering
python run.py --stage all           # default — run everything
```

To run the unit tests:

```bash
pytest tests/
```

## 6. Method overview

The pipeline is organised into three phases on top of a shared feature layer.

### 6.1 Feature engineering

Per subzone the pipeline computes:

- **Demographics** — `elderly_count`, `elderly_pct`, `elderly_density`,
  `elderly_density_log`, `area_km2`.
- **Food access** — straight-line nearest distance to the nearest hawker,
  supermarket and market; counts within a 1 km buffer plus a composite
  `food_access_score`, `food_amenity_diversity` and `nearest_food_km`.
- **Transit access** — bus stops within 0.5 km and MRT exits within 0.8 km,
  combined into `transit_access_score`.
- **Accessibility support** — count of barrier-free `FriendlyBuildings`
  within 0.5 km, combined into `accessibility_support_score`.
- **Land use** — overlay against the Master Plan 2025 layer gives
  `residential_share`, `residential_area_km2` and `is_residential`. Only
  subzones with non-trivial residential land area enter the modelling subset.

All distances are computed in EPSG:3414 (SVY21) for metric correctness, then
reprojected back for plotting. Feature construction lives in
`src/features/{demographic, food_access, transit, accessibility, landuse,
binary, assemble}.py`.

### 6.2 Phase 1 — Apriori association rule mining

`src/mining/apriori.py` builds a binary item matrix from a small set of
explicit, configurable, quantile-based rules (e.g. `high_elderly_share`,
`poor_food_access`, `rich_transit_access`, `hawker_present`,
`diverse_food_environment` …) and runs **mlxtend's Apriori** with
configurable `min_support`, `min_confidence` and `min_lift`. A
**support sweep** repeats the run at five support levels to demonstrate
sensitivity. Outputs include the full rule table, the binary rules
definition, a scatter plot of support vs confidence coloured by lift, and
the sweep curve.

### 6.3 Phase 2 — Unsupervised clustering

`src/modelling/clustering.py` and `src/modelling/evaluation.py` jointly:

1. Standard-scale the engineered feature columns.
2. Sweep `k` from 2 to 8 for **K-Means** and **Agglomerative (Ward)**, and
   sweep `min_samples` for **DBSCAN** with an elbow-based eps.
3. Score every configuration on **silhouette**, **Davies-Bouldin** and
   **Calinski-Harabasz**.
4. Pick the best configuration by mean rank across the three metrics.
5. Run a **bootstrap ARI stability** analysis on the chosen fit.
6. Produce a **cluster profile table**, a **cluster size** chart, a **PCA
   scatter**, a **dendrogram**, the **DBSCAN k-distance** elbow and a
   **cluster profile heatmap**, plus a choropleth map of cluster labels.

### 6.4 Phase 3 — Risk model

`src/modelling/target.py` builds a **transparent proxy target**: a weighted
sum of standardised vulnerability components (positive sign on
`elderly_pct`, `elderly_density_log` and `nearest_food_km`; negative sign on
`food_access_score`, `transit_access_score` and `accessibility_support_score`).
The exact weights and reasons live in the `risk_model.components` block of
`config/settings.yaml` and the resolved formula is also written to
`outputs/tables/vulnerability_target_definition.txt`. The **top quartile**
(`high_risk_quantile: 0.75`) becomes the binary "high-risk" label.

`src/modelling/risk_model.py` then trains four models on the same
train/test split with stratified 5-fold CV: **Random Forest** (the headline
model), **Logistic Regression** (linear baseline), **Decision Tree** at
depth 4 (interpretable baseline) and **DummyClassifier** (sanity floor).
Outputs include the full metric table, the feature importance bar chart and
a model-vs-model comparison plot.

> The risk model is presented openly as a *proxy* learning task. Because the
> label is a function of the features, the goal is **not** out-of-sample
> generalisation in the traditional sense, it is to verify that the
> Random Forest can recover the proxy structure and to surface which features
> matter most.

## 7. Expected outputs

After `python run.py` the following directories are populated:

- `outputs/figures/` — 28 PNG figures including
  `map_overview.png`, `map_elderly_share.png`, `map_elderly_density.png`,
  `map_food_amenities.png`, `map_supermarkets.png`, `map_markets.png`,
  `map_transit_score.png`, `map_accessibility_score.png`,
  `map_residential_filter.png`, `map_clusters.png`,
  `map_vulnerability_score.png`, `map_hotspots.png`,
  `feature_distributions.png`, `feature_correlations.png`,
  `binary_amenity_presence.png`, `apriori_rules_scatter.png`,
  `apriori_support_sweep.png`, `clustering_metric_sweep.png`,
  `kmeans_silhouette_vs_k.png`, `dbscan_kdistance.png`,
  `hierarchical_dendrogram.png`, `cluster_pca_scatter.png`,
  `cluster_sizes.png`, `cluster_profile_heatmap.png`,
  `vulnerability_score_distribution.png`, `elderly_vs_food_scatter.png`,
  `risk_model_comparison.png`, `rf_feature_importance.png`.
- `outputs/tables/` — `apriori_rules.csv`, `apriori_support_sweep.csv`,
  `binary_rules_definition.csv`, `clustering_metrics.csv`,
  `final_clustering_metrics.json`, `cluster_profiles.csv`,
  `cluster_labels.json`, `risk_model_metrics.csv`,
  `risk_model_feature_importance.csv`, `subzone_vulnerability_scores.csv`,
  `vulnerability_target_components.csv`,
  `vulnerability_target_definition.txt`, `hotspots_top_n.csv`,
  `run_summary.json`.
- `data/processed/subzone_features.csv` and `subzone_features.geojson` —
  the master feature matrix joined to subzone geometry.
- `data/interim/*.geojson` — cached cleaned intermediates.
- `outputs/logs/pipeline.log` — full structured log of the run.

## 8. Reproducibility

- Python 3.11+; dependencies pinned in `requirements.txt` and mirrored in
  `environment.yml`. The Apriori implementation is `mlxtend>=0.23`.
- The random seed (default `42`) is applied globally via
  `src/utils/seed.py`, and explicitly to scikit-learn estimators.
- All paths in `config/settings.yaml` are resolved relative to the project
  root, so the pipeline behaves identically regardless of working directory.
- The pipeline is idempotent: re-running `python run.py` overwrites the
  same artefacts byte-for-byte.

## 9. Configuration knobs

All tunable parameters live in `config/settings.yaml`:

- `features.food_access.buffer_km` — radius for hawker / supermarket / market
  counts (default 1.0 km).
- `features.transit.bus_buffer_km`, `features.transit.mrt_buffer_km` —
  transit catchment radii.
- `features.accessibility.friendly_buffer_km` — barrier-free buffer.
- `features.land_use.min_residential_share`,
  `features.land_use.min_residential_area_m2` — residential filter.
- `binary_thresholds.*` — quantile cut-offs for the Apriori binary items.
- `mining.apriori_min_support`, `min_confidence`, `min_lift`, `max_len`,
  `support_sweep` — Apriori parameters.
- `clustering.feature_columns`, `k_range`, `algorithms`,
  `kmeans_n_init`, `dbscan_min_samples_list`, `final_algorithm`, `final_k`.
- `risk_model.components` (each with feature, direction, weight, reason),
  `high_risk_quantile`, `test_size`, `rf_*`, `cv_folds`.

## 10. Troubleshooting

- **`ModuleNotFoundError: mlxtend`** → run `pip install mlxtend`.
- **`KeyError` from a loader** → re-check that the file in `data/raw/` matches
  the filename in `raw_files:` in `config/settings.yaml`. The pipeline
  validates raw files up-front and exits with a clear message listing what
  is missing.
- **Empty / collapsed DBSCAN cluster** → expected on this dataset; DBSCAN is
  retained for diagnostics only and the metric sweep automatically picks
  K-Means as the best partition.
- **Master Plan layer slow to load** — `MasterPlan2025LandUseLayer.geojson`
  is ~190 MB; first run takes longer because the cleaned subset is then
  cached to `data/interim/`.
- **GeoPandas install fails on Windows** → use the conda environment
  (`environment.yml`); `geopandas`, `shapely` and `pyproj` install cleanly
  via conda-forge.
