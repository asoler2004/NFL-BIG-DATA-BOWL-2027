# NFL Big Data Bowl 2027 Repository Roadmap

The **NFL Big Data Bowl 2027 Repository (`asoler2004/NFL-BIG-DATA-BOWL-2027`)** establishes an end-to-end analytical framework designed to evaluate how 10 Hz Scouting Combine optical tracking sensor data translates into regular-season NFL game performance and long-term career outcomes. Built upon a PostgreSQL database foundation, a Streamlit interactive dashboard (`app2.py`), and a Semantic Web RDF knowledge graph (`nfl_graph.ttl` via `PostgresToJena.py`), this repository bridges pre-draft physical traits with micro-level on-field execution across 510 rookie prospects from the 2023, 2024, and 2025 draft classes.

This document outlines the strategic engineering roadmap and planned enhancements to prepare the repository for production deployment and competition submission under the Apache 2.0 License.

---

## 1. Vision & Core Architecture Strategic Roadmap

The primary objective of the repository is to provide scout-accessible, statistically rigorous tools that isolate high-frequency kinematic movement signatures—such as burst acceleration, change-of-direction (COD) efficiency, spectral power distribution, and derivative jerk—and map them directly to regular-season performance metrics like receiver separation, pass-rusher get-off speed, and pass-protection duration.

The roadmap is structured into five sequential execution phases, moving from low-level signal processing to semantic graph queries, cross-validated modeling, and interactive visualization.

| Phase | Core Objective | Primary Modules | Target Timeline |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Kinematic Signal Processing & Feature Extraction | `nfl_data.py`, `test_movement_translation.py` | Weeks 1–3 |
| **Phase 2** | Knowledge Graph & SPARQL Semantic Query Engine | `PostgresToJena.py`, `nfl_graph.ttl` | Weeks 4–6 |
| **Phase 3** | Predictive Modeling, FDR Control & Causal Inference | `ntbk.ipynb`, `nfl-big-data-bowl-2027.ipynb` | Weeks 7–9 |
| **Phase 4** | Streamlit UI/UX & Real-Time Scouting Dashboard | `app.py`, `app2.py` | Weeks 10–12 |
| **Phase 5** | Competition Artifact & Kaggle Submission | `README.md`, Public Notebook | Milestone Final |

---

## 2. Phase 1: Kinematic Signal Processing & Feature Engineering

Phase 1 focuses on expanding the high-frequency 10 Hz sensor processing pipeline within `nfl_data.py` and `test_movement_translation.py`. While current tracking features capture baseline velocity ($s$), peak acceleration ($a$), and derivative jerk ($j = \frac{da}{dt}$ in $\text{yd/s}^3$), raw optical tracking frames require enhanced filtering and biomechanical feature isolation.

### Planned Signal Processing Enhancements
* **Butterworth & Savitzky-Golay Filtering**: Implement low-pass digital filtering on raw $x, y$ coordinate streams to remove optical sensor noise prior to calculating numerical derivatives for velocity, acceleration, and jerk.
* **Planar Acceleration Vector Decomposition**: Decompose scalar acceleration into longitudinal burst acceleration ($a_{\parallel}$) along the motion vector and lateral cutting acceleration ($a_{\perp}$) during directional changes in 3-cone and short shuttle agility drills.
* **Spectral Density & Fast Fourier Transform (FFT) Features**: Expand upon the linearly detrended periodogram in `test_movement_translation.py` by extracting dominant frequency bands and spectral entropy to quantify movement smoothness and stride frequency consistency.
* **Curvature & Change-of-Direction (COD) Radius**: Compute instantaneous path curvature $\kappa = \frac{|\dot{x}\ddot{y} - \dot{y}\ddot{x}|}{(\dot{x}^2 + \dot{y}^2)^{3/2}}$ during route breaks and agility cuts to measure tight-turn efficiency.

---

## 3. Phase 2: Knowledge Graph & Semantic SPARQL Query Engine

The repository includes a relational-to-graph pipeline (`PostgresToJena.py`) that exports PostgreSQL tables (`create_postgres_tables.sql`) into an RDF Turtle graph (`nfl_graph.ttl`) for Apache Jena. Phase 2 upgrades this semantic architecture to enable complex multi-relational queries across player demographics, Combine sensor metrics, and play-by-play game tracking.

### Planned Knowledge Graph Enhancements
* **Ontology Extension (NFL-OWL)**: Expand the RDF schema to map domain-specific relationships, such as `:playedInGame`, `:executedRoute`, `:encounteredPassRusher`, and `:alignedPosition`.
* **SPARQL Endpoint Integration**: Embed an in-memory Apache Jena Fuseki or rdflib SPARQL query engine within `app2.py`, allowing users to execute graph queries directly from the Streamlit sidebar.
* **Semantic Trajectory Alignment**: Link Combine optical tracking event frames directly to in-game tracking event milestones (`ball_snap`, `pass_forward`, `first_contact`, `tackle`) using standardized time-ontology predicates.

---

## 4. Phase 3: Machine Learning Framework & Statistical Validation

Phase 3 builds upon the "Model & Hidden Gems" and "Relationship Finder" tabs in `app2.py` to deliver robust predictive models linking Combine kinematics to NFL performance while controlling for draft capital and multiple testing errors.

### Planned Modeling Enhancements
* **Draft Capital Partialling & Residualization**: Expand linear and non-linear partialing of draft overall pick (`draft_overall_pick` / `log_pick`) to isolate true athletic value from draft investment bias.
* **Benjamini-Hochberg False Discovery Rate (FDR) Tuning**: Implement adaptive FDR thresholds across pairwise correlation scans to prevent false positives when testing high-dimensional feature spaces.
* **Cross-Validated ElasticNet & XGBoost Benchmarking**: Implement a standardized 5-fold cross-validation framework evaluating three feature sets:
  1. *Baseline Model*: Traditional Combine testing splits (`forty`, `ten_yd_split`, `three_cone`, `short_shuttle`, `vertical`, `broad_jump`).
  2. *Kinematic Model*: 10 Hz sensor metrics (`trk40_vmax`, `trk40_amax`, peak jerk, spectral entropy, tracking delta).
  3. *Hybrid Model*: Anthropometrics (`bmi`, `ape_index`), draft capital, traditional splits, and sensor kinematics.
* **Sleeper / Over-Performer Identification**: Automate the detection of prospects whose model-predicted in-game success significantly exceeds their actual draft position.

---

## 5. Phase 4: Streamlit UI/UX & Real-Time Scouting Dashboard

Phase 4 transforms `app2.py` into a production-grade scouting workstation featuring customizable layouts, interactive tracking overlays, and exportable scouting reports.

### Planned Dashboard Enhancements
* **Multi-Panel Layout Builder**: Enhance JSON layout saving and loading to allow scouts to export and import custom dashboard configurations.
* **Synchronized 2D Field Trajectory Viewer**: Build interactive Plotly field visualizers that animate 10 Hz Combine drill runs side-by-side with regular-season NFL play tracking.
* **Position Cohort Deep Dives**: Create automated profile cards for position groups (e.g., Wide Receivers, Edge Rushers, Offensive Tackles) matching Combine kinematic signatures against position-specific in-game metrics (e.g., `separation_at_pass_forward`, `player_get_off`, `time_to_pressure_allowed`).

---

## 6. Phase 5: Competition Submission & Kaggle Protocol

To ensure full compliance with the official **NFL Big Data Bowl 2027** submission guidelines, the final milestone focuses on packaging the research writeup and code pipeline.

### Competition Compliance Guidelines
* **Word & Figure Limits**: Ensure the final executive writeup contains under 2,000 words and fewer than 10 tables/figures.
* **Evaluation Alignment**: Structure deliverables to maximize scoring across all four judging criteria:
  * *Football Score (30%)*: Actionable scouting metrics applicable for team evaluation.
  * *Data Science Score (30%)*: Correct statistical modeling, cross-validation, and FDR controls.
  * *Writeup Score (20%)*: Clear motivation, narrative flow, and structured findings.
  * *Data Visualization Score (20%)*: Accessible, accurate, and innovative interactive charts.
* **Code Reproducibility**: Package all data processing scripts (`nfl_data.py`), model pipelines, and visualization modules into a clean, reproducible public Kaggle notebook.

---

## 7. Development Guidelines & Contribution Workflow

Contributions to the repository are welcomed under the **Apache License 2.0**.

### Contribution Workflow
1. **Fork & Branch**: Create a feature branch (`git checkout -b feature/kinematic-filter`).
2. **Environment Setup**: Install dependencies using `pip install -r requirements.txt`.
3. **Database Configuration**: Set up PostgreSQL connection credentials in `.streamlit/secrets.toml` or environment variables (`PGHOST`, `PGPORT`, `PGDBNAME`, `PGUSER`, `PGPASSWORD`).
4. **License Notice**: Retain the standard Apache 2.0 license notice at the top of all new Python modules and documentation files.
