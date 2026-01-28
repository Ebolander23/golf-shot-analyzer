# ⛳ Golf Shot Performance Analyzer

**A Python-based shot analysis tool for processing launch monitor data, performing statistical analysis, and generating interactive visualizations.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://golf-analyzer.streamlit.app)

---

## 🎯 Project Overview

This project demonstrates the full data science workflow for golf equipment testing and player fitting:

1. **Data Ingestion** — Load and clean Trackman launch monitor exports
2. **Statistical Analysis** — ANOVA, regression, consistency metrics
3. **Visualization** — Dispersion plots, correlation heatmaps, distribution charts
4. **Automated Insights** — Generate human-readable performance summaries
5. **Interactive Dashboard** — Streamlit app for exploration and filtering

### Why This Project?

Built as a portfolio project for the **TaylorMade Performance Research Engineer** role, demonstrating:

- ✅ **Python proficiency** (pandas, numpy, scipy, plotly)
- ✅ **Advanced statistics** (ANOVA, regression modeling, effect sizes)
- ✅ **Automated data pipelines** (data cleaning, outlier detection)
- ✅ **Interactive dashboards** (Streamlit deployment)
- ✅ **Golf domain expertise** (+1.8 USGA handicap, active TaylorMade tester)

---

## 📊 Features

### Data Processing
- Load Trackman CSV exports with automatic column mapping
- Physics-based data cleaning (smash factor limits, spin rate validation)
- Skill level segmentation by ball speed:
  - Elite (145+ mph): Tour/elite amateur level
  - Scratch (130-145 mph): Low single-digit handicaps
  - Mid (110-130 mph): 8-18 handicap range
  - High (<110 mph): Beginners, seniors, juniors

### Statistical Analysis
- **Consistency Metrics**: Carry std dev, dispersion, spin consistency, smash factor variance
- **Performance Metrics**: Ball speed, carry, total distance, launch angle, spin rate
- **ANOVA**: Compare metrics across skill levels with effect size (η²)
- **Regression**: Predict carry from launch conditions (ball speed, launch angle, spin rate)
- **Dispersion Ellipse**: Calculate 68% confidence ellipse for shot patterns

### Visualizations
- Shot dispersion plots with ellipse overlay
- Box plots comparing skill levels
- Correlation heatmaps
- Regression scatter with fit line
- Multi-skill overlay dispersion

### Automated Insights
- Performance comparisons to benchmarks (Tour, Scratch, Amateur)
- Optimal launch condition identification
- Equipment recommendations based on data patterns
- Skill level comparison insights

---

## 🚀 Quick Start

### Live Demo
👉 **[Launch the App](https://golf-analyzer.streamlit.app)**

### Local Installation

```bash
# clone the repository
git clone https://github.com/ericbolander/golf-shot-analyzer.git
cd golf-shot-analyzer

# install dependencies
pip install -r requirements.txt

# run the streamlit app
streamlit run app.py
```

### Using the Modules Directly

```python
from src.data_loader import load_data, clean_data, segment_by_skill
from src.analysis import calculate_consistency_metrics, run_anova_analysis
from src.visualizations import create_dispersion_plot

# load and clean data
raw_df = load_data('data/sample_data.csv')
clean_df, removed_df = clean_data(raw_df)

# segment by skill level
segments = segment_by_skill(clean_df)

# analyze elite players
elite_metrics = calculate_consistency_metrics(segments['elite'])
print(f"Elite dispersion: ±{elite_metrics.offline_std:.1f} yards")

# run anova comparing carry across skill levels
anova_result = run_anova_analysis(segments, 'carry')
print(f"ANOVA p-value: {anova_result['p_value_formatted']}")
```

---

## 📁 Project Structure

```
golf-shot-analyzer/
├── app.py                   # streamlit dashboard
├── src/
│   ├── data_loader.py       # data ingestion & cleaning
│   ├── analysis.py          # statistical calculations
│   ├── visualizations.py    # plotly visualizations
│   └── insights.py          # auto-generated findings
├── data/
│   └── sample_data.csv      # sample trackman export
├── requirements.txt
├── README.md
└── docs/
    └── methodology.md       # detailed methodology notes
```

---

## 📈 Key Findings from Sample Data

Analyzing ~10,000 driver shots across skill levels:

| Metric | Elite | Scratch | Mid HCP | High HCP |
|--------|-------|---------|---------|----------|
| Ball Speed (mph) | 154.6 | 138.0 | 121.3 | 81.2 |
| Carry (yds) | 245.1 | 201.9 | 159.5 | 85.1 |
| Spin (rpm) | 2,788 | 3,212 | 3,760 | 3,996 |
| Dispersion (±yds) | 24.6 | 26.4 | 24.2 | 14.5 |

**Key Insight**: High handicappers show tighter dispersion because shorter shots have less time to curve offline — an important consideration for equipment design.

---

## 🔧 Data Quality Filters

The cleaning pipeline applies physics-based filters:

| Filter | Threshold | Reasoning |
|--------|-----------|-----------|
| Smash Factor Max | 1.55 | USGA COR limit is ~1.50 |
| Smash Factor Min | 0.8 | Below indicates severe mishit |
| Spin Rate Max | 10,000 rpm | Above is pop-up or data error |
| Spin Rate Min | 0 rpm | Negative = integer overflow |
| Launch Angle Max | 30° | Above is pop-up |
| Launch Angle Min | -5° | Below is measurement error |

---

## 🧮 Statistical Methods

### ANOVA (Analysis of Variance)
Tests whether performance differs significantly across groups.

```python
anova_result = run_anova_analysis(segments, 'carry')
# f_statistic: 6993.66
# p_value: < 0.001
# eta_squared: 0.70 (large effect)
```

### Multiple Linear Regression
Predicts carry distance from launch conditions.

```
carry = 2.07 × ball_speed + 4.66 × launch_angle - 0.003 × spin_rate - 123.1
```

**Interpretation**:
- +1 mph ball speed → +2.1 yards carry
- +1° launch angle → +4.7 yards (up to optimal)
- +100 rpm spin → -0.3 yards

---

## 👤 About the Author

**Eric Bolander**  
UCSD Bioinformatics '25 | +1.8 USGA Handicap | TaylorMade Player Tester

- 🎓 Computational science background with statistics focus
- ⛳ Elite amateur golfer, annual US Amateur qualifier
- 🔬 Currently participating in TaylorMade player testing
- 📹 YouTube: [One Take Golf](https://youtube.com/@onetakegolf)

**Contact**: [LinkedIn](https://linkedin.com/in/ericbolander) | [GitHub](https://github.com/ericbolander)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Data Source**: Public Trackman dataset from [tim-blackmore/launch-monitor-regression](https://github.com/tim-blackmore/launch-monitor-regression)
- **Inspiration**: TaylorMade Performance Research team and the opportunity to combine data science with golf
