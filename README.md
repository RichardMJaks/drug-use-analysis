# drug-use-analysis

Welcome to the **Drug Use & Personality Analysis** repository.

This project investigates how **personality traits**, **demographics**, and **behavioral patterns** relate to **self-reported substance use**. The work combines data cleaning, statistical analysis, visualization, and clustering to uncover meaningful patterns in how different groups consume various drugs.

---

## 📊 Dataset

**Drug Consumption Classification**  
Size: **345.63 KB**  
Samples: **1,880**  
Features: **32** (Big Five personality scores, demographics, and consumption records for 18 substances).  
Source: Kaggle  
https://www.kaggle.com/datasets/mexwell/drug-consumption-classification/data

---

## 🎯 Project Goals

**Goal 1: Data Preparation**  
Clean, preprocess, and transform the raw dataset for reliable analysis.

**Goal 2: Analytical Insights**  
Explore relationships between:  
- Personality traits (Big Five)  
- Demographics  
- Drug-use frequency and categories  
Additionally, cluster individuals based on behavioral similarity and investigate cross-drug usage patterns.

**Goal 3: Visualization & Interpretation**  
Create clear, interpretable plots and derive meaningful findings about predictors and patterns of drug use.

---

## 👥 Team

- **Hannes Jaakson** — GitHub: [HannesJaakson](https://github.com/HannesJaakson)  
- **Richard Miikael Jaks** — GitHub: [RichardMJaks](https://github.com/RichardMJaks)

---

## 📂 Project Structure

**`main.ipynb`**  
Primary notebook containing the exploratory analysis, modeling, and visualizations.

**`main.py`**  
Automatically generated from the notebook on Richard’s machine (bidirectional conversion: `.ipynb` ↔ `.py`).

**`data_analysis.py`**  
Helper functions for data processing, cleaning, transformations, and statistical calculations.

**`data_reading.py`**  
Contains lists of trait/drug variables and four utility functions for reading, converting, and preparing dataset columns.

---

## 🧭 How to Use

1. Clone the repository:  
   ```bash
   git clone https://github.com/HannesJaakson/drug-use-analysis
2. Install dependencies:
   pip install -r requirements.txt
3. Open main.ipynb to explore the full analysis.

## 📌 License
This project is for educational and research purposes only.
Please use responsibly.
