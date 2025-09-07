# indian-crop-yield-analysis
# 🌾 Agricultural Crop Yield in Indian States (2010–2017)

## 📘 Overview
This dataset contains detailed agricultural statistics across Indian districts and states from 2010 to 2017. It includes crop-wise data on area cultivated, production volume, and yield per hectare for major crops such as rice, wheat, maize, pulses, oilseeds, and more.

The dataset is intended to support research and analysis in sustainable agriculture, food security, and regional crop performance. It can be used for trend analysis, visualization, policy modeling, and machine learning applications.

---

## 📁 Dataset Details

- **File Name**: `Crops_data.csv`
- **Rows**: 2,484
- **Columns**: 80
- **Format**: CSV
- **License**: CC0 (Public Domain)

### 🔑 Key Columns
- `Year`: Year of data collection
- `State Name`, `Dist Name`: Geographic identifiers
- `Crop AREA (1000 ha)`: Area cultivated for each crop
- `Crop PRODUCTION (1000 tons)`: Total production
- `Crop YIELD (Kg per ha)`: Yield per hectare

---

## 📊 Use Cases
- Crop yield trend analysis
- Regional performance comparison
- Sustainable agriculture modeling
- Data visualization and storytelling
- ML model training for agri-tech applications

---

## 🧠 Source
Originally compiled for educational use in the IBM SkillsBuild program and Kaggle-based sustainability projects.
---

## 📅 Weekly Progress Summary

### 🟩 Week 1 – Data Exploration & Problem Understanding

In the first week, I focused on understanding the scope of the crop yield prediction project and familiarizing myself with the dataset. Key activities included:

- ✅ Loaded and explored `Crops_data.csv` containing 2,484 rows and 80 columns
- ✅ Identified key features such as crop area, production, and yield per hectare
- ✅ Performed initial data inspection using `pandas` and Jupyter Notebook
- ✅ Visualized rice yield trends over time using `matplotlib`
- ✅ Pushed the dataset to GitHub and created the project repository
- ✅ Drafted a clean README to document the dataset and use cases

This phase helped clarify the problem statement and set the foundation for modeling and dashboard development.

---

### 🟨 Week 2 – Model Building & Dashboard Development

Week 2 focused on preparing the data for machine learning and building a regression model to predict crop yields. Key accomplishments:

- ✅ Cleaned and preprocessed the dataset for modeling
- ✅ Selected `LinearRegression` as the baseline model
- ✅ Performed train-test split (`X_train`, `X_test`, `y_train`, `y_test`)
- ✅ Trained the model and generated predictions
- ✅ Created `predictions.csv` comparing actual vs predicted yields
- ✅ Saved model artifacts including `rice_yield_model.pkl` and split datasets
- ✅ Built an interactive dashboard using **Dash** and **Plotly Express**
- ✅ Visualized model performance with scatter plots and trendlines

All model-related files are stored in the `/model_files` folder for reproducibility and future evaluation.

---



---

## 🤝 Contributions
Feel free to fork, analyze, or build on this dataset. If you use it in a project or publication, a mention or link back is appreciated!

