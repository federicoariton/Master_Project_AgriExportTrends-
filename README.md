# Agriculture Data Analysis and Forecasting Project

### Overview

This project aims to analyze and forecast key trends in Ireland's agricultural sector by leveraging data science techniques. It provides insights into production and export dynamics, trade balances, and pricing trends through statistical analysis, predictive modeling, and interactive data visualizations. An interactive dashboard enables stakeholders to explore and act on these insights effectively.

## Additional Resources

- More Information: For a detailed explanation of the project structure, methodologies, and findings, refer to the Full Project Report.[Full Project Report](https://github.com/federicoariton/Master_Project_AgriExportTrends-/Federico_Ariton_sba22090_Lvl9_CA2_Integrated_Report.docx)


- Interactive Experience: Explore the data, predictions, and visualizations through the [Interactive Dashboard](https://github.com/federicoariton/Interactive_dasbhboard_Streamlit.git)


## Objectives

- Analyze agricultural production and export trends in Ireland and other countries.

- Identify key drivers of export values and forecast future trends.

- Conduct sentiment analysis to capture consumer and producer perspectives.

- Develop an interactive dashboard for decision-making and forecasting.

## Methodology

The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

- Business Understanding: Define objectives and challenges in Ireland's agricultural trade.

- Data Understanding: Gather and preprocess datasets from the FAO database.

- Data Preparation: Clean, reshape, and merge datasets to create a master dataset.

- Modeling: Apply statistical methods and machine learning models to derive insights.

- Evaluation: Compare model performances and validate results.

- Deployment: Present findings via an interactive dashboard.

  
![Structure of the project](https://github.com/user-attachments/assets/1828878b-3ddf-483a-9a95-cea62225c92c)

## Features

### Data Preparation:

- Filtering datasets from 2000 onward for relevance.

- Addressing missing values using imputation techniques.

- Reshaping datasets for seamless analysis.

### Statistical Analysis and EDA:

- Hypothesis testing to identify trends and relationships.

- Visualization of production and export metrics.

### Predictive Modeling:

- Models: Linear Regression, Ridge Regression, and Random Forest.

- Forecasting export values using SARIMA models.

### Sentiment Analysis:

- Analyzing Reddit discussions for insights into milk and meat production challenges.

### Interactive Dashboard:

- Features include real-time predictions, visualizations, and 20-year forecasts.

- Export and import trade balance visualizations.

## Tools and Technologies

- Languages: Python

- Libraries: Pandas, NumPy, Plotly, Scikit-learn, Statsmodels, SciPy

- Platform: Jupyter Notebooks

- Dashboard Framework: Dash by Plotly

## Datasets

#### Source: FAO Database

### Key Datasets Used:

- Value of Agricultural Production

- Trade (Crops and Livestock)

- Production (Crops and Livestock)

- Price Indices

## Results and Insights

- Export and Import Trends: Stable trade dynamics with no significant differences between export and import values.

- Production vs. Export: Strong positive correlation, indicating higher production drives exports.

- Pricing Dynamics: Significant variability in producer prices across items and regions.

- Predictive Accuracy: Linear Regression achieved an R² score of 0.96, while Random Forest reached 0.9783 with hyperparameter tuning.



