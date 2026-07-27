# WATTLINE | Energy Load Predictor

An interactive machine learning dashboard designed as a comparative analytical lab to predict a building's energy consumption based on its structural and environmental specifications.

## Project Overview

Relying on pre-built machine learning libraries gets the job done, but building algorithms from scratch demonstrates true mathematical and engineering depth. This project evaluates a custom-built machine learning algorithm against an industry-standard framework.

Instead of solely using existing packages, I engineered a **Linear Regression model with Gradient Descent entirely from scratch** using NumPy. The algorithm manually calculates cost history and optimizes weights iteratively. To validate its structural accuracy, this foundational model is benchmarked directly against a `scikit-learn` model, comparing their MSE, MAE, and R-squared metrics side-by-side.

## Key Architectural Insights

1. **Deterministic Validation (R-squared = 1.0):** 
   The perfect evaluation score is not an overfitting error; it is a controlled test. A deterministic dataset was utilized to verify the raw mathematical precision of the custom Gradient Descent algorithm. Achieving absolute parity with Scikit-Learn proves the algorithm is mathematically sound before being exposed to noisy, real-world datasets.
   
2. **Evaluation vs. Inference Boundary:** 
   The architecture strictly separates model evaluation from real-time inference. Feeding new parameters into the prediction engine computes a real-time output for a single instance without altering the historical analytics metrics, mirroring robust production-level ML pipelines.

## Repository Structure

* `app.py`: The main Streamlit application containing the UI, prediction engine, and visual analytics.
* `LinearRegression.ipynb`: The Jupyter Notebook containing the data exploration, algorithm training from scratch, and automated model serialization (`joblib`).
* `train_energy_data.csv` & `test_energy_data.csv`: The deterministic datasets used for training and validation.

## Tech Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Custom Gradient Descent Algorithm
* **Frontend UI:** Streamlit
* **Data Visualization:** Plotly
* **Persistence:** Joblib

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Wattline-Energy-Predictor.git](https://github.com/Maria-Esam/Wattline-Energy-Predictor.git)
   
## Dashboard Preview

![Prediction Engine View](<img width="1899" height="892" alt="Screenshot 2026-07-27 162654" src="https://github.com/user-attachments/assets/ca2fb70f-2d12-4de4-8252-f24dd991a6d1" />
)

![Model Analytics View](<img width="1910" height="907" alt="Screenshot 2026-07-27 162738" src="https://github.com/user-attachments/assets/65cbf2ef-2122-42b2-9e5e-7c14f73eaaa6" />
)
