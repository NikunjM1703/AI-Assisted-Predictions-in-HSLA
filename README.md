#AI-Assisted Prediction of Stacking Fault Energy (SFE) in HSLA Steels

## 📘 Overview
-This project explores how **machine learning** can accelerate material design by predicting the **Stacking Fault Energy (SFE)** of High-Strength Low-Alloy (HSLA) steels directly from their chemical composition and simulation data.  
-Developed as part of my **Undergraduate Project (UGP)** at IIT Kanpur, it demonstrates how AI models can complement computational materials science to reduce dependency on costly experiments.

## 🔍 Key Features
- Compiled dataset of **~474 HSLA compositions** from literature and CALPHAD simulations.  
- Feature engineering with temperature, phase, and composition normalization.  
- Built and compared multiple regression models (Linear, SVR, Random Forest, Gradient Boosting, Extra Trees).  
- Evaluated performance with R², RMSE, and cross-validation metrics.

## 🧰 Tools & Tech Stack
- **Languages:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib  
- **Environment:** Jupyter Notebook  


## 📊 Dataset / Input
- Raw data collected from public metallurgical datasets and CALPHAD-based calculations.  
- Cleaned and normalized dataset included in `/data/processed/` (synthetic or masked sample if real data restricted).  
- Features: atomic %, temperature (K), phase stability parameters.

