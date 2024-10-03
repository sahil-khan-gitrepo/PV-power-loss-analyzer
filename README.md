# Photovoltaic Power Loss Analyzer

## Project Overview
The **Photovoltaic Power Loss Analyzer** is a machine learning project focused on analyzing the impact of dust accumulation on solar photovoltaic (PV) panel efficiency. Dust and other environmental factors can significantly reduce the power output of solar panels. This project aims to quantify the power loss due to dust and other environmental factors and to optimize the cleaning schedules of PV panels to improve their performance.

The analysis is based on real-world data from NSUT for solar PV power output, combined with environmental data from NASA. Using advanced data preprocessing techniques and machine learning models, we predict the power loss due to dust and provide insights to help improve solar panel efficiency.

## Table of Contents
1. [Project Motivation](#project-motivation)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Project Workflow](#project-workflow)
    - Data Preprocessing
    - Feature Engineering
    - Model Training
    - Evaluation
5. [Results](#results)
6. [Future Work](#future-work)
7. [Contributors](#contributors)
8. [References](#references)

## Project Motivation
The accumulation of dust on solar panels is a known factor that reduces their power output. For large-scale solar farms, dust-related losses can have a significant economic impact. This project seeks to address this problem by building a machine learning model that analyzes the relationship between environmental factors (like dust) and solar power output, with the goal of predicting when cleaning should be performed to optimize power generation.

By analyzing historical data and building predictive models, this project provides a tool that helps solar farm operators make data-driven decisions to maximize efficiency.

## Dataset

### 1. NSUT Dataset
- Contains real power output from PV panels located at the NSUT campus.
- Features include: timestamps, panel orientation, panel temperature, and power output.

### 2. NASA Dataset
- Contains environmental factors such as irradiance, air temperature, wind speed, and dust concentration.
- Data was collected for the same geographical location as the NSUT dataset.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - **Pandas**: For data cleaning and manipulation.
  - **NumPy**: For numerical calculations.
  - **Scikit-Learn**: For machine learning models and preprocessing.
  - **Matplotlib & Seaborn**: For data visualization.
  - **OpenCV**: For image analysis and visual understanding.
  - **Principal Component Analysis (PCA)**: For dimensionality reduction and feature engineering.
  
## Project Workflow

### 1. Data Preprocessing
- Cleaned the NSUT dataset by addressing **missing values**, ensuring data consistency and accuracy.
- Preprocessed the NASA Power dataset to extract relevant environmental features like irradiance, dust levels, and temperature.
- **Normalization**: Applied normalization to ensure that features had similar ranges, allowing for better model performance.
  
### 2. Feature Engineering
- Performed **Principal Component Analysis (PCA)** to reduce the dimensionality of the feature space and highlight the most important variables affecting PV performance.
- Created new features combining dust concentration with other environmental variables to better understand their combined effect on power output.
- ![image](https://github.com/user-attachments/assets/64662454-647a-4757-b726-9bc4d6e08205)

### 3. Model Training
- Built multiple regression models using **Random Forest** to predict PV power loss based on the input features.
- Trained the model twice:
  - **Without dust**: The model was trained on environmental factors excluding dust concentration, achieving an **R-square value of 0.924**.
  - **Including dust**: Dust was added to the feature set, which improved the prediction accuracy to an **R-square value of 0.933**.

### 4. Evaluation
- The model performance was evaluated using standard regression metrics, including **R-squared**, **Mean Absolute Error (MAE)**, and **Mean Squared Error (MSE)**.
- The addition of dust as a feature improved the overall model performance, showing its significant impact on power loss.


## Results
- The **Random Forest** model was able to predict solar power output with an R-square value of **0.933** when dust concentration was included in the feature set.
- The analysis highlighted that dust accumulation can lead to significant power losses, and regular cleaning schedules should be optimized based on environmental data.

### Key Metrics:

| Model                | R-Squared Value |
|----------------------|-----------------|
| Without Dust         | 0.924           |
| Including Dust       | 0.933           |

The analysis suggests that regular cleaning of solar panels based on environmental factors, especially dust levels, can enhance power output and optimize operational costs.

## Future Work
- **Expand Dataset**: Collect more extensive data from different geographical locations to improve model generalizability.
- **Explore Deep Learning Models**: Investigate the use of **deep learning** models such as CNNs or RNNs for time-series analysis of solar PV data.
- **Real-time Deployment**: Develop a real-time monitoring system to predict power loss and trigger cleaning schedules based on live environmental data.
  
## Contributors
- **Sahil** and **Rohan** - Data Collection, preprocessing and model development.
-  **Jatin**, **Pankaj** - Research paper writing and presentation.


## References
[1] A smart short-term solar power output prediction by artificial neural network Ali Erduman :(https://link.springer.com/article/10.1007/s00202-020-00971-2).
[2] Prediction of a Grid-Connected Photovoltaic Park’s Output with Artificial Neural Networks Trained by Actual Performance Data
Elias Roumpakias and Tassos Stamatelos , (https://www.mdpi.com/2076-3417/12/13/6458)
[3] Investigation of performance reduction of PV system due to environmental dust: Indoor and real-time analysis, : 
https://www.sciencedirect.com/science/article/pii/S2772671124002377
[4] A Review of Machine Learning-Based Photovoltaic Output Power Forecasting: Nordic Context, 
https://ieeexplore.ieee.org/document/9729194
[5]Power loss due to soiling on photovoltaic module with and without anti-soiling coating at different angle of incidence Bushra
Mahnoor, Muhammad Noman, Muhammad Saad Rehan & Adnan Daud Khan List of issues International Journal of Green Energy (tandfonline.com)
[6] Prediction of Photovoltaic Power by ANN Based on Various Environmental Factors in India 
,https://onlinelibrary.wiley.com/doi/10.1155/2022/4905980
[7] Photovoltaic Power Prediction Using Analytical Models and Homer-Pro: Investigation of Results Reliability, 
https://www.mdpi.com/2071-1050/15/11/8904
[8] Weather and Pollution dataset from https://power.larc.nasa.gov/data-access-viewer/ & https://airquality.cpcb.gov.in/ccr/#/logi
