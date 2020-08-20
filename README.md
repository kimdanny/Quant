# Quant
Simple Stock Price Forecasting using FinanceData

## Package Structure
Below illustrates the file structure after every auto-generated file and folder is created.

```
.Quant
+-- Crawler
|   +-- crawled_result
|       +-- [stock_code]
|           +-- News
|               +-- fullPage
|                   +-- pages[from-to].csv
|               +-- page[page].csv
|           +-- Research
|               +-- fullPage
|                   +-- pages[from-to].csv
|               +-- page[page].csv
|   +-- __init__.py
|   +-- Google_Crawler.py
|   +-- Naver_Crawler.py
|
+-- FinanceData
|   +-- __init__.py
|   +-- FinanceDataCollection.py
|
+-- Model
|   +-- [stock_code]_final_data
|       +-- image.png
|   +-- [stock_code]_plots
|       +-- image.png
|   +-- Feature_Engineering
|       +-- image.png
|   +-- model_checkpoints
|       +-- [model_name].h5
|   +-- __init__.py
|   +-- data_config.py
|   +-- Data_preparation.py
|   +-- FeatureEngineering.py
|   +-- Gradient_Boosting.py
|   +-- LSTM_Model_.py
|   +-- Multivariate_LSTM.py
|   +-- Multivariage_supervised_LSTM.py
|   +-- Prophet_Model.py
|
+-- NLP
|   +-- __init__.py
|   +-- config.py
|   +-- GCP_Language.py
|   +-- Saltlux_Language.py
|   +-- entities[id].json
|   +-- sentiment[id].json
|   +-- keyword[id].json
|
+-- __init__.py
+-- [your_GCP_credential].json
+-- Predict.py
```

## How to Run
will be updated

## Visualisation
**Samsung Open, High, Low, Close Stock Price Prediction (LSTM)**  
![Samsung_4_stock_prices](Model/005930_plots/AllInOne.png)

**Samsung Golden and Dead Cross based on Moving Average**  
![Samsung_Golden_Dead_Cross](Model/005930_plots/MA_Golden_Cross.png)

**Fourier Transforms on Samsung stock price**  
![Samsung_4_stock_prices](Model/005930_plots/Fourier_Transforms.png)

**Feature Importance Test**  
![Samsung_4_stock_prices](Model/Feature_Engineering/Close_x_Feature_Importance.png)

**Facebook Prophet's prediction on Samsung Close price**  
![Samsung_4_stock_prices](Model/005930_plots/prophet_pred.png)

**Facebook Prophet's prediction on ARKQ ETF**  
![Samsung_4_stock_prices](Model/ARKQ_plots/prophet_pred.png)

**Facebook Prophet's prediction on Samsung Close price**  
![Samsung_4_stock_prices](Model/005930_plots/prophet_pred.png)







