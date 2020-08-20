import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FinanceDataReader import DataReader
from fbprophet import Prophet
import os
from pathlib import Path


class Prophet_Model:
    def __init__(self, company_code='005930', target='Close', _from=None, data_path=None,
                 predict_period=365, daily_seasonality=True
                 ):
        if data_path is not None:
            data = pd.read_csv(data_path)
            data = data[['Date', 'Close_x']]
            # Rename the features: These names are NEEDED for the model fitting
            data.columns = ['ds', 'y']
            self.data = data
        else:
            data = DataReader(company_code, _from)
            data = data[[target]]
            data = data.reset_index()
            data.columns = ['ds', 'y']
            self.data = data

        self.company_code = company_code
        self.target_col = target

        # we need to specify the number of days in future
        self.predict_period = predict_period
        self.model = Prophet(daily_seasonality=daily_seasonality)

        # Path Handling for plots
        plot_dirname = self.company_code + '_plots'
        root_dir = os.path.dirname(__file__)
        self.plots_dir_path = os.path.join(root_dir, plot_dirname)

    def fit_and_predict(self):
        self.model.fit(self.data)  # fit the model using all data
        future = self.model.make_future_dataframe(periods=self.predict_period)
        prediction = self.model.predict(future)

        return prediction

    def plot_prediction(self, prediction_df):
        Path(self.plots_dir_path).mkdir(parents=True, exist_ok=True)

        self.model.plot(prediction_df)
        plt.title(f"Prediction of the {self.company_code} Stock Price using the Prophet")
        plt.xlabel("Trading Day")
        plt.ylabel(f"{self.target_col} Stock Price")
        plt.savefig('./' + self.company_code + '_plots/prophet_pred.png')

    def plot_pred_details(self, prediction_df):
        Path(self.plots_dir_path).mkdir(parents=True, exist_ok=True)

        self.model.plot_components(prediction_df)
        plt.savefig('./' + self.company_code + '_plots/prophet_pred_details.png')


# main
# path = './005930_final_data/from_2017-08-19.csv'
company_code = '005930'
prophet = Prophet_Model(company_code=company_code, _from='2010', predict_period=365, daily_seasonality=True)
prediction = prophet.fit_and_predict()
prophet.plot_prediction(prediction_df=prediction)
prophet.plot_pred_details(prediction_df=prediction)



