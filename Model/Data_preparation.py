import sys
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from math import sqrt

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NLP import GCP_Language, Saltlux_Language
from FinanceData import FinanceDataCollection
from Crawler import Naver_Crawler, Google_Crawler


# should I make json or yaml config and read it??


class CombineData:
    def __init__(self, company_code: str, years: int, include_language=False, save_as_csv=True):
        """

        :param company_code: type==str
                    Korean Stock code will only contain numbers, whereas that of US will only contain Alphabets
        :param years: type==int
                    Number of past years you wanna collect.
                    ex) years=2 --> will collect 2 years of past data from current date.
        :param include_language: type==bool
                    Will activate crawlers and Language APIs for collecting and analyzing Language data.
        :param save_as_csv:  type==bool
                    Will save final combined data as a CSV file in current directory.
        """
        self.company_code = company_code
        self.from_ = (datetime.now() - relativedelta(years=years)).date().strftime('%Y-%m-%d')  # stringify datetime

        # PRIVATE variables
        self._save_as_csv = save_as_csv
        self._include_language = include_language

        if self._include_language:
            if self.company_code.isdecimal():  # string contains only numbers --> Korean Stocks
                print("=== Korean Stock ===")
                self.naver_crawler = Naver_Crawler.Naver_Crawler(self.company_code)
                print(f'Naver Crawler is set with company code: {self.naver_crawler.company_code}')
                self.saltlux = Saltlux_Language.Saltlux_Language()
                print('Saltlux API is set')
            else:  # elif self.company_code.isalpha(): --> US Stocks
                print("=== US Stock ===")
                self.google_crawler = Google_Crawler.Google_Crawler(self.company_code)
                print(f'Google Crawler is set with company code: {self.google_crawler.company_code}')
                self.gcp = GCP_Language.GCP_Language()
                print('US Stock: GCP Natural Language is set')

        self.finance_data = FinanceDataCollection.FinanceDataCollection(self.from_)
        print(f'FinanceDataCollection Class is set from date: {self.finance_data.from_}')

        self.encoder = LabelEncoder()

        if self._save_as_csv:
            # Path Handling
            dirname = self.company_code + '_final_data'
            root_dir = os.path.dirname(__file__)
            self.target_dir_path = os.path.join(root_dir, dirname)
            Path(self.target_dir_path).mkdir(parents=True, exist_ok=True)
            self.file_name = 'from_' + self.from_ + '.csv'

        # Path Handling for plots
        plot_dirname = self.company_code + '_plots'
        root_dir = os.path.dirname(__file__)
        self.plots_dir_path = os.path.join(root_dir, plot_dirname)

    # Private Method
    # TODO: Multiprocessing?
    def combine_finance_data(self, us_currecny=True, volume_change=True, moving_avg=True, fourier=True):
        """

        :param volume_change: (Derivative Column) Trading Volume change in Percentage
        :param moving_avg:    (Derivative Column) Moving Average with window size 20 days
        :return: combined data
        """
        source = self.finance_data  # for better computability

        # price_df --> Date, Open, High, Low, Close, Volume, Change
        price_df = source.get_company_price_data(self.company_code)

        # market_index_df --> Date, Open, High, Low, Close, Volume, Change
        market_index_df = source.get_index_by_market_data('KS200')
        # TODO: should I just remove Open, High and Low ??
        # market_index_df = market_index_df[['Date', 'Close', 'Volume', 'Change']]

        """
        stock_info --> Symbol, Market, Name, Sector, Industry, ListingDate, SettleMonth,
                       Representative, Homepage, Region
        same_sector_companies_df --> specified in columns of interest
        same_sector_close_df --> Date, same sector companies
        """
        stock_info = source.get_stock_info('KOSPI', columns_of_interest=['Symbol', 'Sector', 'Name'])
        same_sector_companies_df = source.find_sameSector_companies(stock_info, self.company_code)
        same_sector_close_df = source.get_close_price_given_companies(same_sector_companies_df)

        del stock_info, same_sector_companies_df

        merged = pd.merge(price_df, market_index_df, on='Date').merge(same_sector_close_df, on='Date')

        if us_currecny:  # if US Stock or Big companies like SamSung, consider currency exchange rate
            # currency_df --> Date, Open, High, Low, Close, Change
            currency_df = source.get_currency_exchange_data()
            currency_df = currency_df[['Close']]
            currency_df = currency_df.rename(columns={'Close': 'usd-krw'})
            merged = pd.merge(merged, currency_df, on='Date')

        del source

        # 액면분할 (Forward Split) 시 생기는 0값들 있는 row drop
        drop_index = merged[merged['Open_x'] == 0].index
        merged = merged.drop(drop_index)

        """Adding new Columns: Derivatives"""
        ########################################################
        # 1. Volume Change of Stock trading and market trading #
        ########################################################
        if volume_change:
            i = 1
            vol_change_x = [0]
            vol_change_y = [0]

            while i < len(merged):
                try:
                    vol_change_x.append(
                        (merged.iloc[i]['Volume_x'] - merged.iloc[i - 1]['Volume_x']) / merged.iloc[i]['Volume_x']
                    )
                except RuntimeWarning:
                    vol_change_x.append(0.0)
                try:
                    vol_change_y.append(
                        (merged.iloc[i]['Volume_y'] - merged.iloc[i - 1]['Volume_y']) / merged.iloc[i]['Volume_y']
                    )
                except RuntimeWarning:
                    vol_change_y.append(0.0)

                i += 1

            merged['vol_change_x'] = vol_change_x
            print("Added derivative: vol_change_x")
            merged['vol_change_y'] = vol_change_y
            print("Added derivative: vol_change_y")

        ###################################################################
        # 2. Adding Moving average MA_5, MA_20, MA_diff, G_cross, D_cross #
        ###################################################################
        if moving_avg:
            merged['MA_5'] = merged['Close_x'].rolling(window=5, min_periods=0).mean()
            print("Added derivative: MA_5")

            merged['MA_20'] = merged['Close_x'].rolling(window=20, min_periods=0).mean()
            print("Added derivative: MA_20")

            merged['MA_diff'] = merged['MA_5'] - merged['MA_20']
            print("Added derivative: MA_diff")

            # Plotting Golden Cross
            Path(self.plots_dir_path).mkdir(parents=True, exist_ok=True)
            rcParams['figure.figsize'] = 18, 6  # width 18, height 6
            ax = merged[['Close_x', 'MA_5', 'MA_20']].plot()

            # Golden and Dead cross finder
            merged['Cross'] = 0.0  # Cross column placeholder

            prev_key = prev_val = 0
            for key, val in merged['MA_diff'].iteritems():
                if val == 0:
                    continue
                if val * prev_val < 0:
                    if val > prev_val:
                        print(f'golden {key}, {val}')
                        ax.annotate('Golden', xy=(key, merged['MA_20'][key]), xytext=(10, -30),
                                    textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))

                        merged.at[key, 'Cross'] = 1.0  # Encode Golden as 1

                    elif val < prev_val:
                        print(f'dead {key}, {val}')
                        ax.annotate('Dead', xy=(key, merged['MA_20'][key]), xytext=(10, 30),
                                    textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))

                        merged.at[key, 'Cross'] = -1.0  # Encode Dead as -1

                prev_key, prev_val = key, val

            plt.savefig('./' + self.company_code + '_plots/MA_Golden_Cross.png')

        ##########################################
        # 3. Exponential Weighted Moving Average #
        ##########################################
        # dataset['ema'] = dataset['price'].ewm(com=0.5).mean()
        # EWMA_12 = pd.ewm(merged['Close_x'], span=12)
        # EWMA_26 = pd.ewm(merged['Close_x'], span=26)
        # merged['MACD'] = (EWMA_12 - EWMA_26)

        ############################
        # 4. Upper and Lower Bands #
        ############################
        # merged['20sd'] = pd.stats.moments.rolling_std(merged['Close_x'], 20)
        # merged['upper_band'] = merged['MA_20'] + (merged['20sd'] * 2)
        # merged['lower_band'] = merged['MA_20'] - (merged['20sd'] * 2)

        #############################################
        # 5. Fourier Transform - 3, 6, 9 components #
        #############################################
        # To Denoise the series and get trends
        if fourier:
            data_FT = merged[['Close_x']]
            close_fft = np.fft.fft(np.asarray(data_FT['Close_x'].tolist()))

            Path(self.plots_dir_path).mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(14, 7), dpi=100)
            for num_ in [9, 50]:
                fft_list_m10 = np.copy(close_fft)
                fft_list_m10[num_:-num_] = 0
                ft_final = list(map(np.real, (np.fft.ifft(fft_list_m10))))  # Discarding Imaginary parts

                merged[f'FT_{num_}'] = ft_final
                merged[f'FT_{num_}'].plot()

            data_FT['Close_x'].plot(label='Real', style='b')

            plt.xlabel('Trading Days')
            plt.ylabel('KRW')
            plt.title('Close stock prices & Fourier transforms')
            plt.legend()
            plt.savefig('./' + self.company_code + '_plots/Fourier_Transforms.png')

        #######################################################
        # 6. ARIMA (Auto Regression Integrated Moving Average #
        #######################################################
        # Target: Close Price
        # close_series = merged['Close_x']
        # model = ARIMA(close_series, order=(5, 1, 0))
        # model_fit = model.fit(disp=0)
        # print(model_fit.summary())
        #
        # autocorrelation_plot(close_series)
        # plt.figure(figsize=(10, 7), dpi=80)
        # plt.show()
        #
        # X = close_series.values
        # size = int(len(X) * 0.66)
        # train, test = X[0:size], X[size:len(X)]
        # history = [x for x in train]
        # predictions = list()
        # for t in range(len(test)):
        #     model = ARIMA(history, order=(5, 1, 0))
        #     model_fit = model.fit(disp=0)
        #     output = model_fit.forecast()
        #     yhat = output[0]
        #     predictions.append(yhat)
        #     obs = test[t]
        #     history.append(obs)
        #
        # error = mean_squared_error(test, predictions)
        # print('Test RMSE: %.3f' % sqrt(error))
        #
        # plt.figure(figsize=(12, 6), dpi=100)
        # plt.plot(test, label='Real')
        # plt.plot(predictions, color='red', label='Predicted')
        # plt.xlabel('Days')
        # plt.ylabel('KRW')
        # plt.title('Figure 5: ARIMA model on Samsung stock')
        # plt.legend()
        # plt.show()

        return merged

    def combine_language_data(self):

        # source = self.finance_data
        # read csv before deployment
        # news = self.naver_crawler.crawl_news()
        # research = self.naver_crawler.crawl_research()
        news = pd.read_csv('../Crawler/crawled_result/005930/News/fullPage/pages1-2.csv')
        research = pd.read_csv('../Crawler/crawled_result/005930/Research/fullPage/pages1-25.csv')

        # not needed when getting object directly from Crawler
        news = news.drop('Unnamed: 0', axis=1)
        research = research.drop('Unnamed: 0', axis=1)

        news = news[['Date', 'Title', 'Body']]
        research = research[['Date', 'Title', 'Body', 'Goal_Price', 'Opinion', 'Views']]
        research['Date'] = '20' + research['Date']

        # convert to Timestamp object
        news['Date'] = pd.to_datetime(news['Date'])
        research['Date'] = pd.to_datetime(research['Date'])

        # TODO: 뉴스와 리서치를 처리하기 전에 애초에 self.from_에서 데이터프레임의 밑단을 잘라버리자
        # 더 좋은 방법은: 날짜를 파라미터로 받아서 크롤링해오자...

        ##########
        #  NEWS  #
        ##########
        def today_or_next_day(time):
            """
            return True if today's news, False if next day's news
            """
            today_close_time = Timestamp(time.date()) + timedelta(hours=15, minutes=30)

            return time <= today_close_time

        # Initialize which_date column
        news['which_date'] = 0

        # Fill in which_date column
        for i, timestamp in enumerate(news['Date']):
            if today_or_next_day(timestamp):
                news.at[i, 'which_date'] = Timestamp(timestamp.date())
            else:
                news.at[i, 'which_date'] = Timestamp(timestamp.date()) + timedelta(days=1)

        news = news[['which_date', 'Title', 'Body']]
        news.columns = ['Date', 'Title', 'Body']

        def concat_series_by_date(data, column):
            """
            concat all the values in specified column, splitted by date
            :param column: target column name
            :return: list of concatenated values
            """
            finals = []

            for date in data['Date'].unique():
                temp_log = " "
                values = data[data['Date'] == date][column].values
                temp_log = temp_log.join(values)

                finals.append(temp_log)

            return finals

        news_title_grouped = concat_series_by_date(data=news, column='Title')
        news_body_grouped = concat_series_by_date(data=news, column='Body')

        news_grouped_temp = pd.DataFrame({
            'Date': news['Date'].unique(),
            'Title': news_title_grouped,
            'Body': news_body_grouped
        })

        news_grouped = pd.DataFrame({
            'Date': news['Date'].unique(),
            'Language': "-",
            'Sentiment': np.nan,
            'Polarity': np.nan
        })

        for i, row in news_grouped_temp.iterrows():
            news_grouped.at[i, "Language"] = row.Title + row.Body

        del news_grouped_temp

        print("news grouped: ", news_grouped)

        # # USE API to get scores
        # for i, title in enumerate(news_grouped['Language']):
        #     sentiment_result = self.saltlux.request_sentiment(text_content=title, dump=True)
        # polarity and score are mean value of many
        #     polarity, score, _ = self.saltlux.parse_sentiment_json(sentiment_json=sentiment_result)
        #     news_grouped.at[i, 'Sentiment'] = score
        #     news_grouped.at[i, 'Polarity'] = polarity

        ############
        # Research #
        ############

        # Opinion Encoding
        # Replace 'BUY', 'StrongBUY' with '매수', '강력매수'
        research['Opinion'] = research['Opinion'].replace(to_replace=['Buy'], value='매수')
        research['Opinion'] = research['Opinion'].replace(to_replace=['StrongBuy'], value='강력매수')

        opinons = research['Opinion'].values
        self.encoder.fit(opinons)
        opinion_encoded = self.encoder.transform(opinons)
        research['Opinion'] = opinion_encoded

        # research grouping
        research_title_grouped = concat_series_by_date(data=research, column='Title')
        research_body_grouped = concat_series_by_date(data=research, column='Body')

        # Contains only numeric values
        research_data_numeric = research[['Date', 'Goal_Price', 'Opinion', 'Views']]
        research_goal_price_grouped = research_data_numeric[['Date', 'Goal_Price']].groupby('Date') \
                                          .mean()['Goal_Price'].values[::-1]
        research_opinion_grouped = research_data_numeric[['Date', 'Opinion']].groupby('Date') \
                                          .mean()['Opinion'].values[::-1]
        research_views_grouped = research_data_numeric[['Date', 'Views']].groupby('Date') \
                                          .mean()['Views'].values[::-1]

        del research_data_numeric

        assert (
                len(research_title_grouped) == len(research_body_grouped) == len(research_goal_price_grouped)
                == len(research_opinion_grouped) == len(research_views_grouped)
        )

        research_grouped_temp = pd.DataFrame({
            'Date': research['Date'].unique(),
            'Title': research_title_grouped,
            'Body': research_body_grouped
        })

        research_grouped = pd.DataFrame({
            'Date': research['Date'].unique(),
            'Language': "-",
            'Goal_Price': research_goal_price_grouped,
            'Opinion': research_opinion_grouped,
            'Views': research_views_grouped,
            'Sentiment': np.nan,
            'Polarity': np.nan
        })

        # Combine title and body into one 'language' column
        for i, row in research_grouped_temp.iterrows():
            research_grouped.at[i, "Language"] = row.Title + row.Body

        del research_grouped_temp
        print("research grouped", research_grouped)

        # # Use API to get scores of the Language
        # # Not trying to use API many times
        # for i, lang in enumerate(research_grouped['Language']):
        #     sentiment_result = self.saltlux.request_sentiment(text_content=lang, dump=True, dump_id="Research")
        #     polarity, score, _ = self.saltlux.parse_sentiment_json(
        #         sentiment_json=sentiment_result)  # polarity and score are mean value of many
        #     research_grouped.at[i, 'Sentiment'] = score
        #     research_grouped.at[i, 'Polarity'] = polarity

        ###################
        # News + Research #
        ###################



        return None

    def combine(self, volume_change=True, moving_avg=True, us_currency=False, fourier=True):

        combined = self.combine_finance_data(volume_change=volume_change, moving_avg=moving_avg,
                                             us_currecny=us_currency, fourier=fourier)

        # volume change, MA_diff 가 0인 row는 제거
        drop_index = combined[combined['vol_change_x'] == 0].index
        combined = combined.drop(drop_index)

        drop_index = combined[combined['vol_change_y'] == 0].index
        combined = combined.drop(drop_index)

        drop_index = combined[combined['MA_diff'] == 0].index
        combined = combined.drop(drop_index)

        if self._include_language:
            # TODO: combine finance and language
            combined = None
            pass

        if self._save_as_csv:
            combined.to_csv(os.path.join(self.target_dir_path, self.file_name))

        return combined


# main
combine_data = CombineData('005930', years=3)
# combined_df = combine_data.combine(us_currency=True)
# print(combined_df)
combine_data.combine_language_data()
