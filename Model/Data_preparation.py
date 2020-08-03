import sys
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pandas as pd
# import multiprocessing
# print(multiprocessing.cpu_count())  # --> 8

# To import from parent Directories
# print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
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
            else:                               # elif self.company_code.isalpha(): --> US Stocks
                print("=== US Stock ===")
                self.google_crawler = Google_Crawler.Google_Crawler(self.company_code)
                print(f'Google Crawler is set with company code: {self.google_crawler.company_code}')
                self.gcp = GCP_Language.GCP_Language()
                print('US Stock: GCP Natural Language is set')

        self.finance_data = FinanceDataCollection.FinanceDataCollection(self.from_)
        print(f'FinanceDataCollection Class is set from date: {self.finance_data.from_}')

        if self._save_as_csv:
            # Path Handling
            dirname = self.company_code + '_final_data'
            root_dir = os.path.dirname(__file__)
            self.target_dir_path = os.path.join(root_dir, dirname)
            Path(self.target_dir_path).mkdir(parents=True, exist_ok=True)
            self.file_name = 'from_' + self.from_ + '.csv'

    # Private Method
    # TODO: Multiprocessing?
    def combine_finance_data(self, volume_change=True, moving_avg=True):
        """

        :param volume_change: (Derivative Column) Trading Volume change in Percentage
        :param moving_avg:    (Derivative Column) Moving Average with window size 20 days
        :return: combined data
        """
        source = self.finance_data

        # price_df --> Date, Open, High, Low, Close, Volume, Change
        price_df = source.get_company_price_data(self.company_code)

        # market_index_df --> Date, Open, High, Low, Close, Volume, Change
        market_index_df = source.get_index_by_market_data('KS200')

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

        if self.company_code.isalpha():  # if US Stock, consider currency exchange rate
            # currency_df --> Date, Open, High, Low, Close, Change
            currency_df = source.get_currency_exchange_data()
            merged = pd.merge(merged, currency_df, on='Date')

        del source

        # 액면분할시 생기는 0값들 있는 row drop
        drop_index = merged[merged['Open_x'] == 0].index
        merged = merged.drop(drop_index)

        """Add new Columns: Derivatives"""
        # 1. Volume Change of Stock trading and market trading
        if volume_change:
            i = 1
            vol_change_x = [0]
            vol_change_y = [0]

            while i < len(merged):
                try:
                    vol_change_x.append((merged.iloc[i]['Volume_x'] - merged.iloc[i - 1]['Volume_x']) / merged.iloc[i]['Volume_x'])
                except RuntimeWarning:
                    vol_change_x.append(0.0)
                try:
                    vol_change_y.append((merged.iloc[i]['Volume_y'] - merged.iloc[i - 1]['Volume_y']) / merged.iloc[i]['Volume_y'])
                except RuntimeWarning:
                    vol_change_y.append(0.0)

                i += 1

            merged['vol_change_x'] = vol_change_x
            print("Added derivative: vol_change_x")
            merged['vol_change_y'] = vol_change_y
            print("Added derivative: vol_change_y")

        # 2. Adding Moving average
        # Reference EDA.ipynb for details and plots
        if moving_avg:
            merged['Moving_av'] = merged['Close_x'].rolling(window=20, min_periods=0).mean()
            print("Added derivative: Moving_avg")

        return merged

    def combine_language_data(self):
        merged = None
        return merged

    def combine(self, finance_volume_change=True, finance_moving_avg=True):

        if self._include_language:
            # TODO: combine finance and language
            combined = None
            pass
        else:
            # TODO: just call combine_finance_data()
            combined = self.combine_finance_data(volume_change=finance_volume_change, moving_avg=finance_moving_avg)

        if self._save_as_csv:
            combined.to_csv(os.path.join(self.target_dir_path, self.file_name))

        return combined


# combine_data = CombineData('005930', years=3)
# combined_df = combine_data.combine()
# print(combined_df)

