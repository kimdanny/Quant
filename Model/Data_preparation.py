import sys
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

# To import from parent Directories
# print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NLP import GCP_Language
from NLP import Saltlux_Language
from FinanceData import FinanceDataCollection
from Crawler import Naver_Crawler, Google_Crawler


class CombineData:
    def __init__(self, company_code, years, include_language=False, save_as_csv=True):
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
        # self.years = years
        self.save_as_csv = save_as_csv
        self.from_ = (datetime.now() - relativedelta(years=years)).date().strftime('%Y-%m-%d')  # stringify datetime

        # PRIVATE variables
        if include_language:
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

        if self.save_as_csv:
            # Path Handling
            dirname = self.company_code + '_final_data'
            root_dir = os.path.dirname(__file__)
            self.target_dir_path = os.path.join(root_dir, dirname)
            Path(self.target_dir_path).mkdir(parents=True, exist_ok=True)
            self.file_name = 'from_' + self.from_ + '.csv'

    def combine(self):
        pass


combined = CombineData('005930', years=2)
