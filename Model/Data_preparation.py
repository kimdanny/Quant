import sys
import os

print(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NLP import GCP_Language
from NLP import Saltlux_Language
from FinanceData import FinanceDataCollection
from Crawler import Crawler


class CombineData:
    def __init__(self):
        self.saltlux = Saltlux_Language.Saltlux_Language()
        self.gcp = GCP_Language.GCP_Language()
        self.finance_data = FinanceDataCollection.FinanceDataCollection('2019')
        self.naver_crawler = Crawler.Naver_Crawler('005380')
        print(self.naver_crawler.company_code)

    def combine(self):
        pass


combined = CombineData()
