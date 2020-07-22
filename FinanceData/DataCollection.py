import numpy as np
from FinanceDataReader import DataReader, StockListing


class DataCollection:
    def __init__(self, *dates):
        """
        Set from, to dates
        :param dates: Can be either  YYYY, YYYY-MM or YYYY-MM-DD
        """
        # Dates from and to
        # possible formats should be the same as below
        assert 0 < len(dates) <= 2
        self.from_ = None
        self.to = None

        if len(dates) == 1:
            self.from_ = dates[0]
        else:
            self.from_ = dates[0]
            self.to = dates[1]

    @staticmethod
    def get_stock_list(market_code, columns_of_interest):
        """
        Does not depend on Dates
        :param market_code:
            KRX	            KRX 종목 전체
            KOSPI	        KOSPI 종목
            KOSDAQ	        KOSDAQ 종목
            KONEX	        KONEX 종목
            KRX-DELISTING	KRX상장폐지종목

            NASDAQ	        나스닥 종목
            NYSE	        뉴욕 증권거래소 종목
            AMEX	        AMEX 종목
            S&P500	        S&P 500 종목

        :param columnsOfInterest: List of colnames of interest  ex) ['Symbol', 'Name', 'ListingDate']
        :return: pd.DataFrame
        """
        market_stock_list = StockListing(market_code)
        market_stock_list = market_stock_list[columns_of_interest]

        return market_stock_list

    def get_company_price_data(self, company_code):
        """
        Stock price data
        :param company_code:
                        if company code is number string like '005390' --> Korean Stock
                        if company code is English Letter like 'AAPL' -->  US Stock
        :return: pd.DataFrame
        """
        df = DataReader(company_code, self.from_, self.to) if self.to is not None else DataReader(company_code, self.from_)
        return df

    def get_index_by_market_data(self, index_code):
        """
        Get market index history data
        :param index_code:  KS11	KOSPI 지수
                            KQ11	KOSDAQ 지수
                            KS50	KOSPI 50 지수
                            KS100	KOSPI 100 지수
                            KS200	KOSPI 200 지수
                            KQ150	KOSDAQ 150 지수
                            KRX100	KRX 100

                            DJI	    다우존스 지수
                            IXIC	나스닥 지수
                            US500	S&P 500 지수
                            VIX	S&P 500 VIX

        :return: pd.DataFrame
        """

        df = DataReader(index_code, self.from_, self.to) if self.to is not None else DataReader(index_code, self.from_)

        return df

    def get_currency_exchange_data(self, currency='USD/KRW'):
        """
        :param currency:    'USD/KRW' by default
        :return: pd.DataFrame
        """

        df = DataReader(currency, self.from_, self.to) if self.to is not None else DataReader(currency, self.from_)

        return df


# For Debugging and Testing purpose
if __name__ == '__main__':
    data_collection = DataCollection('2019-03', '2019-08')
    stocklist = data_collection.get_stock_list('KOSPI', ['Symbol', 'Name', 'ListingDate'])
    samsung_price = data_collection.get_company_price_data('005930')
    apple_price = data_collection.get_company_price_data('AAPL')
    kospi_index = data_collection.get_index_by_market_data('KS11')
    currecny = data_collection.get_currency_exchange_data()

    print(stocklist)
    print("=" * 20)

    print(samsung_price)
    print("=" * 20)

    print(apple_price)
    print("=" * 20)

    print(kospi_index)
    print("=" * 20)

    print(currecny)
