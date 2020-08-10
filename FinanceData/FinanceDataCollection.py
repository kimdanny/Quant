import numpy as np
import pandas as pd
from FinanceDataReader import DataReader, StockListing


class FinanceDataCollection:
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
    def get_stock_info(market_code: str, columns_of_interest=None):
        """
        Does not depend on Dates
        <columns>
         Symbol, Market, Name, Sector, Industry, ListingDate, SettleMonth, Representative, HomePage, Region

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

        :param columnsOfInterest: List of colnames of interest  ex) ['Symbol', 'Name', 'Sector', 'ListingDate']
        :return: pd.DataFrame
        """
        market_stock_list = StockListing(market_code)

        if columns_of_interest is not None:
            market_stock_list = market_stock_list[columns_of_interest]

        return market_stock_list

    def get_company_price_data(self, company_code: str):
        """
        Stock price data
        :param company_code:
                        if company code is number string like '005390' --> Korean Stock
                        if company code is English Letter like 'AAPL' -->  US Stock
        :return: pd.DataFrame
        """
        df = DataReader(company_code, self.from_, self.to) if self.to is not None else DataReader(company_code,
                                                                                                  self.from_)
        return df

    def get_index_by_market_data(self, index_code: str):
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

    @staticmethod
    def find_sameSector_companies(stocklist, company_code: str):
        """

        :param stocklist: Market Stock list. Can get from self.get_stock_list()
                         Must contain colnames 'Symbol' and 'Sector'
        :param company_code: company code
        :return: pd.DataFrame
        """

        assert ('Symbol' in stocklist.columns and 'Sector' in stocklist.columns)

        company_sector = stocklist[stocklist['Symbol'] == company_code]['Sector'].values[0]
        df = stocklist[stocklist['Sector'] == company_sector]
        # dropping current company
        drop_index = df.loc[df['Symbol'] == company_code].index
        df = df.drop(drop_index)

        return df

    def get_close_price_given_companies(self, companies_df):
        """
        Beware: Takes some time to return close_prices_df

        <use cases>
        Can get sameSector_df from self.find_sameSector_companies(), and then get close prices of all sameSector
        companies by calling self.get_close_price_given_companies(sameSector_df)

        :param companies_df: df of companies. Must contain colnames 'Symbol' and 'Name'
        :return:  (pd.DataFrame) close prices
        """
        close_prices_df = pd.DataFrame()

        for _, row in companies_df.iterrows():
            code, name = row['Symbol'], row['Name']

            df = DataReader(code, self.from_, self.to) if self.to is not None else DataReader(code, self.from_)

            # Adding Close price to company name column
            close_prices_df[name] = df['Close']
        return close_prices_df


# # For Debugging and Testing purpose
# if __name__ == '__main__':
#     data_collection = FinanceDataCollection('2019-07')
#     # stocklist = data_collection.get_stock_list('KOSPI', ['Symbol', 'Name', 'ListingDate'])
#     # stocklist = data_collection.get_stock_info('KOSPI', ['Symbol', 'Name', 'Sector'])
#     # samsung_sameSector_df = data_collection.find_sameSector_companies(stocklist=stocklist, company_code='005930')
#     # samsung_sameSector_comp_close = data_collection.get_close_price_given_companies(samsung_sameSector_df)
#     # samsung_price = data_collection.get_company_price_data('005930')
#     # apple_price = data_collection.get_company_price_data('AAPL')
#     # kospi_index = data_collection.get_index_by_market_data('KS11')
#     df = data_collection.get_currency_exchange_data()
#
#     """
#     <market index>
#     Close, Open, High, Low,
#     Volume: 거래량
#     Change: 등락률(%)
#     """
#     # currecny = data_collection.get_currency_exchange_data()
#
#     # print(stocklist)
#     # print("=" * 20)
#
#     # print(samsung_price)
#     # print("=" * 20)
#     #
#     # print(apple_price)
#     # print("=" * 20)
#     #
#     # print(kospi_index)
#     # print("=" * 20)
#     #
#     # print(currecny)
#
#     print(df.columns)
