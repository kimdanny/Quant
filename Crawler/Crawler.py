import sys
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os
from pathlib import Path
import numpy as np

from FinanceData import DataCollection

# TODO: crawl by dates range ex) crawl news from when to when
# TODO: Use multiprocessing to speed up crawler

class Naver_Crawler:

    def __init__(self, company_code):
        self.base_url = 'https://finance.naver.com'
        self.company_code = company_code
        assert type(self.company_code) == str
        root_dir = os.path.dirname(__file__)
        print(Path(__file__).resolve().parent.parent)
        base_dir = os.path.join(root_dir, "crawled_result")
        self.company_dir_path = os.path.join(base_dir, self.company_code)

    def crawl_news(self, maxpage, page_to_csv=True, full_pages_to_csv=True):
        """
        Example URL:
            https://finance.naver.com/item/news.nhn?code=095570&page=2&sm=entity_id.basic

        :param maxpage:  (int or None) Crawl to `maxpage`page
        :param page_to_csv: (Bool)  Set True if you want csv for separate pages, otherwise False
        :param full_pages_to_csv: (Bool)  Set True if you want all pages' result into one csv, otherwise False
        :return: (pd.DataFrame) crawled result
        """
        # Path Handling
        news_dir_path = os.path.join(self.company_dir_path, 'News')
        if full_pages_to_csv:
            fullPage_dir_path = Path(os.path.join(news_dir_path, 'fullPage'))
            fullPage_dir_path.mkdir(parents=True, exist_ok=True)

        page = 1
        # Tracking first and last page for file name
        firstpage = page
        last_read_page = None

        assert type(maxpage) == int

        result_df = None

        while page <= maxpage:

            url = self.base_url + '/item/news_news.nhn?code=' + self.company_code + '&page=' + str(page)

            html_text = requests.get(url).text
            html = BeautifulSoup(html_text, "html.parser")

            # TODO: Try getting the last_page as I did from below
            # Possible future Error Handling: maxpage Error -> Currently handled by Naver themself
            # 실제 웹에는 5페이지까지 밖에 없는데 maxpage를 10으로 설정한 경우 5페이지에서 loop break 시킴
            try:
                print(f'News Current page: {page}')
                last_read_page = page
                current_page_on_html = html.select('.on')[1].text.replace("\n", "").replace("\t", "")
            except IndexError:
                current_page_on_html = page
            if current_page_on_html != str(page):
                break

            # 1. ==Date==
            dates = html.select('.date')
            date_result = [date.get_text() for date in dates]

            # 2. ==Source==
            sources = html.select('.info')
            source_result = [source.get_text() for source in sources]

            # 3. ==Title==
            titles = html.select('.title')
            title_result = []
            for title in titles:
                title = title.get_text()
                title = re.sub('\n', '', title)
                title_result.append(title)

            # 4. ==Link==
            links = html.select('.title')

            link_result = []
            article_body_result = []
            for link in links:
                article_url = self.base_url + link.find('a')['href']
                link_result.append(article_url)

                # 5. ==Body==
                article_html_text = requests.get(article_url).text
                article_html = BeautifulSoup(article_html_text, "html.parser")

                body = article_html.find('div', id='news_read')
                body = body.text  # type --> sting
                body = body.replace("\n", "").replace("\t", "")
                # TODO: Reminder! body 내 특수문자 다 없애기 -> 모델에 넣을떄 하자
                article_body_result.append(body)

                # 6. TODO: ==Reaction==
                reaction_space = article_html.find('ul', class_='u_likeit_layer _faceLayer')

                good_reaction_count = int(reaction_space.find('li', class_='u_likeit_list good') \
                                          .find('span', class_='u_likeit_list_count _count').text)

                warm_reaction_count = int(reaction_space.find('li', class_='u_likeit_list warm') \
                                          .find('span', class_='u_likeit_list_count _count').text)

                sad_reaction_count = int(reaction_space.find('li', class_='u_likeit_list sad') \
                                         .find('span', class_='u_likeit_list_count _count').text)

                angry_reaction_count = int(reaction_space.find('li', class_='u_likeit_list angry') \
                                           .find('span', class_='u_likeit_list_count _count').text)

                want_reaction_count = int(reaction_space.find('li', class_='u_likeit_list want') \
                                          .find('span', class_='u_likeit_list_count _count').text)

                # print(reaction_space)
                # print("="*20)

                # 7. TODO: ==Commentary==
                comments = article_html.find_all(
                    lambda tag: tag.name == 'span' and tag.get('class') == 'u_cbox_contents')
                # print(comments)

            # To Dataframe and To CSV (optional)
            page_result = {
                "Date": date_result, "Source": source_result, "Title": title_result,
                "Link": link_result, "Body": article_body_result,
                "good_count": good_reaction_count,
                "warm_count": warm_reaction_count,
                "sad_count": sad_reaction_count,
                "angry_count": angry_reaction_count,
                "want_count": want_reaction_count
            }

            page_df = pd.DataFrame(page_result)

            if result_df is None:
                result_df = page_df
            else:
                # bind page_df at the bottom of the result_df
                result_df = result_df.append(page_df, ignore_index=True)

            if page_to_csv:
                page_df.to_csv(os.path.join(news_dir_path, 'page' + str(page) + '.csv'),
                               mode='w', encoding='utf-8-sig')  # 한글 깨짐 방지 인코딩

            page += 1

        if full_pages_to_csv:
            result_df.to_csv(os.path.join(fullPage_dir_path, 'pages' + str(firstpage) + '-' + str(last_read_page) + '.csv'),
                mode='w', encoding='utf-8-sig')  # 한글 깨짐 방지 인코딩

        return result_df

    def crawl_research(self, maxpage=None, page_to_csv=True, full_pages_to_csv=True):
        """
        Example URL:
            https://finance.naver.com/research/company_list.nhn?keyword=&searchType=itemCode&itemCode=105560&page=1

        :param maxpage:  (int or None) Crawl to `maxpage`page
        :param page_to_csv: (Bool)  Set True if you want csv for separate pages, otherwise False
        :param full_pages_to_csv: (Bool)  Set True if you want all pages' result into one csv, otherwise False
        :return: (pd.DataFrame) crawled result
        """
        # Path Handling
        research_dir_path = os.path.join(self.company_dir_path, 'Research')

        if full_pages_to_csv:
            fullPage_dir_path = Path(os.path.join(research_dir_path, 'fullPage'))
            fullPage_dir_path.mkdir(parents=True, exist_ok=True)

        page = 1
        firstpage = page
        last_read_page = None

        # Get Last page number
        url = self.base_url + '/research/company_list.nhn?keyword=&searchType=itemCode&itemCode=' + self.company_code
        page_nav = BeautifulSoup(requests.get(url).text, 'html.parser').select('.pgRR')[0]
        last_page = page_nav.find('a')['href']
        match = re.search(r'page=', last_page)
        last_page = int(last_page[match.end():])
        del match

        maxpage = last_page if maxpage is None else maxpage

        assert type(maxpage) == int and maxpage <= last_page

        result_df = None

        while page <= maxpage:
            url = self.base_url + '/research/company_list.nhn?keyword=&searchType=itemCode&itemCode=' \
                  + self.company_code + '&page=' + str(page)

            html_text = requests.get(url).text
            html = BeautifulSoup(html_text, "html.parser")
            html = html.select('.box_type_m')
            html = html[0].find_all('tr')  # tr list

            stock, title, link, body, goal_price, opinion, source, date, views = [], [], [], [], [], [], [], [], []

            for tr in iter(html[2:-3]):
                td_list = tr.find_all('td')
                if not (len(td_list) == 1):
                    # get info
                    stock.append(td_list[0].find('a')['title'])
                    title.append(td_list[1].find('a').text)
                    research_url = self.base_url + "/research/" + td_list[1].find('a')['href']
                    link.append(research_url)

                    # surf into link and get goal_price, opinion and body
                    article_html = BeautifulSoup(requests.get(research_url).text, 'html.parser')

                    goalNopinion = article_html.select('.view_info_1')  # div
                    price = goalNopinion[0].find_all('em')[0].text[:-1]
                    try:
                        price = int(price)
                    except ValueError:      # 가끔 어떤 목표가는 N/R 이라고 적혀있다
                        price = np.nan
                    goal_price.append(price)
                    opinion.append(goalNopinion[0].find_all('em')[1].text)

                    body.append(article_html.find('td', class_='view_cnt').find('div')
                                .text.replace("\n", "").replace("\t", ""))

                    source.append(td_list[2].text)
                    date.append(td_list[4].text)
                    views.append(td_list[5].text)

            # To Dataframe and To CSV (optional)
            page_result = {
                "Stock": stock, "Title": title,
                "Link": link, "Source": source,
                "Body": body, "Goal_Price": goal_price, "Opinion": opinion,
                "Date": date, "Views": views
            }

            page_df = pd.DataFrame(page_result)

            if result_df is None:
                result_df = page_df
            else:
                # bind page_df at the bottom of the result_df
                result_df = result_df.append(page_df, ignore_index=True)

            if page_to_csv:
                page_df.to_csv(os.path.join(research_dir_path, 'page' + str(page) + '.csv'),
                               mode='w', encoding='utf-8-sig')  # 한글 깨짐 방지 인코딩

            last_read_page = page
            page += 1

        if full_pages_to_csv:
            result_df.to_csv(os.path.join(fullPage_dir_path, 'pages' + str(firstpage) + '-' + str(last_read_page) + '.csv'),
                             mode='w', encoding='utf-8-sig')  # 한글 깨짐 방지 인코딩

        return result_df


if __name__ == '__main__':
    # 종목 코드로 기사 크롤링 --> 종목코드는 FinancialDataReader에서 받아온다.

    # sample code --> { '005930': '삼성전자',
    #                   '005380': '현대차',
    #                   '015760': '한국전력',
    #                   '005490': 'POSCO',
    #                   '105560': 'KB금융'
    #                   '95570' : 'AJ네트웍스'}

    naver_crawler = Naver_Crawler('005380')
    # news_df = naver_crawler.crawl_news(maxpage=2)
    # print(sample_df)

    # research = naver_crawler.crawl_research(2)
    # print(research)

    # discussion = naver_crawler.crawl_discussion(1)
    # print(discussion)
