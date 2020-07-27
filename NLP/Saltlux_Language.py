import json
import requests
import config

class Saltlux_Language:
    """
    Preferably for Korean Language
    """

    def __init__(self):
        self.api_key = config.saltlux_api_key

    def request_sentiment(self, text_content, lang='kor'):
        """
        :param text_content: content to be analyzed
        :param _type: kor, eng, chi
        :return: json result
        """
        # http://api.adams.ai/datamixiApi/tms?query=한국의 가을은 매우 아름답습니다.&lang=kor&analysis=om

        END_POINT = "http://api.adams.ai/datamixiApi/tms?query=" + text_content + "&lang=" + lang + \
                    "&analysis=om" + "&key=" + self.api_key
        sentiment_result = requests.get(END_POINT)
        return sentiment_result.json()

    def parse_sentiment_json(self, sentiment_json):
        """

        :param sentiment_json: JSON formatted result. Can get from get_sentiment(..)
        :return: Average of sentiment polarity and score.
        """
        for key, value in sentiment_json.items():
            print(key, ":", value)

        # TODO: Get average of sentiment polarity and score
        pass

    def request_keyword(self, text_content, lang='kor'):
        """

        :param text_content: content to be analyzed
        :param lang: kor, eng, chi
        :return: json result
        """
        # http://api.adams.ai/datamixiApi/tms?query=아버지가 태어난 나라의 도시는 나이로비이다.&lang=kor&analysis=ne&key=self.api_key

        END_POINT = "http://api.adams.ai/datamixiApi/tms?query=" + text_content + "&lang=" + lang +\
                    "&analysis=ne&key=" + self.api_key

        extraction_result = requests.get(END_POINT)
        return extraction_result.json()

    def parse_keyword_json(self, keyword_json):
        """

        :param keyword_json: JSON formatted result. Can get from extract_keyword(..)
        :return:
        """
        # TODO: decide what i should extract
        pass


saltlux = Saltlux_Language()
content = "고가차를 많이 팔면된다현대차에 대한 투자의견 BUY와 목표주가 16만원(목표 P/B 0.6배)을 유지한다. 4분기실적은 예상보다 좋았던 ASP와 판매보증충당금비용의 감소에 힘입어 최근 낮아졌던 시장 기대치를 상회했다. 실적 자체보다 긍정적이었던 것은 판매가 감소했음에도 불구하고, 외형과수익성이 기대 이상이었다는 것이다. 믹스 개선과 인센티브 감소를 통해 덜 팔아도 외형/이익이 성장하는 방법이 작동한 것이다."
content1 = "신풍제약은 1962년 설립된 제네릭 의약품 제조업체로 1990년 1월 유가증권시장에 상장. 사업 초기 구충제 품목군으로 사업 기반을 구축하였으며 이후 제네릭 의약품 품목 다각화를 통해 2,000억원대 매출규모로 성장. 피라맥스(말라리아 치료제) 신약 개발 경험을 기반으로 신약 파이프라인을 구축하며 새로운 방향성 모색. "
sentiment_json = saltlux.request_sentiment(text_content=content1)
# print(sentiment_json)

print("="*20)

saltlux.parse_sentiment_json(sentiment_json=sentiment_json)

