import json
import requests
import config
from statistics import mean


class Saltlux_Language:
    """
    Not limited to, but preferably for Korean Language Analysis.
    """

    def __init__(self):
        self.api_key = config.saltlux_api_key

    @staticmethod
    def dump_json(json_object, filepath):
        with open(filepath, 'w', encoding='utf-8-sig') as f:
            json.dump(json_object, f, ensure_ascii=False, indent=4)  # reads korean without prob

    @staticmethod
    def read_json(filepath):
        with open(filepath, encoding='utf-8-sig') as f:
            return json.load(f)

    def request_sentiment(self, text_content, dump=False, dump_id=0, lang='kor'):
        """
        :param text_content: content to be analyzed
        :param dump_id: unique id to distinguish sentiment.json file
        :param dump: (Bool) Do you want to dump json file to filepath
        :param lang: kor, eng, chi
        :return: json result
        """
        # http://api.adams.ai/datamixiApi/tms?query=한국의 가을은 매우 아름답습니다.&lang=kor&analysis=om

        END_POINT = "http://api.adams.ai/datamixiApi/tms?query=" + text_content + "&lang=" + lang + \
                    "&analysis=om" + "&key=" + self.api_key
        sentiment_result = requests.get(END_POINT).json()

        if dump:
            self.dump_json(sentiment_result, './sentiment' + str(dump_id) + '.json')

        return sentiment_result

    def parse_sentiment_json(self, sentiment_json):
        """

        :param sentiment_json: JSON formatted result. Can get from get_sentiment(..)
        :return: 1) Average of polarities of all sentences
                 2) Average of scores of all sentences
                 3) List of sentiment words

                <json path>
                return_object -> sentence (type==list) --> 'sa' (find if presents and continue looping)
        """
        polarity_log = []
        score_log = []
        sentiword_log = []

        # TODO: Possible Error (or unwanted way) handling.
        # Empty log
        # maybe i should use different approach
        for sentence in sentiment_json['return_object']['sentence']:
            if 'sa' in sentence:
                sa = sentence['sa']
                polarity = sa['polarity'] if 'polarity' in sa else None
                score = sa['score'] if 'score' in sa else None
                sentiword = sa['sentiword'] if 'sentiword' in sa else None

                polarity_log.append(polarity)
                score_log.append(score)
                sentiword_log.extend(sentiword)

        return mean(polarity_log), mean(score_log), sentiword_log

    def request_entities(self, text_content, dump=False, dump_id=0, lang='kor'):
        """

        :param text_content: content to be analyzed
        :param dump_id: unique id to distinguish entities.json file
        :param dump: (Bool) Do you want to dump json file to filepath
        :param lang: kor, eng, chi
        :return: json result
        """
        # http://api.adams.ai/datamixiApi/tms?query=아버지가 태어난 나라의 도시는 나이로비이다.&lang=kor&analysis=ne&key=self.api_key

        END_POINT = "http://api.adams.ai/datamixiApi/tms?query=" + text_content + "&lang=" + lang + \
                    "&analysis=ne&key=" + self.api_key

        extraction_result = requests.get(END_POINT).json()

        if dump:
            self.dump_json(extraction_result, './entities' + str(dump_id) + '.json')

        return extraction_result

    def parse_entities_json(self, entities_json):
        """

        :param entities_json: JSON formatted result. Can get from request_entities(..)
        :return: <json path>
                return_object -> sentence (type==list) --> 'ne' (find if presents and continue looping)
        """

        entities = []
        for sentence in entities_json['return_object']['sentence']:
            if 'ne' in sentence:
                ne = sentence['ne']
                content_log = dict()
                for ne_content in ne:
                    content_log.update({ne_content['text'] : ne_content['tag']})
                entities.append(content_log)

        # TODO: Which Tag should I extract in the end?

        return entities

    def request_keywords(self, text_content, request_id=0, dump=False):
        """
        Korean and English are both acceptable without language selection

        :param text_content: content to be analyzed
        :param request_id: unique id for api request (can be used for distinguishing json files, as dump_id does)
        :param dump: (Bool) Do you want to dump json file to filepath
        :return: json result
        """
        # http://api.adams.ai/datamixiApi/keywordextract?text=This is a sample text.&request_id=0&key=self.api_key

        END_POINT = "http://api.adams.ai/datamixiApi/keywordextract?text=" + text_content + "&request_id=" + \
                    str(request_id) + "&key=" + self.api_key
        extraction_result = requests.get(END_POINT).json()

        if dump:
            self.dump_json(extraction_result, './keyword' + str(request_id) + '.json')

        return extraction_result

    def parse_keywords_json(self, keyword_json, top_n=3):
        """

        :param keyword_json:
        :return: Top N important keywords sorted by weights, weight
                <json path>
                return_object (type==list) --> 'term (up until '|')' and 'weight'
        """
        # API Ensures that return_object is sorted by weights

        top_n_objects = keyword_json['return_object'][:top_n]  # max top_n

        for i, dic in enumerate(top_n_objects):
            term = dic['term'].split('|')[0]
            top_n_objects[i]['term'] = term

        return top_n_objects


if __name__ == '__main__':
    saltlux = Saltlux_Language()
    content = "고가차를 많이 팔면된다. 현대차에 대한 투자의견 BUY와 목표주가 16만원(목표 P/B 0.6배)을 유지한다. 4분기실적은 예상보다 좋았던 ASP와 판매보증충당금비용의 감소에 힘입어 최근 낮아졌던 시장 기대치를 상회했다. 실적 자체보다 긍정적이었던 것은 판매가 감소했음에도 불구하고, 외형과수익성이 기대 이상이었다는 것이다. 믹스 개선과 인센티브 감소를 통해 덜 팔아도 외형/이익이 성장하는 방법이 작동한 것이다."
    content1 = "신풍제약은 1962년 설립된 제네릭 의약품 제조업체로 1990년 1월 유가증권시장에 상장. 사업 초기 구충제 품목군으로 사업 기반을 구축하였으며 이후 제네릭 의약품 품목 다각화를 통해 2,000억원대 매출규모로 성장. 피라맥스(말라리아 치료제) 신약 개발 경험을 기반으로 신약 파이프라인을 구축하며 새로운 방향성 모색. "

    # sentiment_json = saltlux.request_sentiment(text_content=content1, dump=True)
    # entities_json = saltlux.request_entities(text_content=content1, dump=True)
    # keywords_json = saltlux.request_keywords(text_content=content1, dump=True)
    # print(sentiment_json)

    print("=" * 20)

    # READ JSON - not to use API calls every time --> Debugging purpose
    sentiment_json = saltlux.read_json('./sentiment0.json')
    entities_json = saltlux.read_json('./entities0.json')
    keywords_json = saltlux.read_json('./keyword0.json')

    polarity, score, sentiwords = saltlux.parse_sentiment_json(sentiment_json=sentiment_json)
    print(polarity)
    print(score)
    print(sentiwords)

    entities = saltlux.parse_entities_json(entities_json=entities_json)
    print(entities)

    keywords = saltlux.parse_keywords_json(keyword_json=keywords_json)
    print(keywords)
