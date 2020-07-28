from google.cloud import language_v1
from google.cloud.language_v1 import enums
from . import config


class GCP_Language:
    """
    Performs well on English Language Analysis. Not very good at Korean.
    """
    def __init__(self):
        self.credential_path = config.GCP_CREDENTIALS_PATH

    def analyze_sentiment(self, text_content):
        """
            Analyzing Sentiment in a String

            Args:
              text_content: The text content to analyze
        """
        client = language_v1.LanguageServiceClient.from_service_account_json(self.credential_path)

        # text_content = 'I am so happy and joyful.'

        # Available types: PLAIN_TEXT, HTML
        type_ = enums.Document.Type.PLAIN_TEXT

        # Optional. If not specified, the language is automatically detected.
        # For list of supported languages:
        # https://cloud.google.com/natural-language/docs/languages
        # language = "ko"  # sometimes text can contain "en"

        document = {"content": text_content, "type": type_}

        # Available values: NONE, UTF8, UTF16, UTF32
        encoding_type = enums.EncodingType.UTF8

        response = client.analyze_sentiment(document, encoding_type=encoding_type)
        # Get overall sentiment of the input document
        print(u"Document sentiment score: {}".format(response.document_sentiment.score))
        print(
            u"Document sentiment magnitude: {}".format(
                response.document_sentiment.magnitude
            )
        )
        # Get sentiment for all sentences in the document
        for sentence in response.sentences:
            print(u"Sentence text: {}".format(sentence.text.content))
            print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
            print(u"Sentence sentiment magnitude: {}".format(sentence.sentiment.magnitude))

        # Get the language of the text, which will be the same as
        # the language specified in the request or, if not specified,
        # the automatically-detected language.
        print(u"Language of the text: {}".format(response.language))

    def analyze_entities(self, text_content):
        """
        Analyzing Entities in a String

        Args:
          text_content: The text content to analyze
        """

        client = language_v1.LanguageServiceClient.from_service_account_json(self.credential_path)

        # text_content = 'California is a state.'

        # Available types: PLAIN_TEXT, HTML
        type_ = enums.Document.Type.PLAIN_TEXT

        # Optional. If not specified, the language is automatically detected.
        # For list of supported languages:
        # https://cloud.google.com/natural-language/docs/languages
        # language = "en"

        document = {"content": text_content, "type": type_}

        # Available values: NONE, UTF8, UTF16, UTF32
        encoding_type = enums.EncodingType.UTF8

        response = client.analyze_entities(document, encoding_type=encoding_type)

        # Loop through entitites returned from the API
        for entity in response.entities:
            print(u"Representative name for the entity: {}".format(entity.name))

            # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
            print(u"Entity type: {}".format(enums.Entity.Type(entity.type).name))

            # Get the salience score associated with the entity in the [0, 1.0] range
            print(u"Salience score: {}".format(entity.salience))

            # Loop over the metadata associated with entity. For many known entities,
            # the metadata is a Wikipedia URL (wikipedia_url) and Knowledge Graph MID (mid).
            # Some entity types may have additional metadata, e.g. ADDRESS entities
            # may have metadata for the address street_name, postal_code, et al.
            for metadata_name, metadata_value in entity.metadata.items():
                print(u"{}: {}".format(metadata_name, metadata_value))

            # Loop over the mentions of this entity in the input document.
            # The API currently supports proper noun mentions.
            for mention in entity.mentions:
                print(u"Mention text: {}".format(mention.text.content))

                # Get the mention type, e.g. PROPER for proper noun
                print(
                    u"Mention type: {}".format(enums.EntityMention.Type(mention.type).name)
                )

        # Get the language of the text, which will be the same as
        # the language specified in the request or, if not specified,
        # the automatically-detected language.
        print(u"Language of the text: {}".format(response.language))

    def analyze_entity_sentiment(self, text_content):
        """
        !! [IMPORTANT] GCP does not support Korean for this functionality !!

        Analyzing Entity Sentiment in a String

        Args:
          text_content The text content to analyze

        """

        client = language_v1.LanguageServiceClient.from_service_account_json(self.credential_path)

        # text_content = 'Grapes are good. Bananas are bad.'

        # Available types: PLAIN_TEXT, HTML
        type_ = enums.Document.Type.PLAIN_TEXT

        # Optional. If not specified, the language is automatically detected.
        # For list of supported languages:
        # https://cloud.google.com/natural-language/docs/languages
        # language = "en"
        document = {"content": text_content, "type": type_}

        # Available values: NONE, UTF8, UTF16, UTF32
        encoding_type = enums.EncodingType.UTF8

        response = client.analyze_entity_sentiment(document, encoding_type=encoding_type)
        # Loop through entitites returned from the API
        for entity in response.entities:
            print(u"Representative name for the entity: {}".format(entity.name))
            # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
            print(u"Entity type: {}".format(enums.Entity.Type(entity.type).name))
            # Get the salience score associated with the entity in the [0, 1.0] range
            print(u"Salience score: {}".format(entity.salience))
            # Get the aggregate sentiment expressed for this entity in the provided document.
            sentiment = entity.sentiment
            print(u"Entity sentiment score: {}".format(sentiment.score))
            print(u"Entity sentiment magnitude: {}".format(sentiment.magnitude))
            # Loop over the metadata associated with entity. For many known entities,
            # the metadata is a Wikipedia URL (wikipedia_url) and Knowledge Graph MID (mid).
            # Some entity types may have additional metadata, e.g. ADDRESS entities
            # may have metadata for the address street_name, postal_code, et al.
            for metadata_name, metadata_value in entity.metadata.items():
                print(u"{} = {}".format(metadata_name, metadata_value))

            # Loop over the mentions of this entity in the input document.
            # The API currently supports proper noun mentions.
            for mention in entity.mentions:
                print(u"Mention text: {}".format(mention.text.content))
                # Get the mention type, e.g. PROPER for proper noun
                print(
                    u"Mention type: {}".format(enums.EntityMention.Type(mention.type).name)
                )

        # Get the language of the text, which will be the same as
        # the language specified in the request or, if not specified,
        # the automatically-detected language.
        print(u"Language of the text: {}".format(response.language))

    def analyze_syntax(self, text_content):
        """
        Analyzing Syntax in a String

        Args:
          text_content The text content to analyze
        """

        client = language_v1.LanguageServiceClient.from_service_account_json(config.GCP_CREDENTIALS_PATH)

        # text_content = 'This is a short sentence.'

        # Available types: PLAIN_TEXT, HTML
        type_ = enums.Document.Type.PLAIN_TEXT

        # Optional. If not specified, the language is automatically detected.
        # language = "en"
        document = {"content": text_content, "type": type_}

        # Available values: NONE, UTF8, UTF16, UTF32
        encoding_type = enums.EncodingType.UTF8

        response = client.analyze_syntax(document, encoding_type=encoding_type)
        # Loop through tokens returned from the API
        for token in response.tokens:
            # Get the text content of this token. Usually a word or punctuation.
            text = token.text
            print(u"Token text: {}".format(text.content))
            print(
                u"Location of this token in overall document: {}".format(text.begin_offset)
            )
            # Get the part of speech information for this token.
            # Parts of spech are as defined in:
            # http://www.lrec-conf.org/proceedings/lrec2012/pdf/274_Paper.pdf
            part_of_speech = token.part_of_speech
            # Get the tag, e.g. NOUN, ADJ for Adjective, et al.
            print(
                u"Part of Speech tag: {}".format(
                    enums.PartOfSpeech.Tag(part_of_speech.tag).name
                )
            )
            # Get the voice, e.g. ACTIVE or PASSIVE
            print(u"Voice: {}".format(enums.PartOfSpeech.Voice(part_of_speech.voice).name))
            # Get the tense, e.g. PAST, FUTURE, PRESENT, et al.
            print(u"Tense: {}".format(enums.PartOfSpeech.Tense(part_of_speech.tense).name))
            # See API reference for additional Part of Speech information available
            # Get the lemma of the token. Wikipedia lemma description
            # https://en.wikipedia.org/wiki/Lemma_(morphology)
            print(u"Lemma: {}".format(token.lemma))
            # Get the dependency tree parse information for this token.
            # For more information on dependency labels:
            # http://www.aclweb.org/anthology/P13-2017
            dependency_edge = token.dependency_edge
            print(u"Head token index: {}".format(dependency_edge.head_token_index))
            print(
                u"Label: {}".format(enums.DependencyEdge.Label(dependency_edge.label).name)
            )

        # Get the language of the text, which will be the same as
        # the language specified in the request or, if not specified,
        # the automatically-detected language.
        print(u"Language of the text: {}".format(response.language))


# if __name__ == '__main__':
#     gcp = GCP_Language()
#     content1 = "Samsung is fucking awesome. However, Google is very bad."
#     content2 = "문제는 비용 증가. 3Q19, 낮아진 기대감도 하회: 3Q19 실적은 매출액 26조 9,689원(+10%YoY, +0%QoQ), 영업이익 3,785억원(+31%YoY, -69%QoQ)으로 일회성 비용에 대한 우려로 낮아진 시장 컨센서스를 하회했음. 본업의 개선이 절실: 일회성 비용을 제외한 3Q18 및 3Q19에 영업이익은 각각 7,800억원과 1조 620억원임. 녹록치 않은 환경: 미국 등 주요 지역의 수요 부진이 지속되고 있는 가운데 경쟁사들의 SUV 신차 출시 확대로 경쟁 강도가 상승 중임. 투자의견을 HOLD로 유지함."
#     content3 = "삼성의 주가는 상승했다. 네이버의 주가는 하락했다. 다음의 주가는 하락세이다. 네이버의 주가는 상승하는 중이다."
#
#     print("="*20)
#     gcp.analyze_sentiment(text_content=content3)
#     print("="*20)
#
#     gcp.analyze_entities(text_content=content3)
#
#     print("=" * 20)
#     #gcp.analyze_sentiment("상승")
#     # gcp.analyz_entity_sentiment(text_content=content2)
#
#     gcp.analyze_syntax(text_content=content3)
