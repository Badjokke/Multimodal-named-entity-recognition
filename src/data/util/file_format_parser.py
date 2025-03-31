class FileFormatParser:

    @staticmethod
    def parse_t15conll_file(file: str) -> list[dict[str, list]]:
        """
        img_id, words, labels for each tweet
        """
        assert file is not None
        tweets = file.split("\n\n")
        result = []
        for tweet in tweets:
            items = tweet.split("\n")
            img_id = FileFormatParser.__get_img_id(items[0])
            words, labels = FileFormatParser.__get_words_and_labels(items[1:])
            result.append(FileFormatParser.__to_t17_json(img_id, words, labels))
        return result

    @staticmethod
    def __get_words_and_labels(items: list[str]) -> tuple[list[str], list[str]]:
        words = []
        labels = []
        for item in items[1:]:
            tmp = item.split("\t")
            words.append(tmp[0])
            labels.append(tmp[1])
        return words, labels

    @staticmethod
    def __get_img_id(img_ref: str) -> str:
        assert img_ref.startswith("IMGID:")
        return img_ref[6:]

    @staticmethod
    def __to_t17_json(img_id, words, labels) -> dict[str, list]:
        return {"image": [f"{img_id}.jpeg"], "text": words, "label": labels}
