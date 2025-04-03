class TokenManager:
    def __init__(self):
        self.__token_path = "token"
        self.__token = None

    def get_access_token(self) -> str:
        return self.__token if self.__token is not None else self.__load_token_from_file()

    def __load_token_from_file(self) -> str:
        with open(self.__token_path, "r", encoding="utf-8") as f:
            self.__token = f.read()
            return self.__token

    def invalidate_token(self):
        self.__token = None
