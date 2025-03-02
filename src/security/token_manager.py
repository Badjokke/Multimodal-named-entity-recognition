class TokenManager:
    def __init__(self):
        self.__token_path = "../token"
        self.__token = None
        self.__open_api_token_path = "../openapi_key"
        self.__open_api_key = None

    def get_access_token(self) -> str:
        return self.__token if self.__token is not None else self.__load_hf_token_from_file()

    def get_openapi_key(self) -> str:
        return self.__open_api_key if self.__open_api_key is not None else self.__load_openapi_token_from_file()

    def __load_hf_token_from_file(self) -> str:
        with open(self.__token_path, "r", encoding="utf-8") as f:
            self.__token = f.read()
            return self.__token

    def __load_openapi_token_from_file(self) -> str:
        with open(self.__open_api_token_path, "r", encoding="utf-8") as f:
            self.__open_api_key = f.read()
            return self.__open_api_key

    def invalidate_token(self):
        self.__token = None
