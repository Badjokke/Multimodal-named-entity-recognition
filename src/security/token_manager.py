token_path = "../token"
def get_access_token():
    with open(token_path,"r",encoding="utf-8") as f:
        return f.read()