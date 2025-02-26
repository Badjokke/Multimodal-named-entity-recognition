from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.response_format = "json"

    def call_chat_completion_api(self, system_prompt: dict, user_prompt: dict) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            store=True,
            response_format={"type": "json_object"},
            messages=[
                system_prompt,
                user_prompt
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content