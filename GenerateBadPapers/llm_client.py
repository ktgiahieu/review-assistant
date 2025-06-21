# llm_client.py
from openai import AzureOpenAI


class LLMClient:
    """
    Thin wrapper around Azure OpenAI's chat completion API.
    """

    def __init__(
        self,
        api_key: str,
        api_version: str,
        endpoint: str,
        deployment: str,
    ) -> None:
        """Initialize the AzureOpenAI client with credentials."""
        self.deployment = deployment
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

    def get_llm_response(self, messages: list[dict]) -> str:
        """
        Send a list of `messages` and return the assistant's reply text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            return None
