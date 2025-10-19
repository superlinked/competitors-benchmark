from superlinked import framework as sl

from superlinked_app.apps.sota_app.config import open_ai_api_key

openai_config = sl.OpenAIClientConfig(api_key=open_ai_api_key, model="gpt-4o")
