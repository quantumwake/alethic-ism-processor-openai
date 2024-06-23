import json
import os.path
import openai
import dotenv
from core.base_message_provider import Monitorable
from core.base_message_route_model import Route
from core.base_processor_lm import BaseProcessorLM

from logger import log
from openai import OpenAI
from core.base_message_router import Router
from core.utils.general_utils import parse_response

dotenv.load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY', None)
openai.api_key = openai_api_key

logging = log.getLogger(__name__)
logging.info(f'**** OPENAI API KEY (last 4 chars): {openai_api_key[-4:]} ****')


class OpenAIChatCompletionProcessor(BaseProcessorLM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # async def process_input_data_entry(self, input_query_state: dict, force: bool = False):
    #     return []

    def _execute(self, user_prompt: str, system_prompt: str, values: dict):
        messages_dict = []

        if user_prompt:
            user_prompt = user_prompt.strip()
            messages_dict.append({
                "role": "user",
                "content": f"{user_prompt}"
            })

        if system_prompt:
            system_prompt = system_prompt.strip()
            messages_dict.append({
                "role": "system",
                "content": system_prompt
            })

        if not messages_dict:
            raise Exception(f'no prompts specified for values {values}')

        client = OpenAI()

        stream = client.chat.completions.create(
            model=self.provider.version,
            messages=messages_dict,
            stream=False,
        )

        # final raw response, without stripping or splitting
        raw_response = stream.choices[0].message.content
        return parse_response(raw_response=raw_response)
