
import json
import os.path
from typing import Any

import openai
import dotenv
from core.base_processor_lm import BaseProcessorLM
from core.utils.ismlogging import ism_logger

from openai import OpenAI, Stream
from core.utils.general_utils import parse_response

dotenv.load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY', None)
openai.api_key = openai_api_key

logging = ism_logger(__name__)
logging.info(f'**** OPENAI API KEY (last 4 chars): {openai_api_key[-4:]} ****')


class OpenAIChatCompletionProcessor(BaseProcessorLM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # async def process_input_data_entry(self, input_query_state: dict, force: bool = False):
    #     return []

    def calculate_usage(self, response: Any):

        if isinstance(response, Stream):
            completion_tokens = response.usage.prompt_tokens
            prompt_tokens = response.usage.prompt_tokens
            total_tokens = response.usage.total_tokens

    async def _stream(self, input_data: Any, template: str):
        if not template:
            template = str(input_data)

        # Prepare the message dictionary with the query_string
        if 'session_id' in input_data:
            session_id = input_data['session_id']
            new_message = {"role": "user", "content": template.strip()}
            self.storage.insert_session_message(session_id, json.dumps(new_message))

        # Add history to the messages if provided (WHICH INCLUDES THE NEW MESSAGE)
        messages_dict = self.fetch_session_data(input_data)
        # messages_dict = []    # TODO FLAG: OFF history flag injected here

        client = OpenAI()

        # Create a streaming completion
        stream = client.chat.completions.create(
            model=self.provider.version,
            messages=messages_dict,
            max_tokens=4096,
            stream=True,  # Enable streaming
        )

        # Iterate over the streamed responses and yield the content
        response_content = []
        for chunk in stream:
            content = chunk.choices[0].delta.content
            response_content.append(content)
            yield content

    # def process_input_data_entry(self, input_query_state: dict, force: bool = False):
    #     input_query_state = [
    #         input_query
    #         for input_query in input_query_state
    #         if not isinstance(input_query, list)
    #     ]
    #     return super().process_input_data_entry(input_query_state=input_query_state, force=force)

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

        # calcualte the usage
        # calculate_usage(response=stream)

        # final raw response, without stripping or splitting
        raw_response = stream.choices[0].message.content
        return parse_response(raw_response=raw_response)
