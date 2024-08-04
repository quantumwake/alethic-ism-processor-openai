import asyncio
import os.path
import time
from typing import Any

import openai
import dotenv
from core.base_processor_lm import BaseProcessorLM
from core.messaging.base_message_router import Router
from core.messaging.nats_message_provider import NATSMessageProvider
from core.processor_state import StateConfigStream
from core.utils.ismlogging import ism_logger

from openai import OpenAI, Stream
from core.utils.general_utils import parse_response, build_template_text

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

    async def stream_input_data_entry(self, input_query_state: dict):
        if not input_query_state:
            raise ValueError("invalid input state, cannot be empty")

        if not isinstance(self.config, StateConfigStream):
            raise NotImplementedError()

        status, template = build_template_text(self.template, input_query_state)

        # begin the processing of the prompts
        logging.debug(f"entered streaming mode, state_id: {self.output_state.id}")
        try:
            # # execute the underlying model function
            stream = self._stream(
                input_data=input_query_state,
                template=template,
            )

            message_provider = NATSMessageProvider()
            router = Router(provider=message_provider, yaml_file="../routing-nats.yaml")
            state_route = router.clone_route(
                selector="processor/state",
                route_config_updates={
                    "subject": f"processor.state.{self.output_state.id}",
                    # "subject": "processor.state",
                    # "subject": "processor.state.d1df2c10",
                    "name": f"processor_state_{self.output_state.id}".replace("-", "_")
                }
            )

            async for content in stream:
                if content:
                    # print(content)
                    await state_route.publish(content)

            await state_route.publish("<<>>DONE<<>>")
            await state_route.flush()
            # time.sleep(1)

            # should gracefully close the connection
            await state_route.disconnect()

            # try:
            #     await state_route.disconnect()
            # except:
            #     pass

            logging.debug(f"exit streaming mode, state_id: {self.output_state.id}")


        except Exception as exception:
            await self.fail_execute_processor_state(
                # self.output_processor_state,
                route_id=self.output_processor_state.id,
                exception=exception,
                data=input_query_state
            )

    async def _stream(self, input_data: Any, template: str):
        if not template:
            template = str(input_data)

        # Prepare the message dictionary with the query_string
        messages_dict = [
            {"role": "user", "content": template.strip()}
        ]

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
