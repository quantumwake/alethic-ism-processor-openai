from typing import List

import os.path
import openai
import dotenv
from core.base_processor import BaseProcessor, ThreadQueueManager
from core.base_question_answer_processor import BaseQuestionAnswerProcessor
from core.processor_state import State, ProcessorStatus
from core.processor_state_storage import ProcessorStateStorage, ProcessorState
from core.utils.general_utils import parse_response
from tenacity import retry, wait_exponential, wait_random, retry_if_not_exception_type
from logger import log
from openai import OpenAI

dotenv.load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY', None)
openai.api_key = openai_api_key

logging = log.getLogger(__name__)
logging.info(f'**** OPENAI API KEY (last 4 chars): {openai_api_key[-4:]} ****')


class BaseQuestionAnswerProcessorDatabaseStorage(BaseQuestionAnswerProcessor):

    @property
    def user_template(self):
        template = self.storage.fetch_template(self.user_template_path)
        return template.template_content

    @property
    def system_template(self):
        template = self.storage.fetch_template(self.system_template_path)
        return template.template_content

    def __init__(self,
                 state: State,
                 processor_state: ProcessorState,
                 processors: List[BaseProcessor] = None,
                 storage: ProcessorStateStorage = None,
                 *args, **kwargs):

        super().__init__(state=state, processors=processors, **kwargs)

        self.manager = ThreadQueueManager(num_workers=10, processor=self)
        self.processor_state = processor_state
        self.storage = storage
        logging.info(f'extended instruction state machine: {type(self)} with config {self.config}')

    def pre_state_apply(self, query_state: dict):
        return super().pre_state_apply(query_state=query_state)

    def post_state_apply(self, query_state: dict):
        query_state = super().post_state_apply(query_state=query_state)
        self.storage.save_state(state=self.state)
        return query_state

    def load_previous_state(self, force: bool = False):
        # do nothing since this was meant to be for loading previous state
        pass

    def build_state_storage_path(self):
        # TODO look into this, not sure if this is a decent method
        return (f'{self.config.name}'
                f':{self.config.version}'
                f':{type(self.config).__name__}'
                f':{self.provider_name}'
                f':{self.model_name}')

    def update_current_status(self, new_status: ProcessorStatus):
        self.processor_state.status = new_status
        self.storage.update_processor_state(self.processor_state)

    def get_current_status(self):
        current_state = self.storage.fetch_processor_states_by(
            processor_id=self.processor_state.processor_id,
            input_state_id=self.processor_state.input_state_id,
            output_state_id=self.processor_state.output_state_id
        )
        return current_state.status


class OpenAIQuestionAnswerProcessor(BaseQuestionAnswerProcessorDatabaseStorage):

    @retry(
        retry=retry_if_not_exception_type(SyntaxError),
        wait=wait_exponential(multiplier=1, min=4, max=10) + wait_random(0, 2))
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
            model=self.model_name,
            messages=messages_dict,
            stream=False,
        )

        # final raw response, without stripping or splitting
        raw_response = stream.choices[0].message.content
        return parse_response(raw_response=raw_response)

