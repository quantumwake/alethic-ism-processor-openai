import os.path
import openai
import dotenv
from core.base_processor_visual import BaseProcessorVisual
from core.monitored_processor_state import MonitoredUsage
from core.utils.ismlogging import ism_logger
from openai import OpenAI

dotenv.load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY', None)
openai.api_key = openai_api_key

logging = ism_logger(__name__)
logging.info(f'**** OPENAI API KEY (last 4 chars): {openai_api_key[-4:]} ****')


class OpenAIVisualCompletionProcessor(BaseProcessorVisual, MonitoredUsage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _execute(self, template: str, values: dict):
        template = template.strip()
        client = OpenAI()

        response = client.images.generate(
            model=self.provider.version,
            prompt=template,
            size=f"{self.config.width}x{self.config.height}",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        query_state_entry = {
            "image_url": image_url
        }

        # await self.send_usage_input_tokens(response.usage.prompt_tokens)
        # await self.send_usage_output_tokens(stream.usage.completion_tokens)

        return query_state_entry, 'json', None
