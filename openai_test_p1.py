import os

import openai
import dotenv
from core.utils.general_utils import parse_response
from openai import OpenAI

dotenv.load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

system_prompt = "For each field, only use text with no newlines."
user_prompt = """Give me some pig meat recipes

```json
{
    "response": "[text only]",
    "justification": "[text justification only]"
}
```
"""

# otherwise process the question
messages_dict = []

messages_dict.append({
    "role": "user",
    "content": f"{user_prompt.strip()}"
})

messages_dict.append({
    "role": "system",
    "content": f"{system_prompt.strip()}"
})

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=messages_dict,
    stream=False,
)

# final raw response, without stripping or splitting
raw_response = stream.choices[0].message.content
response, data_type, raw_response = parse_response(raw_response=raw_response)
