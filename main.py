# The Alethic Instruction-Based State Machine (ISM) is a versatile framework designed to 
# efficiently process a broad spectrum of instructions. Initially conceived to prioritize
# animal welfare, it employs language-based instructions in a graph of interconnected
# processing and state transitions, to rigorously evaluate and benchmark AI models
# apropos of their implications for animal well-being. 
# 
# This foundation in ethical evaluation sets the stage for the framework's broader applications,
# including legal, medical, multi-dialogue conversational systems.
# 
# Copyright (C) 2023 Kasra Rasaee, Sankalpa Ghose, Yip Fai Tse (Alethic Research) 
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 
# 
import os
import signal
import sys

import dotenv
import pulsar
import asyncio

from core.processor_state import ProcessorStatus
from core.processor_state_storage import ProcessorState
from db.processor_state_db_storage import ProcessorStateDatabaseStorage
from pydantic import ValidationError
from processor_question_answer import OpenAIQuestionAnswerProcessor
from logger import logging

dotenv.load_dotenv()

logging.info('starting up pulsar consumer for openai state processor.')

# pulsar/kafka related
MSG_URL = os.environ.get("MSG_URL", "pulsar://localhost:6650")
MSG_TOPIC = os.environ.get("MSG_TOPIC", "ism_openai_qa")
MSG_MANAGE_TOPIC = os.environ.get("MSG_MANAGE_TOPIC", "ism_openai_manage_topic")
MSG_TOPIC_SUBSCRIPTION = os.environ.get("MSG_TOPIC_SUBSCRIPTION", "ism_openai_qa_subscription")

# database related
STATE_DATABASE_URL = os.environ.get("STATE_DATABASE_URL", "postgresql://postgres:postgres1@localhost:5432/postgres")

# flag that determines whether to shut down the consumers
RUNNING = True

# consumer config
client = pulsar.Client(MSG_URL)
processor_consumer = client.subscribe(MSG_TOPIC, MSG_TOPIC_SUBSCRIPTION)
management_consumer = client.subscribe(MSG_MANAGE_TOPIC, MSG_TOPIC_SUBSCRIPTION)

#
# state storage specifically to handle this processor state (stateless obj)
#
state_storage = ProcessorStateDatabaseStorage(
    database_url=STATE_DATABASE_URL,
    incremental=True
)


def close(consumer):
    consumer.close()


async def execute(processor_state: ProcessorState):
    input_state = state_storage.load_state(processor_state.input_state_id)
    output_state = state_storage.load_state(processor_state.output_state_id)
    processor_info = state_storage.fetch_processor(
        processor_id=processor_state.processor_id
    )

    # process the input state
    processor = OpenAIQuestionAnswerProcessor(
        state=output_state,
        storage=state_storage,
        processor_state=processor_state
    )

    try:
        processor(input_state=input_state)
    except Exception as exception:
        processor_state.status = ProcessorStatus.FAILED
        logging.error(f'critical error {exception}')
    finally:
        state_storage.update_processor_state(processor_state=processor_state)


async def qa_topic_management_consumer():
    while RUNNING:
        try:
            msg = management_consumer.receive()
            data = msg.data().decode("utf-8")
            logging.info(f'Message received with {data}')

            # the configuration of the state
            processor_state = ProcessorState.model_validate_json(data)
            if processor_state.status in [
                ProcessorStatus.TERMINATED,
                ProcessorStatus.STOPPED]:
                logging.info(f'terminating processor_state: {processor_state}')
                # TODO update the state, ensure that the state information is properly set, 
                #  do not forward the msg unless the state has been terminated.
            
            else:
                logging.info(f'nothing to do for processor_state: {processor_state}')
        except Exception as e:
            logging.error(e)


async def qa_topic_consumer():
    while RUNNING:
        try:
            msg = processor_consumer.receive()
            data = msg.data().decode("utf-8")
            logging.info(f'Message received with {data}')

            # TODO check whether the message is for the appropriate processor
            
            # the configuration of the state
            processor_state = ProcessorState.model_validate_json(data)
            processor_state = state_storage.fetch_processor_states_by(
                processor_id=processor_state.processor_id,
                input_state_id=processor_state.input_state_id,
                output_state_id=processor_state.output_state_id
            )
            if processor_state.status in [ProcessorStatus.QUEUED, ProcessorStatus.RUNNING]:
                await execute(processor_state=processor_state)
            else:
                logging.error(f'status not in QUEUED, unable to processor state: {processor_state}  ')

            # send ack that the message was consumed.
            processor_consumer.acknowledge(msg)

            # Log success
            # logger.info(
            #     f"Message successfully consumed and stored with asset id {asset.id} for account {asset.library_id}")
        except pulsar.Interrupted:
            logging.error("Stop receiving messages")
            break
        except ValidationError as e:
            # it is safe to assume that if we get a validation error, there is a problem with the json object
            # TODO throw into an exception log or trace it such that we can see it on a dashboard
            processor_consumer.acknowledge(msg)
            logging.error(f"Message validation error: {e} on asset data {data}")
        except Exception as e:
            processor_consumer.acknowledge(msg)
            # TODO need to send this to a dashboard, all excdptions in consumers need to be sent to a dashboard
            logging.error(f"An error occurred: {e} on asset data {data}")


def graceful_shutdown(signum, frame):
    global RUNNING
    print("Received SIGTERM signal. Gracefully shutting down.")
    RUNNING = False

    sys.exit(0)


# Attach the SIGTERM signal handler
signal.signal(signal.SIGTERM, graceful_shutdown)

if __name__ == '__main__':
    asyncio.run(qa_topic_consumer())
