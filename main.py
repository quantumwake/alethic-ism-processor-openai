import json
import os
import signal
import sys

import dotenv
import pulsar
import asyncio

from core.base_model import ProcessorStateDirection
from db.processor_state_db_storage import PostgresDatabaseStorage
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

# state storage specifically to handle this processor state (stateless obj)
state_storage = PostgresDatabaseStorage(
    database_url=STATE_DATABASE_URL,
    incremental=True
)


def close(consumer):
    consumer.close()


async def execute(message_dict: dict):
    message_type = message_dict['type']
    if message_type != 'query_state':
        raise NotImplemented(f'unsupported message type: {message_type}')

    processor_id = message_dict['processor_id']
    provider_id = message_dict['provider_id']

    # fetch provider details to identify the model version
    provider = state_storage.fetch_processor_provider(id=provider_id)

    # fetch the processors to forward the state query to, state must be an input of the state id
    output_states = state_storage.fetch_processor_state(processor_id=processor_id,
                                                        direction=ProcessorStateDirection.OUTPUT)

    if not output_states:
        raise BrokenPipeError(f'no output state found for processor id: {processor_id} provider {provider_id}')

    # fetch query state input entries
    query_states = message_dict['query_state']

    logging.info(f'found {len(output_states)} on processor id {processor_id} with provider {provider_id}')

    for state in output_states:
        # load the output state and relevant state instruction
        output_state = state_storage.load_state(state_id=state.state_id, load_data=False)

        logging.info(f'creating processor provider {processor_id} on output state id {state.state_id} with '
                     f'current index: {state.current_index}, '
                     f'maximum processed index: {state.maximum_index}, '
                     f'count: {state.count}')

        # create (or fetch cached state) processoring handling this state output instruction
        processor = OpenAIQuestionAnswerProcessor(
            output_state=output_state,
            provider=provider,
            state_machine_storage=state_storage,
        )

        # iterate each query state entry and forward it to the processor
        if isinstance(query_states, dict):
            logging.debug(f'submitting single query state entry count: solo, '
                          f'with processor_id: {processor_id}, '
                          f'provider_id: {provider_id}')

            processor.process_input_data_entry(input_query_state=query_states)
        elif isinstance(query_states, list):
            logging.debug(f'submitting batch query state entries count: {len(query_states)}, '
                          f'with processor_id: {processor_id}, '
                          f'provider_id: {provider_id}')

            # iterate each individual entry and submit
            # TODO modify to submit as a batch?? although this consumer should be handling 1 request
            for query_state_entry in query_states:
                processor.process_input_data_entry(input_query_state=query_state_entry)
        else:
            raise NotImplemented('unsupported query state entry, it must be a Dict or a List[Dict] where Dict is a '
                                 'key value pair of values, defining a single row and a column per key entry')

    try:
        # processor.process_input_data_entry(input_query_state=input_query_state)
        pass
    except Exception as exception:
        # processor_state.status = ProcessorStatus.FAILED
        logging.error(f'critical error {exception}')
    finally:
        pass
        # state_storage.update_processor_state(processor_state=processor_state)


async def qa_topic_management_consumer():
    while RUNNING:
        msg = None
        try:
            msg = management_consumer.receive()
            data = msg.data().decode("utf-8")
            logging.info(f'Message received with {data}')

            # the configuration of the state
            # processor_state = ProcessorState.model_validate_json(data)
            # if processor_state.status in [
            #     ProcessorStatus.TERMINATED,
            #     ProcessorStatus.STOPPED]:
            #     logging.info(f'terminating processor_state: {processor_state}')
            # TODO update the state, ensure that the state information is properly set,
            #  do not forward the msg unless the state has been terminated.

            # else:
            #     logging.info(f'nothing to do for processor_state: {processor_state}')
        except Exception as e:
            logging.error(e)
        finally:
            management_consumer.acknowledge(msg)


async def topic_consumer():
    while RUNNING:
        try:
            msg = processor_consumer.receive()
            data = msg.data().decode("utf-8")
            logging.info(f'Message received with {data}')

            # TODO change this to a model object
            message_dict = json.loads(data)

            if 'type' not in message_dict:
                raise ValidationError(f'unable to identity type for consumed message {data}')

            await execute(message_dict)

            # if 'state_id' not in message_dict:

            # TODO check whether the message is for the appropriate processor

            # the configuration of the state
            # processor_state = ProcessorState.model_validate_json(data)
            # processor_state = state_storage.fetch_processor_states_by(
            #     processor_id=processor_state.processor_id,
            #     input_state_id=processor_state.input_state_id,
            #     output_state_id=processor_state.output_state_id
            # )
            # if processor_state.status in [ProcessorStatus.QUEUED, ProcessorStatus.RUNNING]:
            #     await execute(processor_state=processor_state)
            # else:
            #     logging.error(f'status not in QUEUED, unable to processor state: {processor_state}  ')

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
    asyncio.run(topic_consumer())
