import asyncio
import json
import os
import random
from typing import Any

import dotenv
from core.base_model import (
    ProcessorProvider,
    Processor,
    ProcessorState, ProcessorStateDirection
)
from core.base_processor import (
    StatePropagationProviderRouterStateSyncStore,
    StatePropagationProviderDistributor,
    StatePropagationProviderRouterStateRouter, StatePropagationProviderRouter
)
from core.messaging.base_message_consumer_processor import BaseMessageConsumerProcessor
from core.messaging.base_message_router import Router
from core.messaging.nats_message_provider import NATSMessageProvider
from core.processor_state import State
from core.utils.ismlogging import ism_logger

from db.processor_state_db_storage import PostgresDatabaseStorage
from openai_lm import OpenAIChatCompletionProcessor
from openai_visual import OpenAIVisualCompletionProcessor

dotenv.load_dotenv()

# message routing file, used for both ingress and egress message handling
ROUTING_FILE = os.environ.get("ROUTING_FILE", '.routing.yaml')

# database related
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres1@localhost:5432/postgres")

# state storage specifically to handle this processor state (stateless obj)
storage = PostgresDatabaseStorage(
    database_url=DATABASE_URL,
    incremental=True
)

# nats messaging provider is used, the routes are defined in the routing.yaml
message_provider = NATSMessageProvider()

# routing the persistence of individual state entries to the state sync store topic
router = Router(
    provider=message_provider,
    yaml_file=ROUTING_FILE
)

# find the monitor route for telemetry updates
monitor_route = router.find_route("processor/monitor")
openai_route = router.find_route_by_subject("processor.models.openai")
state_sync_route = router.find_route('processor/state/sync')
state_router_route = router.find_route('processor/state/router')


class StatePropagationProviderRouterStateRouter2(StatePropagationProviderRouter):
    async def apply_state(self, processor: 'BaseProcessor',
                          input_query_state: Any,
                          output_query_states: [dict]) -> [dict]:

        output_state = processor.output_state

        # If the flag is set and the flat is false, then skip it
        if not processor.config.flag_auto_route_output_state:
            logging.debug(f'skipping auto route of output state events, for state id: {output_state.id}')
            return output_query_states

        # I know this is confusing, basically what we want to do is get all processors that
        # have a state input id = to the current output state of the previous processor
        processors_with_state_id_as_input = storage.fetch_processor_state_route(
            state_id=output_state.id,
            direction=ProcessorStateDirection.INPUT
        )   # if any

        # ensure that there are output processors for the newly created input (from the output of previous)
        if not processors_with_state_id_as_input:
            return output_query_states

        # iterate all output processors and submit query state entries to the state router
        for processor_state in processors_with_state_id_as_input:
            # create a new message for routing purposes
            route_message = {
                "route_id": processor_state.id,
                "type": "query_state_route",
                "input_query_state": input_query_state,
                "query_state": output_query_states
            }

            await self.route.publish(json.dumps(route_message))

        return output_query_states

        # return await super().apply_state(
        #     processor=processor,
        #     input_query_state=input_query_state,
        #     output_query_states=output_query_states,
        # )


# state_router_route = router.find_router("processor/monitor")
state_propagation_provider = StatePropagationProviderDistributor(
    propagators=[
        StatePropagationProviderRouterStateSyncStore(route=state_sync_route),
        # StatePropagationProviderRouterStateRouter2(route=state_router_route)
    ]
)

logging = ism_logger(__name__)


class MessagingConsumerOpenAI(BaseMessageConsumerProcessor):

    def create_processor(self,
                         processor: Processor,
                         provider: ProcessorProvider,
                         output_processor_state: ProcessorState,
                         output_state: State):

        logging.debug(f"received create processor request {provider.class_name}")

        if provider.class_name == "NaturalLanguageProcessing":

            processor = OpenAIChatCompletionProcessor(
                # storage class information
                state_machine_storage=storage,

                # state processing information
                output_state=output_state,
                provider=provider,
                processor=processor,
                output_processor_state=output_processor_state,

                # state information routing routers
                monitor_route=self.monitor_route,
                state_propagation_provider=state_propagation_provider
            )

        elif provider.class_name == "ImageProcessing":

            processor = OpenAIVisualCompletionProcessor(
                # storage class information
                state_machine_storage=storage,

                # state processing information
                output_state=output_state,
                provider=provider,
                processor=processor,
                output_processor_state=output_processor_state,

                # state information routing routers
                monitor_route=self.monitor_route,
                state_propagation_provider=state_propagation_provider
            )

        return processor

    # async def execute(self, consumer_message_mapping: dict):
    #     pass


if __name__ == '__main__':
    consumer = MessagingConsumerOpenAI(
        storage=storage,
        route=openai_route,
        monitor_route=monitor_route
    )

    # TODO think this through - important for workload creation. Here we randomly select a consumer number,
    #  and hope it does not collide with another consumer; if it does, then the consumer will throw an error
    #  and should exit, whereby next choosing a different number. Eventually, the consumer will find a spot.
    consumer_no = random.randint(0, 20)     # this should be a workload identity subscription

    consumer.setup_shutdown_signal()
    asyncio.get_event_loop().run_until_complete(consumer.start_consumer(consumer_no=consumer_no))
