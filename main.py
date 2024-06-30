import os
import dotenv
from core.base_message_consumer_lm import BaseMessagingConsumerLM
from core.base_message_router import Router
from core.base_model import ProcessorProvider, Processor, ProcessorState, ProcessorStatusCode
from core.base_processor import StatePropagationProviderRouterStateSyncStore, StatePropagationProviderDistributor, \
    StatePropagationProviderRouterStateRouter
from core.processor_state import State
from core.pulsar_message_producer_provider import PulsarMessagingProducerProvider
from core.pulsar_messaging_provider import PulsarMessagingConsumerProvider
from db.processor_state_db_storage import PostgresDatabaseStorage

from logger import logging
from openai_lm import OpenAIChatCompletionProcessor
from openai_visual import OpenAIVisualCompletionProcessor

dotenv.load_dotenv()
MSG_URL = os.environ.get("MSG_URL", "pulsar://localhost:6650")
MSG_TOPIC = os.environ.get("MSG_TOPIC", "ism_openai_qa")
MSG_MANAGE_TOPIC = os.environ.get("MSG_MANAGE_TOPIC", "ism_openai_manage_topic")
MSG_TOPIC_SUBSCRIPTION = os.environ.get("MSG_TOPIC_SUBSCRIPTION", "ism_openai_qa_subscription")

# Message Routing File (
#   Used for routing processed messages, e.g: input comes in,
#   processed and output needs to be routed to the connected edges/processors
# )
ROUTING_FILE = os.environ.get("ROUTING_FILE", '.routing.yaml')
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# database related
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres1@localhost:5432/postgres")

# state storage specifically to handle this processor state (stateless obj)
storage = PostgresDatabaseStorage(
    database_url=DATABASE_URL,
    incremental=True
)

messaging_provider = PulsarMessagingConsumerProvider(
    message_url=MSG_URL,
    message_topic=MSG_TOPIC,
    message_topic_subscription=MSG_TOPIC_SUBSCRIPTION,
    management_topic=MSG_MANAGE_TOPIC
)

# pulsar messaging provider is used, the routes are defined in the routing.yaml
pulsar_provider = PulsarMessagingProducerProvider()

# routing the persistence of individual state entries to the state sync store topic
router = Router(
    provider=pulsar_provider,
    yaml_file=ROUTING_FILE
)

# find the monitor route for telemetry updates
monitor_route = router.find_router("processor/monitor")

# state_router_route = router.find_router("processor/monitor")
state_propagation_provider = StatePropagationProviderDistributor(
    propagators=[
        StatePropagationProviderRouterStateSyncStore(route=router.find_router('state/sync/store')),
        StatePropagationProviderRouterStateRouter(route=router.find_router('state/router'))
    ]
)

class MessagingConsumerOpenAI(BaseMessagingConsumerLM):

    def create_processor(self,
                         processor: Processor,
                         provider: ProcessorProvider,
                         output_processor_state: ProcessorState,
                         output_state: State):

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
        name="MessagingConsumerOpenAI",
        storage=storage,
        messaging_provider=messaging_provider,
        monitor_route=monitor_route
    )

    consumer.setup_shutdown_signal()
    consumer.start_topic_consumer()
