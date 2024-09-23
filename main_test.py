import asyncio
import random

import dotenv
import nats
from nats.aio.client import Client as NATS
from nats.js import JetStreamContext, api
from nats.js.api import ConsumerConfig, AckPolicy, DeliverPolicy
from core.utils.ismlogging import ism_logger
from nats.js.errors import NotFoundError

dotenv.load_dotenv()
logger = ism_logger(__name__)


class Consumer:
    def __init__(self):
        logger.info("Creating consumer")
        self._nc: NATS = NATS()
        self._js: JetStreamContext = None
        self._sub: nats.js.client.PullSubscription = None

    async def create_stream(self):
        logger.info("Connecting to JetStream")
        self._js = self._nc.jetstream()
        stream_config = nats.js.api.StreamConfig(
            name="test_route",
            subjects=["test.route.*"],
            storage="file",
        )
        try:
            await self._js.find_stream_name_by_subject("test.route.*")
        except NotFoundError:
            # await self._js.add_stream(name=self.name, subjects=[self.subject])
            await self._js.add_stream(stream_config)

        logger.info("Connected to JetStream")
        return self._js

    async def subscribe(self):
        # Corrected subject to match the stream's subject configuration
        self._sub = await self._js.pull_subscribe(
            subject="test.route.*",
            durable="test_route_consumer",
            config=ConsumerConfig(
                ack_wait=5,
                deliver_policy=DeliverPolicy.ALL,
                ack_policy=AckPolicy.EXPLICIT,
                max_ack_pending=1000,
                flow_control=False,
            ),
        )

    async def connect(self):
        await self._nc.connect(
            servers=["nats://127.0.0.1:4222"],
        )

        await self.create_stream()

        logger.info("Subscribed to JetStream")
        return self._sub

    async def consume(self):
        count = 0
        while True:
            try:
                msgs = await self._sub.fetch(1, timeout=1)
                for msg in msgs:
                    data = msg.data.decode("utf-8")
                    logger.info(f"Message index: {count} - {data}")
                    # Simulate processing time
                    await asyncio.sleep(random.randint(1, 2))
                    # Acknowledge the message
                    logger.info(f"Acknowledging message index: {count}")
                    await msg.ack()
                    logger.info(f"Acknowledged message index: {count}")
                    count += 1
            except nats.js.errors.FetchTimeoutError:
                logger.info("No data received, waiting...")
                await asyncio.sleep(1)  # Prevent tight loop when no messages are available
            except Exception as e:
                logger.error(f"Error consuming messages: {e}")
                await asyncio.sleep(1)

    async def run(self):
        await self.connect()
        await self.consume()


if __name__ == "__main__":
    logger.info("Starting test")
    consumer = Consumer()
    asyncio.run(consumer.run())
    logger.info("Exiting test")
