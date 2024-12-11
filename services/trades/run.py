from typing import Union
from loguru import logger
from quixstreams import Application

from kraken_api.mock_data import KrakenMockAPI
from kraken_api.websocket import KrakenWebsocketAPI


def main(
    kafka_broker_address: str,
    kafka_topic: str,
    kraken_api: Union[KrakenMockAPI, KrakenWebsocketAPI],
):
    """
    Does 2 things:
    1. Reads trades from Kraken Websocket API
    2. Pushes trades to a Kraken topic

    Args:
        kafka_broker_address: str
        kafka_topic: str
        kraken_api: Union[KrakenMockAPI, KrakenWebsocketAPI]

    Returns:
        None
    """
    logger.info("Start the trades service")

    # Init the Quix Streams app.
    # This class handles all the low-level details to connect to Kafka.
    app = Application(
        broker_address=kafka_broker_address,
    )

    # Define a topic with JSON serialization
    topic = app.topic(name=kafka_topic, value_serializer="json")

    with app.get_producer() as producer:
        while True:
            trades = kraken_api.get_trades()

            for trade in trades:
                # Seralize the trade as bytes
                message = topic.serialize(
                    key=trade.pair,
                    value=trade.to_str(),
                )

                # Push the seralized message to the topic
                producer.produce(topic=topic.name, value=message.value, key=message.key)

                logger.info(f"Pushed trade to Kafka: {trade}")


if __name__ == "__main__":
    from config import config

    # Get the mock data
    # kraken_api = KrakenMockAPI(pair='BTC/USD')

    # Init the Kraken API and get real data from Kraken Websocket API
    kraken_api = KrakenWebsocketAPI(pairs=config.pairs)

    main(
        kafka_broker_address=config.kafka_broker_address,
        kafka_topic=config.kafka_topic,
        kraken_api=kraken_api,
    )
