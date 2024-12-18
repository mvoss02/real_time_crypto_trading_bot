from typing import List

from llms.base import BaseNewsSignalExtractor
from loguru import logger
from quixstreams import Application


def add_signal_to_news(value: dict) -> dict:
    news_signal: List[dict] = llm.get_signal(value['title'], output_format='list')
    model_name = llm.model_name
    timestamp_ms = value['timestamp_ms']

    return [
        {
            'coin': n['coin'],
            'signal': n['signal'],
            'model_name': model_name,
            'timestamp_ms': timestamp_ms,
        }
        for n in news_signal
    ]


def main(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_output_topic: str,
    kafka_consumer_group: str,
    llm: BaseNewsSignalExtractor,
):
    logger.info('Hello from news-signal!')

    # create a unique id from current milliseconds
    # TODO: remove this once we are done debuggin
    import time

    unique_id = str(int(time.time() * 1000))
    kafka_consumer_group = f'{kafka_consumer_group}-{unique_id}'

    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
        auto_offset_reset='earliest',
    )

    input_topic = app.topic(
        name=kafka_input_topic,
        value_deserializer='json',
    )

    output_topic = app.topic(
        name=kafka_output_topic,
        value_serializer='json',
    )

    sdf = app.dataframe(input_topic)

    sdf = sdf.apply(add_signal_to_news, expand=True)

    sdf = sdf.update(lambda value: logger.debug(f'Final message: {value}'))
    # sdf = sdf.update(lambda value: breakpoint())

    sdf = sdf.to_topic(output_topic)

    app.run()


if __name__ == '__main__':
    from config import config
    from llms.factory import get_llm

    logger.info(f'Using model provider: {config.model_provider}')
    llm = get_llm(config.model_provider)

    main(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_output_topic=config.kafka_output_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        llm=llm,
    )
