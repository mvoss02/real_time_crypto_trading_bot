# News Signal Microservice

The News Signal Microservice is a component of the crypto trading bot project. It fetches data from the news Kafka topic and outputs the corresponding cryptocurrency coin along with the trading signal.

## Features

- Fetches news data from a Kafka topic.
- Analyzes the sentiment of news articles using Claude or LLama as the LLM (Large Language Model).
- Outputs the corresponding cryptocurrency coin and the trading signal based on the sentiment analysis.

## How It Works

1. **Data Ingestion**: The microservice listens to thew  news Kafka topic for incoming news articles.
2. **Sentiment Analysis**: Utilizes Claude or LLama to analyze the sentiment of the news articles.
3. **Signal Generation**: Based on the sentiment analysis, the microservice generates a trading signal for the corresponding cryptocurrency coin.
4. **Output**: The generated signal and the corresponding coin are outputted for further processing by other components of the trading bot.

## Requirements

- Kafka
- Claude or LLama LLM