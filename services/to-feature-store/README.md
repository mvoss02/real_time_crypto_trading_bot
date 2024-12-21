# Crypto Trading Bot - Feature Store Service

This microservice is responsible for pushing features for the machine learning model to Hopsworks. The features consist of news sentiment signals and 60-second candles. We store the most recent data to the online feature store and features for training to the offline feature store.

## Features

- **News Sentiment Signals**: Extracted from various news sources to gauge market sentiment.
- **60-Second Candles**: Aggregated trading data in 60-second intervals.

## Data Storage

- **Online Feature Store**: Stores the most recent data for real-time predictions.
- **Offline Feature Store**: Stores historical data for training the machine learning model.