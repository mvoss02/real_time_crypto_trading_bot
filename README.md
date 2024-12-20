# Real Time Crypto Trading Bot

This repository contains the code for a Real-Time Crypto Currency Price Predictor, a ML system designed to predict cryptocurrency price movements by combining real-time market data with news sentiment analysis. Various LLMs are being employed to obtain the sentiment scores. Ranging from free options, such as Llama, to paid optionsm, such as Claude from Anthropic. 


## Overview
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The main data scources are:

1. https://cryptopanic.com/ - where we retrieve live news within the crypto space
    1.1. Historic data has been retrieved from https://github.com/soheilrahsaz/cryptoNewsDataset, as there is no API to obtain histoirc news headlines from cryptopanic
2. https://docs.kraken.com/api/
    2.1. Live data is being retrieved from the Websocket API
    2.2. Historical data from the REST API

## Prerequisites
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
- uv 
- Docker and Docker Compose
- Kafka broker
- Python 3.12+
- Make
- (A GPU if you are planning on fine tuning your own LLM - could also be done via cloud platform provider)

## Todos: 
- Bakcfill all the services with historic data
- Ingest the data in order to train and serve a ML model