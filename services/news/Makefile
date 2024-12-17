run-dev:
	uv run python run.py

build:
	docker build -f Dockerfile -t news .

run: build
	docker run -it \
			--network redpanda_network \
    		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
			--env-file cryptopanic_credentials.env \
    		-v news_volume:/app/state \
			news:latest