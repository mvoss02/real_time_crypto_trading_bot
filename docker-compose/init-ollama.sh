#!/bin/bash
set -e

# Install dependencies
apt-get update && apt-get install -y curl

# Start Ollama server
echo "Starting Ollama server..."
ollama serve &
SERVER_PID=$!

# Wait for server with timeout
echo "Waiting for server to be ready..."
TIMEOUT=60
COUNTER=0
while ! curl -s http://localhost:11434/api/tags >/dev/null; do
    if [ $COUNTER -gt $TIMEOUT ]; then
        echo "Timeout waiting for server"
        exit 1
    fi
    sleep 1
    COUNTER=$((COUNTER+1))
done

# Pull model with verification
echo "Pulling llama3.2:3b model..."
if ! ollama pull llama3.2:3b; then
    echo "Failed to pull model"
    exit 1
fi

# Verify model is available
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "Model not found after pull"
    exit 1
fi

echo "Model successfully initialized"
# Keep container running with original server process
wait $SERVER_PID


# Needs to be made executable: chmod +x init-ollama.sh