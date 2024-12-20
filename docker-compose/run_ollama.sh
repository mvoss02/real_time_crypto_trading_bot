#!/bin/bash

echo "Starting Ollama server..."
ollama serve &  # Start Ollama in the background

echo "Ollama is ready, creating the model..."

# ollama create finetuned_mistral -f model_files/Modelfile
ollama run llama3.2:3b