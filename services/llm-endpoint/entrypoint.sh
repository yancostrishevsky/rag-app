#!/bin/sh

ollama serve &
SERVER_PID=$!

echo "Waiting for ollama server to start..."
sleep 5

echo "Pulling the model..."
ollama run "${LLM_GENERATOR_MODEL}"

wait $SERVER_PID