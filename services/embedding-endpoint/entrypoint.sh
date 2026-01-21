#!/bin/sh

ollama serve &
SERVER_PID=$!

echo "Waiting for ollama server to start..."
sleep 5

echo "Loading the model..."
ollama pull ${EMBEDDING_MODEL}

wait $SERVER_PID