#!/bin/sh

ollama serve &
SERVER_PID=$!

echo "Waiting for ollama server to start..."
sleep 5

echo "Creating model from Modelfile..."
sed -i "s/\${LLM_GENERATOR_MODEL}/${LLM_GENERATOR_MODEL}/g" Modelfile
ollama create -f Modelfile local_ollama_model

ollama run local_ollama_model

wait $SERVER_PID