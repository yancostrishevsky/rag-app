FROM ollama/ollama:latest

ENV OLLAMA_HOST="0.0.0.0:11434"

COPY ./entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

WORKDIR /root

EXPOSE 11434

ENTRYPOINT ["./entrypoint.sh"]