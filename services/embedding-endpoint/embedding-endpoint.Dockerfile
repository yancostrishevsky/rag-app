FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y curl

COPY ./entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

WORKDIR /root

EXPOSE ${EXPOSED_PORT}

ENTRYPOINT ["./entrypoint.sh"]