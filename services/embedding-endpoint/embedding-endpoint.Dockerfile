FROM ollama/ollama:latest

COPY ./entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

WORKDIR /root

EXPOSE ${EXPOSED_PORT}

ENTRYPOINT ["./entrypoint.sh"]