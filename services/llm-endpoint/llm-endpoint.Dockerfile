FROM ollama/ollama:latest

COPY ./entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

WORKDIR /root

EXPOSE 11434

ENTRYPOINT ["./entrypoint.sh"]