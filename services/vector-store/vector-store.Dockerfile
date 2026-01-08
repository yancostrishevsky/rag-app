FROM qdrant/qdrant

RUN apt update && apt install -y curl

EXPOSE ${EXPOSED_PORT_REST}
EXPOSE ${EXPOSED_PORT_GRPC}
