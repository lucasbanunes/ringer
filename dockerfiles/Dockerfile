FROM nvcr.io/nvidia/tensorflow:22.07-tf2-py3

# Exposing a port in case of necessity
EXPOSE 8888

#Installing extra dependencies
COPY setup.sh .
COPY requirements.txt .
RUN bash ./setup.sh
CMD jupyter lab --port 8888 --no-browser