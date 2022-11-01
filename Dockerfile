FROM nvcr.io/nvidia/tensorflow:22.07-tf2-py3

# Exposing jupyter port
EXPOSE 8888

#Installing extra dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD jupyter lab --port 8888 --no-browser