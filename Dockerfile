FROM python:3.8.14

ARG default_jupyter_port=8888
ENV jupyter_port=${default_jupyter_port}

#Installing extra dependencies
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD jupyter lab --port ${jupyter_port} --no-browser --allow-root --ip 0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''