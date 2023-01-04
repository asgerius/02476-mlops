FROM python:3.9-slim

RUN apt update
RUN apt install --no-install-recommends -y build-essential gcc git
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
RUN mkdir models

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT [ "python", "-u",  "src/models/train_model.py" ]
