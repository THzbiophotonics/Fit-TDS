FROM python:2.7

MAINTAINER benjamin.lecha@iemn.fr

RUN apt-get update && apt-get install -y mpich swig

RUN mkdir /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install git+https://github.com/madebr/pyOpt.git

ENV PYTHONPATH=".:$PYTHONPATH"

STOPSIGNAL SIGINT

COPY . ./app

CMD ["python","app/fit@TDS.py"]
