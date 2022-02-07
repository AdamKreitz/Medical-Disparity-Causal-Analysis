FROM python:3

USER root

RUN pip install numpy
RUN pip install pandas
RUN pip install causal-learn

COPY . .

CMD [ "python", "./src/scripts/run.py" ]