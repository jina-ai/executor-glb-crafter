FROM jinaai/jina:2.0.23-py37-standard

RUN apt-get -y update && apt-get install -y libenchant-2-2

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY ./ /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml", "--native", "true"]
