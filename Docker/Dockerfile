FROM ubuntu:18.04


MAINTAINER Lezin Galibert "lezingalibert@gmail.com"


RUN apt -y update &&\
    apt -y install python3 python3-pip

RUN python3 -m pip install --upgrade pip

 
ADD ./python_requirements.txt /
RUN python3 -m pip install -r python_requirements.txt

ENTRYPOINT ["bash","/scripts/start_flask.sh"]