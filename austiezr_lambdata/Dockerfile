FROM debian


### So logging/io works reliably w/Docker
ENV PYTHONUNBUFFERED=1
### UTF Python issue for Click package (pipenv dependency)
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
### Need to explicitly set this so `pipenv shell` works
ENV SHELL=/bin/bash

COPY austiezr_lambdata.py austiezr_lambdata.py

### Basic Python dev dependencies
RUN apt-get update && \
  apt-get upgrade && \
  apt-get install python3-pip -y && \
  pip3 install --upgrade pip && \
  pip3 install --upgrade setuptools wheel scikit-learn pandas numpy

CMD python3