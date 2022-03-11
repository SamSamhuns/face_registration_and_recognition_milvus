FROM nvcr.io/nvidia/tritonserver:22.01-py3

# maintainer
LABEL maintainer="fname.lname@domain.com"

# install opencv & python-venv reqs
RUN apt-get update \
 && apt-get install libsm6 libxext6 libgl1-mesa-glx python3-venv -y

# set work directory
WORKDIR /app

# setup virtual env for python
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --default-timeout=100 -r requirements.txt

# copy all files to app exclude those in dockerignore
COPY . /app
