FROM nvcr.io/nvidia/tritonserver:22.12-py3

LABEL maintainer="fname.lname@domain.com"

# install opencv & python-venv reqs
RUN apt-get update \
    && apt-get install libsm6 libxext6 libgl1-mesa-glx python3-venv --no-install-recommends -y

# set work directory
WORKDIR /src

# set temp download env var
ENV TEMP_DOWNLOAD_URL="/src/app/.data_cache"
# setup python env vars & virtual env
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --default-timeout=100 -r requirements.txt

# remove cache
RUN pip cache purge \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

# copy all files to src & exclude those in dockerignore
COPY . /src

# make server start script executable
RUN chmod 777 /src/scripts/start_servers.sh
CMD ["/bin/bash", "/src/scripts/start_servers.sh"]