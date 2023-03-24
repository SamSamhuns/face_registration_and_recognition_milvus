FROM nvcr.io/nvidia/tritonserver:22.12-py3

LABEL maintainer="fname.lname@domain.com"

# install opencv & python-venv reqs
RUN apt-get update --no-install-recommends \
    && apt-get install libsm6 libxext6 libgl1-mesa-glx python3-venv --no-install-recommends -y

# set username & uid inside docker
ARG UNAME=triton-server

# set work directory
WORKDIR "/home/$UNAME/src"

# setup python env vars & virtual env
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV VIRTUAL_ENV="/home/$UNAME/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies
RUN pip install --upgrade pip
COPY requirements.txt  "/home/$UNAME/src/requirements.txt"
RUN pip install --no-cache-dir --default-timeout=100 -r "/home/$UNAME/src/requirements.txt"

# remove cache
RUN pip cache purge \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

# copy all files to src & exclude those in dockerignore
COPY . "/home/$UNAME/src"

# change file ownership to docker user
RUN chown -R "$UNAME" "/home/$UNAME/src"

# make server start script executable
RUN chmod 777 "/home/$UNAME/src/scripts/start_servers.sh"

USER "$UNAME"
CMD ["/bin/bash", "/home/triton-server/src/scripts/start_servers.sh"]