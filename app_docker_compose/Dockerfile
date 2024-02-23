FROM nvcr.io/nvidia/tritonserver:24.01-py3

LABEL maintainer="fname.lname@domain.com"

# install opencv & python-venv reqs
RUN apt-get update --no-install-recommends \
    && apt-get install libsm6 libxext6 libgl1-mesa-glx python3-venv --no-install-recommends -y

# set username & uid inside docker
ARG UNAME=triton-server
ENV WORKDIR="/home/$UNAME/src"

# set work directory
WORKDIR ${WORKDIR}

# setup python env vars & virtual env
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV VIRTUAL_ENV="/home/$UNAME/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies
RUN pip install pip==24.0
COPY requirements.txt  "$WORKDIR/requirements.txt"
RUN pip install --no-cache-dir --default-timeout=100 -r "$WORKDIR/requirements.txt"

# remove cache
RUN pip cache purge \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip

# copy app & scripts to src & exclude those in dockerignore
COPY app "$WORKDIR/app"
COPY scripts "$WORKDIR/scripts"

# change file ownership to docker user:usergroup
RUN chown -R "$UNAME:$UNAME" "$WORKDIR"

# make server start script executable
RUN chmod 777 "$WORKDIR/scripts/start_servers.sh"

USER "$UNAME"
CMD ["/bin/bash", "/home/triton-server/src/scripts/start_servers.sh"]