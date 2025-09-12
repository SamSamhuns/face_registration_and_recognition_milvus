# Stage 1: Builder stage
FROM nvcr.io/nvidia/tritonserver:25.02-py3 AS builder

# Install build dependencies
RUN apt-get update --no-install-recommends \
    && apt-get install python3-venv --no-install-recommends -y

# Set build environment
ARG UNAME=triton-server
ENV WORKDIR="/home/$UNAME/src"
ENV VIRTUAL_ENV="/home/$UNAME/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

# Create directories
RUN mkdir -p ${WORKDIR} ${VIRTUAL_ENV}

# Set working directory
WORKDIR ${WORKDIR}

# Create virtual environment and install dependencies
COPY requirements.txt .
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --no-cache-dir pip==24.0 \
    && pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy application code
COPY app ${WORKDIR}/app
COPY scripts ${WORKDIR}/scripts

# Stage 2: Runtime stage
FROM nvcr.io/nvidia/tritonserver:25.02-py3

LABEL maintainer="fname.lname@domain.com"

# Install runtime dependencies
RUN apt-get update --no-install-recommends \
    && apt-get install --no-install-recommends -y python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set runtime environment
ARG UNAME=triton-server
ENV WORKDIR="/home/$UNAME/src"
ENV VIRTUAL_ENV="/home/$UNAME/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

# Check if user exists, if not create it
RUN id -u $UNAME 2>/dev/null || useradd -m -s /bin/bash $UNAME

# Ensure home directory exists
RUN mkdir -p /home/$UNAME

# Set working directory
WORKDIR ${WORKDIR}

# Copy from builder with proper ownership
COPY --from=builder --chown=$UNAME:$UNAME ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=builder --chown=$UNAME:$UNAME ${WORKDIR}/app ${WORKDIR}/app
COPY --from=builder --chown=$UNAME:$UNAME ${WORKDIR}/scripts ${WORKDIR}/scripts

# Make scripts executable
RUN chmod 777 ${WORKDIR}/scripts/start_servers.sh

# Switch to user
USER $UNAME

CMD ["/bin/bash", "/home/triton-server/src/scripts/start_servers.sh"]