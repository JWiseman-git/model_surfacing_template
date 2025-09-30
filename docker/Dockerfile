FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

#
# Build poetry
RUN apt-get update
RUN apt-get install --no-install-recommends -y build-essential

RUN pip install -U --no-cache-dir pip
ENV POETRY_VERSION=1.8.2
RUN pip install poetry==${POETRY_VERSION}

#
# Install project dependencies
WORKDIR /project
COPY ./* ./
RUN make install

#
# This barebones does not currently implement the application
# set up and launcher

CMD ["sleep", "120"]
