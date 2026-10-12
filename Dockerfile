# Forecast API image for AWS Lambda.
# Built by .github/workflows/deploy_api.yml, which first writes:
#   build/requirements-serve.txt   uv export --only-group serve (pinned from uv.lock)
#   build/model/                   scripts/fetch_model.py (one registered model version)
FROM public.ecr.aws/lambda/python:3.14

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /bin/uv

COPY build/requirements-serve.txt /tmp/requirements-serve.txt
# tzdata: America/Chicago must not depend on the base image's zoneinfo.
RUN uv pip install --system --no-cache -r /tmp/requirements-serve.txt tzdata

# Mirror the repo layout: shef.config finds features.yaml at parents[2] of itself.
COPY features.yaml ${LAMBDA_TASK_ROOT}/features.yaml
COPY src/shef ${LAMBDA_TASK_ROOT}/src/shef
COPY build/model /opt/model

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/src \
    MODEL_DIR=/opt/model

CMD ["shef.serving.app.handler"]
