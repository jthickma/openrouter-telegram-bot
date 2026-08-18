FROM python:3.12-alpine

ENV PYTHONFAULTHANDLER=1 \
     PYTHONUNBUFFERED=1 \
     PYTHONDONTWRITEBYTECODE=1 \
     PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt --no-cache-dir

RUN addgroup -S bot && adduser -S -G bot bot \
    && mkdir -p /app/usage_logs /app/user_data \
    && chown -R bot:bot /app

USER bot

CMD ["python", "bot/main.py"]
