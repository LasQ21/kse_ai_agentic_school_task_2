# Образ для воркера живих даних (і, далі, API). Секрети — лише через --env-file, у образ не потрапляють.
# НЕ перевірено збиранням: на машині розробки немає Docker.
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY scripts/ scripts/
# data/ і models/ монтуються томами (див. docker-compose.yml)
ENV PYTHONUNBUFFERED=1
CMD ["python", "scripts/live_worker.py"]
