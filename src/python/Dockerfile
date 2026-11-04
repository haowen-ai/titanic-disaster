FROM python:3.12.16-slim

WORKDIR /app

# Install packages first so code changes do not redo this step.
COPY src/python/requirements.txt src/python/requirements.txt
RUN pip install --no-cache-dir -r src/python/requirements.txt

COPY src/python/ src/python/
COPY src/data/ src/data/

CMD ["python", "src/python/titanic.py"]
