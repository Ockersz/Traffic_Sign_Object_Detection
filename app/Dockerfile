FROM python:3.8

WORKDIR /app

COPY model.py requirements.txt /app/

RUN pip install --upgrade pip setuptools

RUN pip install --no-cache-dir -r requirements.txt

# Specify the command to run your model when the container starts
CMD ["python", "model.py"]
