FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8502

ENTRYPOINT ["streamlit", "run", "src/main.py"]