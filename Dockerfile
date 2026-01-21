FROM python:3.10-slim

WORKDIR /app

# Copy all files
COPY . /app/

# Install dependencies
RUN pip install fastapi==0.104.1 uvicorn==0.24.0 \
    torch torchvision torchaudio \
    pytorch-tabnet \
    mlflow \
    scikit-learn \
    pandas numpy \
    imbalanced-learn \
    scipy \
    pydantic

EXPOSE 8000

CMD ["uvicorn", "src.api_service:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
