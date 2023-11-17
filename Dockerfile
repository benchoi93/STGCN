FROM tensorflow/tensorflow:2.12.0rc0-gpu-jupyter

COPY . /app
WORKDIR /app

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt