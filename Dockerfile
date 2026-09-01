# cuDNN 9 is required by CTranslate2, the runtime under faster-whisper.
# The -runtime image carries it; -base does not and fails at model load.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY transcribe.py .

# Bake the weights into the image so the container never needs the network at
# run time. Drop this line to mount a pre-populated /models volume instead --
# the right call when the runtime perimeter has no egress at all.
ARG MODEL=large-v3
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('${MODEL}', device='cpu')"

ENTRYPOINT ["python3", "transcribe.py"]
