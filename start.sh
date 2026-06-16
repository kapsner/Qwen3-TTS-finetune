#!/bin/bash

./train.sh \
    --audio_dir /opt/raw-dataset \
    --ref_audio /opt/raw-dataset/ref.wav \
    --speaker_name my_voice \
    --output_dir /opt/output \
    --batch_size 2 \
    --lr 1e-8 \
    --epochs 5 \
    --whisper_model large-v3 \
    --language de
