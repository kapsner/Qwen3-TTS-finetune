#!/bin/bash

# use speaker "aiden" for compatibility with voicebox
./train.sh \
    --audio_dir /opt/raw-dataset \
    --ref_audio /opt/raw-dataset/ref.wav \
    --speaker_name aiden \
    --output_dir /opt/output \
    --batch_size 2 \
    --lr 5e-10 \
    --epochs 5 \
    --whisper_model large-v3 \
    --language de

# lr:
# - 5e-9 already very good
# - 2.5e-9 also very good
# - 1e-9 also very good
# - 5e-10 also very good (current favourit)
# - 1e-10 ?
