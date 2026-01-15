#!/bin/bash
set -e

# 2) Start main Rasa server (foreground)
rasa run \
  --enable-api \
  --cors "*" \
  --debug \
  --port 5005 \
  --interface 0.0.0.0
