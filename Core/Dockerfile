FROM rasa/rasa:latest

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Switch to root to install dependencies
USER root

# Ensure Protobuf version is compatible with TensorFlow 2.11.1 ru
RUN pip install "protobuf==3.19.6"

# Install project-specific dependencies without overriding previously pinned ones
RUN if [ -f requirements.txt ]; then pip install --no-deps -r requirements.txt; fi

# Force Python implementation of Protobuf to avoid runtime conflicts
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Correct file ownership
RUN chown -R rasa:rasa /app

# Switch back to rasa user
USER rasa

RUN chmod +x /app/start.sh

# Train the model
RUN rasa train

# Expose default Rasa API port
EXPOSE 5005

# Clear any inherited ENTRYPOINT from the base image
ENTRYPOINT ["./start.sh"]