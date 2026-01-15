FROM rasa/rasa:latest

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Switch to root to install dependencies
USER root

# Ensure Protobuf version is compatible with TensorFlow 2.11.1
RUN pip install "protobuf==3.19.6"

# Install project-specific dependencies
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm

# Force Python implementation of Protobuf to avoid runtime conflicts
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Ensure Python outputs are unbuffered so logs appear in real-time
ENV PYTHONUNBUFFERED=1

# Correct file ownership
RUN chown -R rasa:rasa /app

# Switch back to rasa user
USER rasa

# Expose action server port
EXPOSE 5055

# inherits ENTRYPOINT ["rasa"] from the base image
CMD ["run", "actions", "--port", "5055"]

