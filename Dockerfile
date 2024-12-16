# Use NVIDIA's PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "-m", "src.simulations.prototypes.self_organizing_sim"]