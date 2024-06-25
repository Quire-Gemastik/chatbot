FROM nvidia/cuda:11.2.2-cudnn8-devel  
# Base image with CUDA and development tools

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools  
# Install system dependencies

RUN pip3 install --no-cache-dir -U install setuptools pip  
# Update pip

# Install Python dependencies from requirements.txt
COPY requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app  
# Set working directory

COPY . .  
# Copy current directory contents to working directory

EXPOSE 8000  
# Expose port for FastAPI application

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]  
# Start FastAPI application
