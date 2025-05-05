# 1. Use specific Python 3.11.11 image
FROM python:3.11.11-slim

# 2. Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# 3. Set working directory inside container
WORKDIR /app

# 4. Install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy Python dependencies and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 6. Copy all project files to /app inside container
COPY . .

# 7. Expose Dash app port
EXPOSE 5000

# 8. Run the Dash app by default
CMD ["python", "application.py"]