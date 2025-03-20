# Build stage
FROM eclipse-temurin:17-jdk-alpine as build
WORKDIR /workspace/app

COPY mvnw .
COPY .mvn .mvn
COPY pom.xml .
RUN ./mvnw dependency:go-offline -B
COPY src src
RUN ./mvnw package -DskipTests
RUN mkdir -p target/dependency && (cd target/dependency; jar -xf ../*.jar)

# Runtime stage - Using OpenJDK slim instead
FROM openjdk:17-slim
VOLUME /tmp

# Install Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python command
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install required Python packages
RUN pip3 install --no-cache-dir numpy pandas scikit-learn xgboost torch matplotlib seaborn plotly openpyxl python-dateutil scipy joblib pillow

# Try installing torch with CPU-only support to reduce size
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application files
ARG DEPENDENCY=/workspace/app/target/dependency
COPY --from=build ${DEPENDENCY}/BOOT-INF/lib /app/lib
COPY --from=build ${DEPENDENCY}/META-INF /app/META-INF
COPY --from=build ${DEPENDENCY}/BOOT-INF/classes /app

# Create directories for configs and models
RUN mkdir -p /app/configs /app/models

# Instead of copying, create a placeholder Python script
RUN echo 'print("MediShare API Server")' > /app/main.py

# Set environment variables
ENV APP_CONFIG_STORAGE_LOCATION=/app/configs
ENV APP_MODEL_STORAGE_LOCATION=/app/models
ENV APP_PYTHON_SCRIPT_PATH=/app/main.py
ENV APP_PYTHON_EXECUTABLE=/usr/bin/python3

ENTRYPOINT ["java","-cp","app:app/lib/*","com.medishare.api.MediShareApiApplication"]