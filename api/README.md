# MediShare API

This is the REST API backend for the MediShare federated learning system, providing interfaces to manage federated learning configurations, models, and operations.

## Overview

The MediShare API acts as a bridge between the web interface and the Python-based federated learning system. It manages:

- Model configurations for different medical datasets
- Global model metadata and storage
- Federated learning server operations
- Client training operations

## Prerequisites

- Java 17
- Maven
- PostgreSQL
- Python environment with MediShare federated learning code

## Setup

1. Clone the repository
2. Configure PostgreSQL database
3. Update `application.properties` with your database credentials
4. Build the project:

```bash
mvn clean package
```

5. Run the application:

```bash
java -jar target/api-0.0.1-SNAPSHOT.jar
```

## API Endpoints

### Configuration Management

- `POST /api/config/create` - Create a new configuration
- `GET /api/config/all` - List all configurations
- `GET /api/config/{datasetType}` - Get configuration for a specific dataset
- `PUT /api/config/{datasetType}` - Update configuration

### Model Management

- `GET /api/models` - List all models
- `GET /api/models/{datasetType}` - List models for a specific dataset
- `GET /api/models/{datasetType}/active` - Get the active model for a dataset
- `GET /api/models/{datasetType}/download` - Download a model
- `POST /api/models/register` - Register a new model

### Server Control

- `POST /api/server/start` - Start the federated learning server
- `POST /api/server/stop` - Stop the server
- `GET /api/server/status` - Check server status

### Client Operations

- `POST /api/client/start` - Start a federated learning client
- `GET /api/client/status/{clientId}` - Check client status
- `POST /api/client/stop/{clientId}` - Stop a client

## Integration with MediShare Federated Learning

This API interfaces with the MediShare federated learning system by:

1. Managing configuration files that the Python code reads
2. Launching Python processes for server and client operations
3. Tracking the status of running processes
4. Storing metadata about models in the database

## Database Schema

The application uses the following database tables:

- `configurations` - Stores configuration data for different dataset types
- `models` - Stores metadata about trained models

## File Storage

The application stores files in two main directories (configurable in application.properties):

- `app.config-storage.location` - For configuration files
- `app.model-storage.location` - For model files

## Development

To run the application in development mode:

```bash
mvn spring-boot:run
```

API documentation is available at:
http://localhost:8085/swagger-ui.html when the application is running.