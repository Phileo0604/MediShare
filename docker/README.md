# MediShare Federated Learning System - Docker Setup

This repository contains Docker configurations to run the complete MediShare federated learning system, including:

- PostgreSQL database
- Spring Boot API
- Federated Learning Server
- PgAdmin for database management

## Project Structure

```
medishare/
├── docker-compose.yml       # Main configuration file
├── init.sql                 # Database initialization script
├── api/                     # Spring Boot API project
│   ├── Dockerfile
│   └── ... (Spring Boot files)
├── federated-server/        # Federated Learning server
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── server_config.json   # Server configuration
│   └── ... (Python files)
```

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed
- Java 17 for local development (optional)
- Python 3.10 for local development (optional)

### Starting the System

1. Clone this repository
2. Navigate to the project directory
3. Run the following command:

```bash
docker-compose up -d
```

This will start all components:
- PostgreSQL on port 5432
- PgAdmin on port 5050
- Spring Boot API on port 8085
- Federated Learning Server on port 8080

### Accessing Components

- **Spring Boot API**: http://localhost:8085
- **PgAdmin**: http://localhost:5050
  - Email: admin@medishare.com
  - Password: admin

### Shared Storage

The system uses Docker volumes to share data between containers:
- `model_storage`: Shared between API and Federated Server for ML models
- `config_storage`: Configurations created by API and used by Federated Server
- `postgres_data`: Database persistence
- `api_data`: API-specific data storage

## Development

### Working with Spring Boot API

To rebuild the API after changes:

```bash
docker-compose build api
docker-compose up -d api
```

### Working with Federated Learning Server

To rebuild the server after changes:

```bash
docker-compose build federated-server
docker-compose up -d federated-server
```

### Viewing Logs

```bash
# View logs for all services
docker-compose logs

# View logs for a specific service
docker-compose logs api
docker-compose logs federated-server
```

## Database Schema

The PostgreSQL database includes the following tables:

1. **users** - User accounts
2. **configurations** - Model configurations
3. **models** - Global ML models
4. **model_versions** - Version history for models
5. **clients** - Federated learning clients
6. **federation_logs** - System activity logs

## Stopping the System

```bash
docker-compose down
```

To remove volumes (this will delete all data):

```bash
docker-compose down -v
```