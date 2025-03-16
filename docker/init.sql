-- Create database if it doesn't exist
-- Note: This is handled by Docker environment variables in docker-compose.yml

-- Connect to database
\c medishare;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    institution VARCHAR(100),
    role VARCHAR(20) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create configurations table
CREATE TABLE IF NOT EXISTS configurations (
    id SERIAL PRIMARY KEY,
    dataset_type VARCHAR(50) NOT NULL,
    config_name VARCHAR(100) NOT NULL,
    config_json JSONB NOT NULL,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE (dataset_type, config_name)
);

-- Create models table
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    dataset_type VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    file_format VARCHAR(20) NOT NULL,
    parameters_count INTEGER,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create model_versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    version_number INTEGER NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    created_by INTEGER REFERENCES users(id),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Create clients table
CREATE TABLE IF NOT EXISTS clients (
    id SERIAL PRIMARY KEY,
    client_name VARCHAR(100) NOT NULL,
    client_id VARCHAR(100) NOT NULL UNIQUE,
    dataset_type VARCHAR(50) NOT NULL,
    created_by INTEGER REFERENCES users(id),
    last_connected TIMESTAMP,
    contribution_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create federation_logs table
CREATE TABLE IF NOT EXISTS federation_logs (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    dataset_type VARCHAR(50) NOT NULL,
    client_id VARCHAR(100),
    user_id INTEGER REFERENCES users(id),
    details JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create global models table
CREATE TABLE IF NOT EXISTS global_models (
    id SERIAL PRIMARY KEY,
    dataset_type VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    parameters_json JSONB,
    creation_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    update_count INTEGER DEFAULT 0,
    client_contributions INTEGER DEFAULT 0,
    metrics JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT unique_active_model_per_dataset UNIQUE (dataset_type, is_active)
);

-- Create indexes for commonly queried fields
CREATE INDEX idx_config_dataset_type ON configurations(dataset_type);
CREATE INDEX idx_model_dataset_type ON models(dataset_type);
CREATE INDEX idx_client_dataset_type ON clients(dataset_type);
CREATE INDEX idx_federation_logs_event_type ON federation_logs(event_type);
CREATE INDEX idx_federation_logs_dataset_type ON federation_logs(dataset_type);
CREATE INDEX idx_global_models_dataset ON global_models(dataset_type);

-- Create an admin user (password: admin123)
INSERT INTO users (username, password_hash, email, role, created_at)
VALUES ('admin', '$2a$10$xVCU3cqPCMyYqBBfvzLdPu2aQJvX1HvA9qtxQfcQP6yfzXbKqPCUi', 'admin@medishare.com', 'admin', CURRENT_TIMESTAMP)
ON CONFLICT (username) DO NOTHING;
