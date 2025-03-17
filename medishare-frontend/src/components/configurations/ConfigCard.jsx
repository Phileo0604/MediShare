// src/components/configurations/ConfigCard.jsx
import React from 'react';

const ConfigCard = ({ config, onEdit, onDelete }) => {
  return (
    <div className="config-card">
      <div className="card-header">
        <h3>{config.configName}</h3>
        <span className="dataset-type">{config.datasetType}</span>
      </div>
      
      <div className="card-body">
        <div className="config-section">
          <h4>Dataset</h4>
          <p><strong>Path:</strong> {config.configData.dataset.path}</p>
          <p><strong>Target Column:</strong> {config.configData.dataset.target_column}</p>
        </div>
        
        <div className="config-section">
          <h4>Training</h4>
          <p><strong>Epochs:</strong> {config.configData.training.epochs}</p>
          <p><strong>Batch Size:</strong> {config.configData.training.batch_size}</p>
          <p><strong>Learning Rate:</strong> {config.configData.training.learning_rate}</p>
        </div>
        
        <div className="config-section">
          <h4>Model</h4>
          <p><strong>Hidden Layers:</strong> {config.configData.model.hidden_layers.join(', ')}</p>
          <p><strong>Parameters File:</strong> {config.configData.model.parameters_file}</p>
        </div>
        
        <div className="config-section">
          <h4>Server</h4>
          <p><strong>Host:</strong> {config.configData.server.host}</p>
          <p><strong>Port:</strong> {config.configData.server.port}</p>
          <p><strong>Update Threshold:</strong> {config.configData.server.update_threshold}</p>
        </div>
      </div>
      
      <div className="card-footer">
        <button 
          className="btn btn-secondary"
          onClick={() => onEdit(config)}
        >
          Edit
        </button>
        <button 
          className="btn btn-danger"
          onClick={() => onDelete(config.datasetType)}
        >
          Delete
        </button>
      </div>
    </div>
  );
};

export default ConfigCard;