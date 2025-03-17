// src/components/models/ModelCard.jsx
import React from 'react';

const ModelCard = ({ model, isActive, onDownload }) => {
  return (
    <div className={`model-card card ${isActive ? 'active-model' : ''}`}>
      <div className="card-header">
        <h3>{model.name}</h3>
        {isActive && <span className="status-badge active">Active</span>}
      </div>
      
      <div className="card-body">
        <p><strong>Created:</strong> {new Date(model.createdAt).toLocaleString()}</p>
        <p><strong>Dataset Type:</strong> {model.datasetType}</p>
        <p><strong>Description:</strong> {model.description}</p>
        
        {model.performance && (
          <div className="model-performance">
            <h4>Performance Metrics</h4>
            <p><strong>Accuracy:</strong> {model.performance.accuracy}</p>
            <p><strong>Precision:</strong> {model.performance.precision}</p>
            <p><strong>Recall:</strong> {model.performance.recall}</p>
          </div>
        )}
      </div>
      
      <div className="card-footer">
        <button 
          className="btn btn-primary"
          onClick={() => onDownload(model.id)}
        >
          Download Model
        </button>
      </div>
    </div>
  );
};

export default ModelCard;