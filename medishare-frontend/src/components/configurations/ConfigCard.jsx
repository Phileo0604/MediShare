// src/components/configurations/ConfigCard.jsx
import React, { useState, useEffect } from 'react';
import { modelApi } from '../../api/modelApi';

const ConfigCard = ({ config, onEdit, onDelete }) => {
  const [modelDetails, setModelDetails] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Fetch model details when component mounts
  useEffect(() => {
    const fetchModelDetails = async () => {
      if (config?.configData?.model?.modelId) {
        setLoading(true);
        try {
          // This would need to be implemented in your modelApi
          const modelInfo = await modelApi.getModelById(config.configData.model.modelId);
          setModelDetails(modelInfo);
        } catch (error) {
          console.error("Error fetching model details:", error);
        } finally {
          setLoading(false);
        }
      }
    };
    
    fetchModelDetails();
  }, [config]);

  return (
    <div className="config-card">
      <div className="card-header">
        <h3>{config.configName}</h3>
        <span className="dataset-type">{config.datasetType}</span>
      </div>
      
      <div className="card-body">
        <div className="config-section">
          <h4>Model</h4>
          {loading ? (
            <p>Loading model details...</p>
          ) : modelDetails ? (
            <>
              <p><strong>Name:</strong> {modelDetails.name}</p>
              <p><strong>Created:</strong> {formatDate(modelDetails.createdAt)}</p>
              {modelDetails.description && (
                <p><strong>Description:</strong> {modelDetails.description}</p>
              )}
            </>
          ) : (
            <p>Model ID: {config.configData.model.modelId || 'Not specified'}</p>
          )}
        </div>
        
        <div className="config-section">
          <h4>Server</h4>
          <p><strong>Host:</strong> {config.configData.server.host}</p>
          <p><strong>Port:</strong> {config.configData.server.port}</p>
          <p><strong>Update Threshold:</strong> {config.configData.server.update_threshold}</p>
          <p><strong>Contribution Weight:</strong> {config.configData.server.contribution_weight}</p>
        </div>
        
        <div className="config-section">
          <h4>Client</h4>
          <p><strong>Cycles:</strong> {config.configData.client.cycles}</p>
          <p><strong>Retry Interval:</strong> {config.configData.client.retry_interval} seconds</p>
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

// Helper function to format date (implement or import from your utils)
const formatDate = (dateString) => {
  if (!dateString) return '';
  return new Date(dateString).toLocaleString();
};