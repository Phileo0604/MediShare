// src/pages/ModelManagement.jsx
import React, { useState, useEffect } from 'react';
import { modelApi } from '../api/modelApi';
import ModelList from '../components/models/ModelList';
import ModelUpload from '../components/models/ModelUpload';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { useGlobalContext } from '../context/GlobalContext';

const ModelManagement = () => {
  const { selectedDatasetType, setSelectedDatasetType } = useGlobalContext();
  
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch models for the selected dataset type
  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const modelsData = await modelApi.getModelsByType(selectedDatasetType);
      setModels(modelsData);
      
      // Fetch active model
      try {
        const activeModelData = await modelApi.getActiveModel(selectedDatasetType);
        setActiveModel(activeModelData);
      } catch (err) {
        // No active model or error fetching active model
        setActiveModel(null);
      }
    } catch (err) {
      setError(err.message || 'Failed to load models');
    } finally {
      setLoading(false);
    }
  };
  
  // Download a model
  const handleDownloadModel = async (modelId) => {
    try {
      const blob = await modelApi.downloadModel(selectedDatasetType);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `model_${selectedDatasetType}_${modelId}.pkl`;
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      link.remove();
    } catch (err) {
      setError(err.message || 'Failed to download model');
    }
  };
  
  // Register a new model
  const handleRegisterModel = async (modelData, filePath) => {
    try {
      await modelApi.registerModel(modelData, filePath);
      await fetchModels();
      return { success: true };
    } catch (err) {
      return { 
        success: false, 
        error: err.message || 'Failed to register model' 
      };
    }
  };
  
  // Handle dataset type change
  const handleDatasetTypeChange = (type) => {
    setSelectedDatasetType(type);
  };
  
  // Fetch models when dataset type changes
  useEffect(() => {
    fetchModels();
  }, [selectedDatasetType]);
  
  if (loading && models.length === 0) {
    return <LoadingSpinner />;
  }
  
  return (
    <div className="model-management">
      <h1>Model Management</h1>
      
      {error && <ErrorMessage message={error} onRetry={fetchModels} />}
      
      <div className="dataset-selector">
        <label htmlFor="datasetType">Dataset Type:</label>
        <select 
          id="datasetType"
          value={selectedDatasetType}
          onChange={(e) => handleDatasetTypeChange(e.target.value)}
        >
          <option value="breast_cancer">Breast Cancer</option>
          <option value="parkinsons">Parkinson's</option>
          <option value="reinopath">Reinopath</option>
        </select>
      </div>
      
      <div className="model-sections">
        <div className="model-list-section">
          <h2>Available Models</h2>
          {models.length > 0 ? (
            <ModelList 
              models={models}
              activeModel={activeModel}
              onDownload={handleDownloadModel}
            />
          ) : (
            <p className="no-models-message">No models available for this dataset type.</p>
          )}
        </div>
        
        <div className="model-upload-section">
          <ModelUpload 
            datasetType={selectedDatasetType}
            onRegister={handleRegisterModel}
          />
        </div>
      </div>
    </div>
  );
};

export default ModelManagement;