// src/hooks/useModels.js
import { useState, useEffect } from 'react';
import { modelApi } from '../api/modelApi';

export const useModels = (datasetType) => {
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch models for the dataset type
  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Get all models for the dataset type
      const modelsData = await modelApi.getModelsByType(datasetType);
      setModels(modelsData);
      
      // Try to get active model
      try {
        const activeModelData = await modelApi.getActiveModel(datasetType);
        setActiveModel(activeModelData);
      } catch {
        // No active model or error fetching it
        setActiveModel(null);
      }
      
      setLoading(false);
      return true;
    } catch (err) {
      setError(err.message || 'Failed to load models');
      setLoading(false);
      return false;
    }
  };

  // Download a model
  const downloadModel = async (modelId) => {
    try {
      const blob = await modelApi.downloadModel(datasetType);
      
      // Create a download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `model_${datasetType}_${modelId}.pkl`;
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      link.remove();
      
      return { success: true };
    } catch (err) {
      setError(err.message || 'Failed to download model');
      return { success: false, error: err.message };
    }
  };

  // Register a new model
  const registerModel = async (modelData, filePath) => {
    try {
      await modelApi.registerModel(modelData, filePath);
      await fetchModels();
      return { success: true };
    } catch (err) {
      return { success: false, error: err.message || 'Failed to register model' };
    }
  };

  // Load models when dataset type changes
  useEffect(() => {
    if (datasetType) {
      fetchModels();
    }
  }, [datasetType]);

  return {
    models,
    activeModel,
    loading,
    error,
    fetchModels,
    downloadModel,
    registerModel
  };
};