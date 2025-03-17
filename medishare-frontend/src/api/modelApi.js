// src/api/modelApi.js
import apiClient from './apiClient';

export const modelApi = {
  // Get all models
  getAllModels: async () => {
    const response = await apiClient.get('/api/models');
    return response.data;
  },

  // Get models by dataset type
  getModelsByType: async (datasetType) => {
    const response = await apiClient.get(`/api/models/${datasetType}`);
    return response.data;
  },

  // Get active model
  getActiveModel: async (datasetType) => {
    const response = await apiClient.get(`/api/models/${datasetType}/active`);
    return response.data;
  },

  // Download model
  downloadModel: async (datasetType) => {
    const response = await apiClient.get(`/api/models/${datasetType}/download`, {
      responseType: 'blob'
    });
    return response.data;
  },

  // Register model
  registerModel: async (modelData, filePath) => {
    const response = await apiClient.post(`/api/models/register?filePath=${filePath}`, modelData);
    return response.data;
  }
};