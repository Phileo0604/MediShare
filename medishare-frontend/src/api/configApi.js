// src/api/configApi.js
import apiClient from './apiClient';

export const configApi = {
  // Get all configurations
  getAllConfigurations: async () => {
    const response = await apiClient.get('/api/config/all');
    return response.data;
  },

  // Get configuration by dataset type
  getConfigurationByType: async (datasetType) => {
    const response = await apiClient.get(`/api/config/${datasetType}`);
    return response.data;
  },

  // Create a new configuration
  createConfiguration: async (configData) => {
    const response = await apiClient.post('/api/config/create', configData);
    return response.data;
  },

  // Update an existing configuration
  updateConfiguration: async (datasetType, configData) => {
    const response = await apiClient.put(`/api/config/${datasetType}`, configData);
    return response.data;
  },

  // Delete a configuration
  deleteConfiguration: async (datasetType) => {
    const response = await apiClient.delete(`/api/config/${datasetType}`);
    return response.data;
  },

  // Delete all configurations
  deleteAllConfigurations: async () => {
    const response = await apiClient.delete('/api/config/all');
    return response.data;
  }
};





