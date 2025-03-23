// src/api/configApi.js
import apiClient from './apiClient';

export const configApi = {
  // Get all configurations
  getAllConfigurations: async () => {
    const response = await apiClient.get('/api/config/all');
    // Make sure configData is parsed JSON, not a string
    return response.data.map(config => ({
      ...config,
      configData: typeof config.configJson === 'string' 
        ? JSON.parse(config.configJson)
        : config.configJson
    }));
  },

  // Get configuration by dataset type
  getConfigurationByType: async (datasetType) => {
    const response = await apiClient.get(`/api/config/${datasetType}`);
    return response.data;
  },

  // Create a new configuration
  createConfiguration: async (configData) => {
    // Create a DTO matching what the backend expects
    const configDTO = {
      datasetType: configData.datasetType,
      configName: configData.configName,
      configData: configData.configData
    };
    
    const response = await apiClient.post('/api/config/create', configDTO);
    return response.data;
  },

  // Update an existing configuration
  updateConfiguration: async (datasetType, configData) => {
    // Create a DTO matching what the backend expects
    const configDTO = {
      datasetType: configData.datasetType,
      configName: configData.configName,
      configData: configData.configData
    };
    
    const response = await apiClient.put(`/api/config/${datasetType}`, configDTO);
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
  },
};