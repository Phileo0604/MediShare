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

  // Get a specific model by ID
  getModelById: async (modelId) => {
    const response = await apiClient.get(`/api/models/byId/${modelId}`);
    return response.data;
  },

  // Get active model
  getActiveModel: async (datasetType) => {
    const response = await apiClient.get(`/api/models/${datasetType}/active`);
    return response.data;
  },

  // Download model parameters
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
  },
  
  // Upload model parameters file
  uploadModelFile: async (file, name, description, datasetType) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    formData.append('description', description);
    formData.append('datasetType', datasetType);
    
    const response = await apiClient.post('/api/models/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: progressEvent => {
        // You can handle upload progress here if needed
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        console.log('Upload progress:', percentCompleted);
      }
    });
    
    return response.data;
  }
};