// src/api/clientApi.js
import apiClient from './apiClient';

export const clientApi = {
  // Start a client with a configuration
  startClient: async (datasetType, cycles = 3, serverHost = '127.0.0.1', configId = null) => {
    let url = `/api/client/start?datasetType=${datasetType}&cycles=${cycles}&serverHost=${serverHost}`;
    
    if (configId) {
      url += `&configId=${configId}`;
    }
    
    const response = await apiClient.post(url);
    return response.data;
  },

  // Start a client with pre-trained model parameters
  startClientWithParameters: async (datasetType, modelId, serverHost = '127.0.0.1') => {
    const response = await apiClient.post(
      `/api/client/start-with-parameters?datasetType=${datasetType}&modelId=${modelId}&serverHost=${serverHost}`
    );
    return response.data;
  },

  // Get client status
  getClientStatus: async (clientId) => {
    const response = await apiClient.get(`/api/client/status/${clientId}`);
    return response.data;
  },

  // Stop a client
  stopClient: async (clientId) => {
    const response = await apiClient.post(`/api/client/stop/${clientId}`);
    return response.data;
  },
  
  // Get client history for a dataset type
  getClientHistory: async (datasetType) => {
    const response = await apiClient.get(`/api/client/history/${datasetType}`);
    return response.data;
  }
};