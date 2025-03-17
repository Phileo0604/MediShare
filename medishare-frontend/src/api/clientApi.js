// src/api/clientApi.js
import apiClient from './apiClient';

export const clientApi = {
  // Start a client
  startClient: async (datasetType, cycles = 3, serverHost = '127.0.0.1') => {
    const response = await apiClient.post(
      `/api/client/start?datasetType=${datasetType}&cycles=${cycles}&serverHost=${serverHost}`
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
  }
};