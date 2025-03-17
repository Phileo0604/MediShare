// src/api/serverApi.js
import apiClient from './apiClient';

export const serverApi = {
  // Start the server
  startServer: async (datasetType) => {
    const response = await apiClient.post(`/api/server/start?datasetType=${datasetType}`);
    return response.data;
  },

  // Stop the server
  stopServer: async () => {
    const response = await apiClient.post('/api/server/stop');
    return response.data;
  },

  // Get server status
  getServerStatus: async () => {
    const response = await apiClient.get('/api/server/status');
    return response.data;
  }
};