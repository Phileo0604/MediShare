// src/api/serverApi.js
import apiClient from './apiClient';

export const serverApi = {
  // Start the server - modified to make datasetType optional
  startServer: async (datasetType = null) => {
    const url = datasetType 
      ? `/api/server/start?datasetType=${datasetType}`
      : '/api/server/start';
    const response = await apiClient.post(url);
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