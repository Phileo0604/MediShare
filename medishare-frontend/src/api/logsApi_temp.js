// src/api/logsApi.js
import apiClient from './apiClient';

export const logsApi = {
  // Get all logs
  getAllLogs: async () => {
    const response = await apiClient.get('/api/logs');
    return response.data;
  },

  // Get logs by client ID
  getLogsByClient: async (clientId) => {
    const response = await apiClient.get(`/api/logs/client/${clientId}`);
    return response.data;
  },

  // Get logs by event type
  getLogsByEventType: async (eventType) => {
    const response = await apiClient.get(`/api/logs/event/${eventType}`);
    return response.data;
  },

  // Get logs by dataset type
  getLogsByDatasetType: async (datasetType) => {
    const response = await apiClient.get(`/api/logs/dataset/${datasetType}`);
    return response.data;
  },

  // Create a new log entry
  createLog: async (logData) => {
    const response = await apiClient.post('/api/logs', logData);
    return response.data;
  }
};