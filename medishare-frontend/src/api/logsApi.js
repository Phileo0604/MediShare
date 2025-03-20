// src/api/logsApi.js
import apiClient from './apiClient';

export const logsApi = {
  // Get logs with specified number of lines
  getLogs: async (lines = 100) => {
    const response = await apiClient.get(`/api/logs?lines=${lines}`);
    return response.data;
  }
};