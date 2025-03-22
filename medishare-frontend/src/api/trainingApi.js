// src/api/trainingApi.js
import apiClient from './apiClient';

export const trainingApi = {
  // Start a new model training job
  startTraining: async (formData) => {
    // Create FormData object for file upload
    const data = new FormData();
    data.append('file', formData.datasetFile);
    data.append('datasetType', formData.datasetType);
    data.append('modelName', formData.modelName);
    data.append('description', formData.description);
    data.append('epochs', formData.epochs);
    data.append('batchSize', formData.batchSize);
    data.append('learningRate', formData.learningRate);
    
    const response = await apiClient.post('/api/training/start', data, {
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
  },

  // Get status of a training job
  getTrainingStatus: async (jobId) => {
    const response = await apiClient.get(`/api/training/status/${jobId}`);
    return response.data;
  },

  // Cancel a training job
  cancelTraining: async (jobId) => {
    const response = await apiClient.post(`/api/training/cancel/${jobId}`);
    return response.data;
  },

  // Get all training jobs
  getAllTrainingJobs: async () => {
    const response = await apiClient.get('/api/training/jobs');
    return response.data;
  },
  
  // Get active training jobs
  getActiveTrainingJobs: async () => {
    const response = await apiClient.get('/api/training/active');
    return response.data;
  }
};