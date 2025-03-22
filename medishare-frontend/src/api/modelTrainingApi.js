// src/api/modelTrainingApi.js
import apiClient from './apiClient';

export const modelTrainingApi = {
  // Train a model with uploaded dataset
  trainModel: async (trainingData) => {
    try {
      // Create FormData object for file upload
      const formData = new FormData();
      formData.append('file', trainingData.datasetFile);
      formData.append('datasetType', trainingData.datasetType);
      formData.append('modelName', trainingData.modelName);
      formData.append('description', trainingData.description);
      formData.append('epochs', trainingData.epochs);
      formData.append('batchSize', trainingData.batchSize);
      formData.append('learningRate', trainingData.learningRate);
      
      // Upload the file and start training
      const response = await apiClient.post('/api/training/start', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        // Track upload progress if needed
        onUploadProgress: progressEvent => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log('Training dataset upload progress:', percentCompleted);
        }
      });
      
      return response.data;
    } catch (error) {
      console.error('Error in model training:', error);
      throw error;
    }
  },
  
  // Check training status
  getTrainingStatus: async (trainingId) => {
    const response = await apiClient.get(`/api/training/status/${trainingId}`);
    return response.data;
  },
  
  // Cancel ongoing training
  cancelTraining: async (trainingId) => {
    const response = await apiClient.post(`/api/training/cancel/${trainingId}`);
    return response.data;
  }
};