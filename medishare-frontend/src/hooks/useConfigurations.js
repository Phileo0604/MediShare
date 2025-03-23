// src/hooks/useConfigurations.js
import { useState, useEffect } from 'react';
import { configApi } from '../api/configApi';

export const useConfigurations = () => {
  const [configurations, setConfigurations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch all configurations
  const fetchConfigurations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const configs = await configApi.getAllConfigurations();
      setConfigurations(configs);
      setLoading(false);
      return true;
    } catch (err) {
      setError(err.message || 'Failed to load configurations');
      setLoading(false);
      return false;
    }
  };

  // Get a configuration by dataset type
  const getConfigurationByType = async (datasetType) => {
    try {
      return await configApi.getConfigurationByType(datasetType);
    } catch (err) {
      setError(err.message || `Failed to load configuration for ${datasetType}`);
      return null;
    }
  };

  // Create a new configuration
  const createConfiguration = async (configData) => {
    setLoading(true);
    try {
      // Make sure configData is in the correct format
      const formattedData = formatConfigData(configData);
      await configApi.createConfiguration(formattedData);
      await fetchConfigurations();
      return { success: true };
    } catch (err) {
      setError(err.message || 'Failed to create configuration');
      setLoading(false);
      return { success: false, error: err.message };
    }
  };

  // Update a configuration
  const updateConfiguration = async (datasetType, configData) => {
    setLoading(true);
    try {
      // Make sure configData is in the correct format for updating
      const formattedData = formatConfigData(configData);
      
      // Make sure datasetType in the path matches the one in the DTO
      if (formattedData.datasetType !== datasetType) {
        throw new Error('Dataset type mismatch between path and data');
      }
      
      await configApi.updateConfiguration(datasetType, formattedData);
      await fetchConfigurations();
      return { success: true };
    } catch (err) {
      setError(err.message || 'Failed to update configuration');
      setLoading(false);
      return { success: false, error: err.message };
    }
  };

  // Helper function to format config data to match the expected DTO format
  const formatConfigData = (configData) => {
    // Clone to avoid mutating the original
    const formatted = { ...configData };
    
    // Make sure the DTO has the required structure
    if (!formatted.datasetType) {
      throw new Error('Dataset type is required');
    }
    
    if (!formatted.configName) {
      formatted.configName = `${formatted.datasetType} Configuration`;
    }
    
    // Make sure configData is an object, not a string
    if (typeof formatted.configData === 'string') {
      try {
        formatted.configData = JSON.parse(formatted.configData);
      } catch {
        throw new Error('Invalid JSON string in configData');
      }
    }
    
    return formatted;
  };

  // Delete a configuration
  const deleteConfiguration = async (datasetType) => {
    setLoading(true);
    try {
      await configApi.deleteConfiguration(datasetType);
      await fetchConfigurations();
      return { success: true };
    } catch (err) {
      setError(err.message || 'Failed to delete configuration');
      setLoading(false);
      return { success: false, error: err.message };
    }
  };

  // Load configurations on mount
  useEffect(() => {
    fetchConfigurations();
  }, []);

  return {
    configurations,
    loading,
    error,
    fetchConfigurations,
    getConfigurationByType,
    createConfiguration,
    updateConfiguration,
    deleteConfiguration
  };
};