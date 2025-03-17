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
      await configApi.createConfiguration(configData);
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
      await configApi.updateConfiguration(datasetType, configData);
      await fetchConfigurations();
      return { success: true };
    } catch (err) {
      setError(err.message || 'Failed to update configuration');
      setLoading(false);
      return { success: false, error: err.message };
    }
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