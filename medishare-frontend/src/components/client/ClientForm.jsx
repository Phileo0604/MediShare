// src/components/client/ClientForm.jsx
import React, { useState, useEffect } from 'react';
import { clientApi } from '../../api/clientApi';
import { configApi } from '../../api/configApi';
import { modelApi } from '../../api/modelApi';
import { useGlobalContext } from '../../context/GlobalContext';
import LoadingSpinner from '../common/LoadingSpinner';

const ClientForm = ({ onClientStarted }) => {
  const { selectedDatasetType, setSelectedDatasetType } = useGlobalContext();
  
  const [formData, setFormData] = useState({
    datasetType: selectedDatasetType,
    configId: '',
    modelId: '',
    cycles: 3,
    serverHost: '127.0.0.1',
    mode: 'config' // 'config' or 'model'
  });
  
  const [configurations, setConfigurations] = useState([]);
  const [models, setModels] = useState([]);
  const [fetchingConfigs, setFetchingConfigs] = useState(true);
  const [fetchingModels, setFetchingModels] = useState(true);
  const [configError, setConfigError] = useState(null);
  const [modelError, setModelError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Fetch available configurations for the selected dataset type
  useEffect(() => {
    const fetchConfigurations = async () => {
      setFetchingConfigs(true);
      setConfigError(null);
      
      try {
        // Get all configurations
        const response = await configApi.getAllConfigurations();
        
        // Filter configurations for the selected dataset type
        const filteredConfigs = response.filter(
          config => config.datasetType === formData.datasetType
        );
        
        setConfigurations(filteredConfigs);
        
        // If there's at least one configuration, select it by default
        if (filteredConfigs.length > 0 && formData.mode === 'config') {
          setFormData(prev => ({
            ...prev,
            configId: filteredConfigs[0].id.toString()
          }));
        } else if (formData.mode === 'config') {
          // Clear the configId if no configurations are available
          setFormData(prev => ({
            ...prev,
            configId: ''
          }));
        }
      } catch (err) {
        setConfigError(err.message || 'Failed to load configurations');
        console.error('Error fetching configurations:', err);
      } finally {
        setFetchingConfigs(false);
      }
    };
    
    fetchConfigurations();
  }, [formData.datasetType]);
  
  // Fetch available models for the selected dataset type
  useEffect(() => {
    const fetchModels = async () => {
      setFetchingModels(true);
      setModelError(null);
      
      try {
        // Get models for the selected dataset type
        const response = await modelApi.getModelsByType(formData.datasetType);
        
        setModels(response);
        
        // If there's at least one model, select it by default if in model mode
        if (response.length > 0 && formData.mode === 'model') {
          setFormData(prev => ({
            ...prev,
            modelId: response[0].id.toString()
          }));
        } else if (formData.mode === 'model') {
          // Clear the modelId if no models are available
          setFormData(prev => ({
            ...prev,
            modelId: ''
          }));
        }
      } catch (err) {
        setModelError(err.message || 'Failed to load models');
        console.error('Error fetching models:', err);
      } finally {
        setFetchingModels(false);
      }
    };
    
    fetchModels();
  }, [formData.datasetType]);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (name === 'datasetType') {
      setSelectedDatasetType(value);
    }
  };
  
  const handleModeChange = (mode) => {
    setFormData(prev => ({ ...prev, mode }));
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      let result;
      
      if (formData.mode === 'model' && formData.modelId) {
        // Start client with model parameters
        result = await clientApi.startClientWithParameters(
          formData.datasetType,
          formData.modelId,
          formData.serverHost
        );
      } else {
        // Start client with configuration
        result = await clientApi.startClient(
          formData.datasetType, 
          formData.cycles, 
          formData.serverHost,
          formData.configId // Pass the selected configuration ID
        );
      }
      
      if (result && result.clientId) {
        if (onClientStarted) {
          onClientStarted(result);
        }
        // Reset form but keep the dataset type and IDs
        setFormData(prev => ({
          ...prev,
          cycles: 3,
          serverHost: '127.0.0.1'
        }));
      } else {
        setError('Failed to start client. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'Failed to start client');
    } finally {
      setLoading(false);
    }
  };
  
  // Check if we can show the start button
  const canStart = (
    formData.mode === 'config' 
    ? configurations.length > 0 && formData.configId
    : models.length > 0 && formData.modelId
  );
  
  return (
    <div className="client-form">
      <h2>Start New Client</h2>
      
      <div className="mode-selector">
        <div className="mode-tabs">
          <button 
            type="button" 
            className={`mode-tab ${formData.mode === 'config' ? 'active' : ''}`}
            onClick={() => handleModeChange('config')}
          >
            Configuration-based
          </button>
          <button 
            type="button" 
            className={`mode-tab ${formData.mode === 'model' ? 'active' : ''}`}
            onClick={() => handleModeChange('model')}
          >
            Model-based
          </button>
        </div>
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="datasetType">Dataset Type:</label>
          <select 
            id="datasetType"
            name="datasetType"
            value={formData.datasetType}
            onChange={handleChange}
            disabled={loading}
          >
            <option value="breast_cancer">Breast Cancer</option>
            <option value="parkinsons">Parkinson's</option>
            <option value="reinopath">Reinopath</option>
          </select>
        </div>
        
        {formData.mode === 'config' ? (
          <>
            <div className="form-group">
              <label htmlFor="configId">Configuration:</label>
              {fetchingConfigs ? (
                <div className="inline-loading">
                  <LoadingSpinner size="small" /> Loading configurations...
                </div>
              ) : configError ? (
                <div className="error-message">{configError}</div>
              ) : configurations.length > 0 ? (
                <select 
                  id="configId"
                  name="configId"
                  value={formData.configId}
                  onChange={handleChange}
                  disabled={loading}
                >
                  {configurations.map(config => (
                    <option key={config.id} value={config.id}>
                      {config.configName}
                    </option>
                  ))}
                </select>
              ) : (
                <div className="note-message">
                  No configurations found for {formData.datasetType}. 
                  <a href="/configurations" className="link-button">Create one</a>
                </div>
              )}
            </div>
            
            <div className="form-group">
              <label htmlFor="cycles">Training Cycles:</label>
              <input 
                type="number"
                id="cycles"
                name="cycles"
                min="1"
                max="10"
                value={formData.cycles}
                onChange={handleChange}
                disabled={loading}
              />
            </div>
          </>
        ) : (
          <div className="form-group">
            <label htmlFor="modelId">Pre-trained Model:</label>
            {fetchingModels ? (
              <div className="inline-loading">
                <LoadingSpinner size="small" /> Loading models...
              </div>
            ) : modelError ? (
              <div className="error-message">{modelError}</div>
            ) : models.length > 0 ? (
              <select 
                id="modelId"
                name="modelId"
                value={formData.modelId}
                onChange={handleChange}
                disabled={loading}
              >
                {models.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            ) : (
              <div className="note-message">
                No models found for {formData.datasetType}. 
                <a href="/models" className="link-button">Upload one</a>
              </div>
            )}
            <p className="helper-text">
              Using pre-trained models skips the training step and only requires data for parameter sharing.
            </p>
          </div>
        )}
        
        <div className="form-group">
          <label htmlFor="serverHost">Server Host:</label>
          <input 
            type="text"
            id="serverHost"
            name="serverHost"
            value={formData.serverHost}
            onChange={handleChange}
            disabled={loading}
          />
        </div>
        
        <button 
          type="submit" 
          className="btn btn-primary"
          disabled={loading || !canStart}
        >
          {loading ? 'Starting...' : 'Start Client'}
        </button>
      </form>
      
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default ClientForm;