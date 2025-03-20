// src/components/client/EnhancedClientForm.jsx
import React, { useState, useEffect } from 'react';
import { clientApi } from '../../api/clientApi';
import { configApi } from '../../api/configApi';
import { useGlobalContext } from '../../context/GlobalContext';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorMessage from '../common/ErrorMessage';

const EnhancedClientForm = ({ onClientStarted }) => {
  const { selectedDatasetType, setSelectedDatasetType } = useGlobalContext();
  
  const [formData, setFormData] = useState({
    datasetType: selectedDatasetType,
    configId: '',
    cycles: 3,
    serverHost: '127.0.0.1'
  });
  
  const [configurations, setConfigurations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetchingConfigs, setFetchingConfigs] = useState(true);
  const [error, setError] = useState(null);
  const [configError, setConfigError] = useState(null);
  
  // Fetch available configurations for the selected dataset type
  useEffect(() => {
    const fetchConfigurations = async () => {
      setFetchingConfigs(true);
      setConfigError(null);
      
      try {
        const response = await configApi.getAllConfigurations();
        // Filter configurations for the selected dataset type
        const filteredConfigs = response.filter(
          config => config.datasetType === formData.datasetType
        );
        setConfigurations(filteredConfigs);
        
        // If there's at least one configuration, select it by default
        if (filteredConfigs.length > 0 && !formData.configId) {
          setFormData(prev => ({
            ...prev,
            configId: filteredConfigs[0].id.toString()
          }));
        }
      } catch (err) {
        setConfigError(err.message || 'Failed to load configurations');
      } finally {
        setFetchingConfigs(false);
      }
    };
    
    fetchConfigurations();
  }, [formData.datasetType]);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (name === 'datasetType') {
      setSelectedDatasetType(value);
    }
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const result = await clientApi.startClient(
        formData.datasetType, 
        formData.cycles, 
        formData.serverHost,
        formData.configId
      );
      
      if (result && result.clientId) {
        if (onClientStarted) {
          onClientStarted(result);
        }
        // Reset form
        setFormData({
          datasetType: selectedDatasetType,
          configId: formData.configId,
          cycles: 3,
          serverHost: '127.0.0.1'
        });
      } else {
        setError('Failed to start client. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'Failed to start client');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="client-form">
      <h2>Start New Client</h2>
      
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
            <option value="reinopath">Diabetic Retinopathy</option>
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="configId">Configuration:</label>
          {fetchingConfigs ? (
            <div className="inline-loading">Loading configurations...</div>
          ) : configError ? (
            <div className="inline-error">{configError}</div>
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
            <div className="inline-error">
              No configurations found for {formData.datasetType}. 
              <a href="/configurations">Create one</a>
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
          disabled={loading || configurations.length === 0}
        >
          {loading ? 'Starting...' : 'Start Client'}
        </button>
      </form>
      
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default EnhancedClientForm;