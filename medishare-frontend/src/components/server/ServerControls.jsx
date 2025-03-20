// src/components/server/ServerControls.jsx
import React, { useState, useEffect } from 'react';
import { useGlobalContext } from '../../context/GlobalContext';
import { configApi } from '../../api/configApi';
import LoadingSpinner from '../common/LoadingSpinner';

const ServerControls = () => {
  const { serverStatus, startServer, stopServer, selectedDatasetType, setSelectedDatasetType } = useGlobalContext();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [configurations, setConfigurations] = useState([]);
  const [fetchingConfigs, setFetchingConfigs] = useState(false);
  const [selectedConfigId, setSelectedConfigId] = useState('');
  
  // Fetch configurations when dataset type changes
  useEffect(() => {
    const fetchConfigurations = async () => {
      if (!selectedDatasetType) return;
      
      setFetchingConfigs(true);
      try {
        const allConfigs = await configApi.getAllConfigurations();
        const filteredConfigs = allConfigs.filter(config => 
          config.datasetType === selectedDatasetType
        );
        setConfigurations(filteredConfigs);
        
        // Select the first config by default if available
        if (filteredConfigs.length > 0) {
          setSelectedConfigId(filteredConfigs[0].id.toString());
        } else {
          setSelectedConfigId('');
        }
      } catch (err) {
        console.error('Failed to fetch configurations:', err);
        setError('Failed to load configurations. Please check if any configurations exist for this dataset type.');
      } finally {
        setFetchingConfigs(false);
      }
    };
    
    fetchConfigurations();
  }, [selectedDatasetType]);
  
  const handleStartServer = async () => {
    setLoading(true);
    setError(null);
    try {
      const success = await startServer(selectedDatasetType, selectedConfigId);
      if (!success) {
        setError('Failed to start server. Please make sure the configuration exists.');
      }
    } catch (err) {
      setError(err.message || 'Failed to start server');
    } finally {
      setLoading(false);
    }
  };
  
  const handleStopServer = async () => {
    setLoading(true);
    setError(null);
    try {
      const success = await stopServer();
      if (!success) {
        setError('Failed to stop server. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'Failed to stop server');
    } finally {
      setLoading(false);
    }
  };
  
  const handleConfigChange = (e) => {
    setSelectedConfigId(e.target.value);
  };
  
  return (
    <div className="server-controls">
      <h2>Server Controls</h2>
      
      {serverStatus.isRunning ? (
        <button 
          className="btn btn-danger"
          onClick={handleStopServer}
          disabled={loading}
        >
          {loading ? 'Stopping...' : 'Stop Server'}
        </button>
      ) : (
        <div className="start-server-form">
          <div className="form-group">
            <label htmlFor="datasetType">Dataset Type:</label>
            <select 
              id="datasetType"
              value={selectedDatasetType}
              onChange={(e) => setSelectedDatasetType(e.target.value)}
              disabled={loading}
              required
            >
              <option value="">Select Dataset Type</option>
              <option value="breast_cancer">Breast Cancer</option>
              <option value="parkinsons">Parkinson's</option>
              <option value="reinopath">Reinopath</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="configId">Configuration:</label>
            {fetchingConfigs ? (
              <div className="inline-loading">
                <LoadingSpinner size="small" /> Loading configurations...
              </div>
            ) : configurations.length > 0 ? (
              <select 
                id="configId"
                value={selectedConfigId}
                onChange={handleConfigChange}
                disabled={loading}
                required
              >
                {configurations.map(config => (
                  <option key={config.id} value={config.id}>
                    {config.configName}
                  </option>
                ))}
              </select>
            ) : (
              <div className="note-message">
                No configurations found for {selectedDatasetType || "selected dataset type"}. 
                <a href="/configurations" className="link-button">Create one</a>
              </div>
            )}
          </div>
          
          <button 
            className="btn btn-primary"
            onClick={handleStartServer}
            disabled={loading || !selectedDatasetType || configurations.length === 0}
          >
            {loading ? 'Starting...' : 'Start Server'}
          </button>
        </div>
      )}
      
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default ServerControls;