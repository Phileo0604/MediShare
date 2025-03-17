// src/components/server/ServerControls.jsx
import React, { useState } from 'react';
import { useGlobalContext } from '../../context/GlobalContext';

const ServerControls = () => {
  const { serverStatus, startServer, stopServer, selectedDatasetType, setSelectedDatasetType } = useGlobalContext();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const handleStartServer = async () => {
    setLoading(true);
    setError(null);
    try {
      const success = await startServer(selectedDatasetType);
      if (!success) {
        setError('Failed to start server. Please try again.');
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
            >
              <option value="breast_cancer">Breast Cancer</option>
              <option value="parkinsons">Parkinson's</option>
              <option value="reinopath">Reinopath</option>
            </select>
          </div>
          
          <button 
            className="btn btn-primary"
            onClick={handleStartServer}
            disabled={loading}
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
