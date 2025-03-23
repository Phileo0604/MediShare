// src/components/server/ServerControls.jsx
import React, { useState } from 'react';
import { useGlobalContext } from '../../context/GlobalContext';
import LoadingSpinner from '../common/LoadingSpinner';

const ServerControls = () => {
  const { serverStatus, startServer, stopServer } = useGlobalContext();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Use a default dataset type for now since the API requires it
  const defaultDatasetType = "breast_cancer";
  
  const handleStartServer = async () => {
    setLoading(true);
    setError(null);
    try {
      const success = await startServer(); // Remove the parameters to use default config
      
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
          <div className="server-info">
            <p>The server will start with the default breast cancer dataset configuration.</p>
            <p>Once running, the server can handle multiple dataset types simultaneously as clients connect.</p>
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