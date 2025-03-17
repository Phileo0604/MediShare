// src/components/server/ServerStatus.jsx
import React from 'react';
import { useGlobalContext } from '../../context/GlobalContext';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorMessage from '../common/ErrorMessage';

const ServerStatus = () => {
  const { serverStatus, fetchServerStatus } = useGlobalContext();
  
  if (serverStatus.loading) {
    return <LoadingSpinner />;
  }
  
  if (serverStatus.error) {
    return <ErrorMessage message={serverStatus.error} onRetry={fetchServerStatus} />;
  }
  
  return (
    <div className="server-status">
      <h2>Server Status</h2>
      <div className="status-card">
        <div className="status-indicator">
          <span className={`status-dot ${serverStatus.isRunning ? 'active' : 'inactive'}`}></span>
          <span className="status-text">
            {serverStatus.isRunning ? 'Running' : 'Stopped'}
          </span>
        </div>
        
        {serverStatus.isRunning && (
          <div className="status-details">
            <p><strong>Dataset Type:</strong> {serverStatus.datasetType}</p>
            <p><strong>Started:</strong> {new Date(serverStatus.startTime).toLocaleString()}</p>
            <p><strong>Active Clients:</strong> {serverStatus.activeClients}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ServerStatus;