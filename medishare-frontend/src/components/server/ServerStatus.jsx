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
            <p><strong>Started:</strong> {serverStatus.startTime ? new Date(serverStatus.startTime).toLocaleString() : 'Unknown'}</p>
            <p><strong>Active Clients:</strong> {serverStatus.activeClients}</p>
          </div>
        )}
        
        {/* Add debug information section */}
        <div className="debug-info">
          <details>
            <summary>Debug Information</summary>
            <pre className="debug-data">
              {JSON.stringify(serverStatus, null, 2)}
            </pre>
          </details>
        </div>
      </div>
    </div>
  );
};

export default ServerStatus;