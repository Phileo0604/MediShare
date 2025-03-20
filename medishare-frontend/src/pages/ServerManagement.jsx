// src/pages/ServerManagement.jsx
import React, { useState } from 'react';
import ServerStatus from '../components/server/ServerStatus';
import ServerControls from '../components/server/ServerControls';
import LogViewer from '../components/logs/LogViewer';
import { useGlobalContext } from '../context/GlobalContext';

const ServerManagement = () => {
  const { serverStatus, activeClients } = useGlobalContext();
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5 seconds default
  
  // Handle change in refresh interval
  const handleRefreshIntervalChange = (e) => {
    setRefreshInterval(parseInt(e.target.value, 10));
  };
  
  return (
    <div className="server-management">
      <h1>Server Management</h1>
      
      <div className="status-section">
        <ServerStatus />
      </div>
      
      <div className="controls-section">
        <ServerControls />
      </div>
      
      {serverStatus.isRunning && activeClients.length > 0 && (
        <div className="active-clients">
          <h2>Connected Clients</h2>
          <table className="clients-table">
            <thead>
              <tr>
                <th>Client ID</th>
                <th>Status</th>
                <th>Started</th>
                <th>Dataset Type</th>
                <th>Cycles</th>
              </tr>
            </thead>
            <tbody>
              {activeClients.map((client) => (
                <tr key={client.clientId}>
                  <td>{client.clientId}</td>
                  <td>
                    <span className="status-badge active">Active</span>
                  </td>
                  <td>{new Date(client.startTime).toLocaleString()}</td>
                  <td>{client.datasetType}</td>
                  <td>{client.cycles}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      <div className="logs-section">
        <div className="logs-header">
          <h2>Server & Client Logs</h2>
          <div className="refresh-control">
            <label htmlFor="refreshInterval">Refresh Interval:</label>
            <select 
              id="refreshInterval" 
              value={refreshInterval}
              onChange={handleRefreshIntervalChange}
            >
              <option value="1000">1 second</option>
              <option value="5000">5 seconds</option>
              <option value="10000">10 seconds</option>
              <option value="30000">30 seconds</option>
              <option value="60000">1 minute</option>
            </select>
          </div>
        </div>
        <LogViewer maxLines={200} refreshInterval={refreshInterval} />
      </div>
    </div>
  );
};

export default ServerManagement;