// src/pages/ServerManagement.jsx
import React from 'react';
import ServerStatus from '../components/server/ServerStatus';
import ServerControls from '../components/server/ServerControls';
import { useGlobalContext } from '../context/GlobalContext';

const ServerManagement = () => {
  const { serverStatus, activeClients } = useGlobalContext();
  
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
    </div>
  );
};

export default ServerManagement;