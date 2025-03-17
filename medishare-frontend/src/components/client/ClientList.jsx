// src/components/client/ClientList.jsx
import React from 'react';

const ClientList = ({ clients, onStopClient }) => {
  if (!clients || clients.length === 0) {
    return <p className="no-items-message">No active clients found.</p>;
  }

  return (
    <div className="client-list">
      <table>
        <thead>
          <tr>
            <th>Client ID</th>
            <th>Dataset Type</th>
            <th>Status</th>
            <th>Started</th>
            <th>Cycles</th>
            <th>Progress</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {clients.map((client) => (
            <tr key={client.clientId}>
              <td>{client.clientId}</td>
              <td>{client.datasetType}</td>
              <td>
                <span className="status-badge active">Active</span>
              </td>
              <td>{new Date(client.startTime).toLocaleString()}</td>
              <td>{client.cycles}</td>
              <td>
                <div className="progress-bar">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${(client.completedCycles / client.cycles) * 100}%` }}
                  ></div>
                </div>
                <span className="progress-text">
                  {client.completedCycles || 0}/{client.cycles} cycles
                </span>
              </td>
              <td>
                <button 
                  className="btn btn-danger btn-sm"
                  onClick={() => onStopClient(client.clientId)}
                >
                  Stop
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ClientList;