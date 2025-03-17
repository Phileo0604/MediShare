// src/components/client/ClientCard.jsx
import React from 'react';

const ClientCard = ({ client, onStopClient }) => {
  // Calculate progress percentage
  const progressPercentage = (client.completedCycles / client.cycles) * 100 || 0;

  return (
    <div className="client-card card">
      <div className="card-header">
        <h3>Client {client.clientId}</h3>
        <span className="status-badge active">Active</span>
      </div>
      
      <div className="card-body">
        <div className="client-info">
          <p><strong>Dataset Type:</strong> {client.datasetType}</p>
          <p><strong>Started:</strong> {new Date(client.startTime).toLocaleString()}</p>
          <p><strong>Server Host:</strong> {client.serverHost}</p>
          <p><strong>Training Progress:</strong></p>
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-bar-fill" 
                style={{ width: `${progressPercentage}%` }}
              ></div>
            </div>
            <span className="progress-text">
              {client.completedCycles || 0}/{client.cycles} cycles
            </span>
          </div>
        </div>
      </div>
      
      <div className="card-footer">
        <button 
          className="btn btn-danger"
          onClick={() => onStopClient(client.clientId)}
        >
          Stop Client
        </button>
      </div>
    </div>
  );
};

export default ClientCard;