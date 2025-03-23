// src/components/configurations/ConfigList.jsx
import React from 'react';
import { formatDate, formatDatasetType } from '../../utils/formatters';

const ConfigList = ({ configurations, onEdit, onDelete }) => {
  if (!configurations || configurations.length === 0) {
    return <p className="no-items-message">No configurations found. Create one to get started.</p>;
  }

  return (
    <div className="config-list">
      <table className="config-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Dataset Type</th>
            <th>Model</th>
            <th>Server Settings</th>
            <th>Client Settings</th>
            <th>Created</th>
            <th>Updated</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {configurations.map((config) => (
            <tr key={config.id}>
              <td>{config.configName}</td>
              <td>
                <span className="dataset-badge">{formatDatasetType(config.datasetType)}</span>
              </td>
              <td>
                {config.configData?.model?.modelId ? (
                  <span className="info-text">ID: {config.configData.model.modelId}</span>
                ) : (
                  <span className="info-text">{config.configData?.model?.parameters_file || 'Not specified'}</span>
                )}
              </td>
              <td>
                <div className="server-info">
                  <span className="info-text">
                    <strong>Host:</strong> {config.configData?.server?.host || '0.0.0.0'}
                  </span>
                  <span className="info-text">
                    <strong>Port:</strong> {config.configData?.server?.port || '8080'}
                  </span>
                </div>
              </td>
              <td>
                <div className="client-info">
                  <span className="info-text">
                    <strong>Cycles:</strong> {config.configData?.client?.cycles || '1'}
                  </span>
                  <span className="info-text">
                    <strong>Retry:</strong> {config.configData?.client?.retry_interval || '10'}s
                  </span>
                </div>
              </td>
              <td>{formatDate(config.createdAt)}</td>
              <td>{formatDate(config.updatedAt)}</td>
              <td>
                <div className="action-buttons">
                  <button 
                    className="btn btn-primary btn-sm"
                    onClick={() => onEdit(config)}
                  >
                    Edit
                  </button>
                  <button 
                    className="btn btn-danger btn-sm"
                    onClick={() => onDelete(config.datasetType)}
                  >
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ConfigList;