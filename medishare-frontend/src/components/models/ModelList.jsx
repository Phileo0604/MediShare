// src/components/models/ModelList.jsx
import React from 'react';

const ModelList = ({ models, activeModel, onDownload }) => {
  if (!models || models.length === 0) {
    return <p className="no-items-message">No models found for this dataset type.</p>;
  }

  return (
    <div className="model-list">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Created</th>
            <th>Description</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.id} className={activeModel && activeModel.id === model.id ? 'active-row' : ''}>
              <td>{model.name}</td>
              <td>{new Date(model.createdAt).toLocaleString()}</td>
              <td>{model.description}</td>
              <td>
                {activeModel && activeModel.id === model.id ? (
                  <span className="status-badge active">Active</span>
                ) : (
                  <span className="status-badge inactive">Inactive</span>
                )}
              </td>
              <td>
                <button 
                  className="btn btn-primary btn-sm"
                  onClick={() => onDownload(model.id)}
                >
                  Download
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ModelList;


