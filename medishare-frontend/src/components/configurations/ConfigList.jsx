// src/components/configurations/ConfigList.jsx
import React from 'react';
import ConfigCard from './ConfigCard';

const ConfigList = ({ configurations, onEdit, onDelete }) => {
  if (!configurations || configurations.length === 0) {
    return <p className="no-items-message">No configurations found. Create one to get started.</p>;
  }

  return (
    <div className="config-list">
      {configurations.map((config) => (
        <ConfigCard 
          key={config.datasetType}
          config={config}
          onEdit={onEdit}
          onDelete={onDelete}
        />
      ))}
    </div>
  );
};

export default ConfigList;