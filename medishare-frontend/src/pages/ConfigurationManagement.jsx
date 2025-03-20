// src/pages/ConfigurationManagement.jsx
import React, { useState } from 'react';
import { useConfigurations } from '../hooks/useConfigurations';
import ConfigList from '../components/configurations/ConfigList';
import SimplifiedConfigForm from '../components/configurations/SimplifiedConfigForm';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';

const ConfigurationManagement = () => {
  const { 
    configurations, 
    loading, 
    error, 
    createConfiguration, 
    updateConfiguration, 
    deleteConfiguration 
  } = useConfigurations();

  const [showForm, setShowForm] = useState(false);
  const [editingConfig, setEditingConfig] = useState(null);

  const handleCreateNew = () => {
    setEditingConfig(null);
    setShowForm(true);
  };

  const handleEdit = (config) => {
    setEditingConfig(config);
    setShowForm(true);
  };

  const handleCancel = () => {
    setShowForm(false);
    setEditingConfig(null);
  };

  const handleDelete = async (datasetType) => {
    if (window.confirm(`Are you sure you want to delete the configuration for ${datasetType}?`)) {
      await deleteConfiguration(datasetType);
    }
  };

  const handleSubmit = async (formData) => {
    try {
      if (editingConfig) {
        await updateConfiguration(formData.datasetType, formData);
      } else {
        await createConfiguration(formData);
      }
      setShowForm(false);
      setEditingConfig(null);
      return { success: true };
    } catch (err) {
      return { 
        success: false, 
        error: err.message || 'Failed to save configuration' 
      };
    }
  };

  if (loading && configurations.length === 0) {
    return <LoadingSpinner />;
  }

  return (
    <div className="configuration-management">
      <div className="page-header">
        <h1>Configuration Management</h1>
        {!showForm && (
          <button 
            className="btn btn-primary"
            onClick={handleCreateNew}
          >
            Create New Configuration
          </button>
        )}
      </div>

      {error && <ErrorMessage message={error} />}

      {showForm ? (
        <Card title={editingConfig ? 'Edit Configuration' : 'Create New Configuration'}>
          <SimplifiedConfigForm 
            config={editingConfig}
            onSubmit={handleSubmit}
            onCancel={handleCancel}
          />
        </Card>
      ) : (
        <>
          <ConfigList 
            configurations={configurations}
            onEdit={handleEdit}
            onDelete={handleDelete}
          />
        </>
      )}
    </div>
  );
};

export default ConfigurationManagement;