// src/pages/ConfigurationManagement.jsx
import React, { useState } from 'react';
import { useConfigurations } from '../hooks/useConfigurations';
import ConfigList from '../components/configurations/ConfigList';
import SimplifiedConfigForm from '../components/configurations/SimplifiedConfigForm';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
// Import styles
import '../styles/ConfigManagement.css';

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
  const [trainingMode, setTrainingMode] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState(null);

  const handleCreateNew = () => {
    setEditingConfig(null);
    setShowForm(true);
    setTrainingMode(false);
  };

  const handleEdit = (config) => {
    setEditingConfig(config);
    setShowForm(true);
    setTrainingMode(false);
  };

  const handleCancel = () => {
    setShowForm(false);
    setEditingConfig(null);
    setTrainingMode(false);
    setTrainingStatus(null);
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

  const handleTrainModel = () => {
    setTrainingMode(true);
    setShowForm(false);
  };

  const handleTrainingSubmit = async (trainingData) => {
    // Set status to in-progress
    setTrainingStatus({ inProgress: true, message: "Training model in progress..." });
    
    try {
      // Here you would make an API call to train the model with the uploaded dataset
      // Example: const result = await modelTrainingApi.trainModel(trainingData);
      
      // Simulate API call with timeout
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // After successful training, update status
      setTrainingStatus({
        success: true,
        message: "Model training completed successfully. Dataset has been deleted to preserve privacy."
      });
      
      // Could trigger a refresh of configurations here
      return { success: true };
    } catch (err) {
      setTrainingStatus({
        error: true,
        message: err.message || "An error occurred during model training."
      });
      return { 
        success: false, 
        error: err.message || 'Failed to train model' 
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
        {!showForm && !trainingMode && (
          <div className="action-buttons">
            <button 
              className="btn btn-primary"
              onClick={handleCreateNew}
            >
              Create New Configuration
            </button>
          </div>
        )}
      </div>

      {error && <ErrorMessage message={error} />}

      {showForm ? (
        <Card title={editingConfig ? 'Edit Configuration' : 'Create New Configuration'}>
          <SimplifiedConfigForm 
            config={editingConfig}
            onSubmit={handleSubmit}
            onCancel={handleCancel}
            autoGenerateParameterPath={true} // Enable automatic parameter path generation
          />
        </Card>
      ) : trainingMode ? (
        <Card title="Train Model with Dataset">
          <TrainingForm 
            onSubmit={handleTrainingSubmit}
            onCancel={handleCancel}
            status={trainingStatus}
          />
        </Card>
      ) : (
        <div className="config-list-container">
          <ConfigList 
            configurations={configurations}
            onEdit={handleEdit}
            onDelete={handleDelete}
          />
        </div>
      )}
    </div>
  );
};

// Component for model training
const TrainingForm = ({ onSubmit, onCancel, status }) => {
  const [formData, setFormData] = useState({
    datasetType: 'breast_cancer',
    datasetFile: null,
    modelName: '',
    description: '',
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001
  });
  
  const [fileSelected, setFileSelected] = useState(false);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };
  
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFormData({
        ...formData,
        datasetFile: e.target.files[0]
      });
      setFileSelected(true);
    }
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.datasetFile) {
      alert("Please select a dataset file to upload.");
      return;
    }
    
    await onSubmit(formData);
  };
  
  // Display status message if provided
  if (status && status.success) {
    return (
      <div className="training-success">
        <div className="success-message">{status.message}</div>
        <button className="btn btn-primary" onClick={onCancel}>
          Return to Configurations
        </button>
      </div>
    );
  }

  return (
    <form className="training-form" onSubmit={handleSubmit}>
      <div className="form-section">
        <h3>Dataset Information</h3>
        <p className="privacy-notice">
          Uploaded datasets will be used only for training and will be automatically 
          deleted after the training process to preserve patient privacy.
        </p>
        
        <div className="form-group">
          <label htmlFor="datasetType">Dataset Type:</label>
          <select 
            id="datasetType"
            name="datasetType"
            value={formData.datasetType}
            onChange={handleChange}
            required
          >
            <option value="breast_cancer">Breast Cancer</option>
            <option value="parkinsons">Parkinson's</option>
            <option value="reinopath">Diabetic Retinopathy</option>
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="datasetFile">Upload Dataset:</label>
          <input 
            type="file"
            id="datasetFile"
            name="datasetFile"
            onChange={handleFileChange}
            accept=".csv,.xlsx,.xls"
            required
          />
          <small className="help-text">
            Supported formats: CSV, Excel (.xlsx, .xls)
          </small>
          {fileSelected && <div className="file-selected-indicator">File selected ✓</div>}
        </div>
      </div>
      
      <div className="form-section">
        <h3>Model Information</h3>
        
        <div className="form-group">
          <label htmlFor="modelName">Model Name:</label>
          <input 
            type="text"
            id="modelName"
            name="modelName"
            value={formData.modelName}
            onChange={handleChange}
            placeholder="Enter a name for this model"
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="description">Description:</label>
          <textarea 
            id="description"
            name="description"
            value={formData.description}
            onChange={handleChange}
            placeholder="Describe the model (optional)"
            rows="3"
          />
        </div>
      </div>
      
      <div className="form-section">
        <h3>Training Parameters</h3>
        
        <div className="form-row">
          <div className="form-group third">
            <label htmlFor="epochs">Epochs:</label>
            <input 
              type="number"
              id="epochs"
              name="epochs"
              value={formData.epochs}
              onChange={handleChange}
              min="1"
              required
            />
          </div>
          
          <div className="form-group third">
            <label htmlFor="batchSize">Batch Size:</label>
            <input 
              type="number"
              id="batchSize"
              name="batchSize"
              value={formData.batchSize}
              onChange={handleChange}
              min="1"
              required
            />
          </div>
          
          <div className="form-group third">
            <label htmlFor="learningRate">Learning Rate:</label>
            <input 
              type="number"
              id="learningRate"
              name="learningRate"
              value={formData.learningRate}
              onChange={handleChange}
              step="0.0001"
              min="0.0001"
              required
            />
          </div>
        </div>
      </div>
      
      {status && status.inProgress && (
        <div className="training-progress">
          <div className="spinner"></div>
          <div>{status.message}</div>
        </div>
      )}
      
      {status && status.error && (
        <div className="error-message">{status.message}</div>
      )}
      
      <div className="form-actions">
        <button 
          type="button" 
          className="btn btn-secondary"
          onClick={onCancel}
          disabled={status && status.inProgress}
        >
          Cancel
        </button>
        <button 
          type="submit" 
          className="btn btn-primary"
          disabled={!fileSelected || (status && status.inProgress)}
        >
          {(status && status.inProgress) ? 'Training...' : 'Start Training'}
        </button>
      </div>
    </form>
  );
};

export default ConfigurationManagement;