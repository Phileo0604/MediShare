import React, { useState, useEffect } from 'react';

const ModelTrainingForm = ({ selectedDatasetType, onSubmit, onCancel, status }) => {
  const [formData, setFormData] = useState({
    datasetType: selectedDatasetType || 'breast_cancer',
    datasetFile: null,
    modelName: '',
    description: '',
    epochs: 10,
    batchSize: 32,
    learningRate: 0.001
  });
  
  const [fileSelected, setFileSelected] = useState(false);
  
  // Update dataset type when parent prop changes
  useEffect(() => {
    if (selectedDatasetType) {
      setFormData(prev => ({
        ...prev,
        datasetType: selectedDatasetType
      }));
    }
  }, [selectedDatasetType]);
  
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
  
  // Display success message if training completed
  if (status && status.completed) {
    return (
      <div className="training-success">
        <div className="success-message">{status.message}</div>
        <button className="btn btn-primary" onClick={onCancel}>
          Close Training Form
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
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-bar-fill" 
                style={{ width: `${status.progress || 0}%` }}
              ></div>
            </div>
            <span>{status.progress || 0}%</span>
          </div>
          <div className="progress-message">{status.message}</div>
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

export default ModelTrainingForm;