// src/pages/ModelParametersManagement.jsx
import React, { useState, useEffect } from 'react';
import { modelApi } from '../api/modelApi';
import { clientApi } from '../api/clientApi';
import { useGlobalContext } from '../context/GlobalContext';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { formatDate } from '../utils/formatters';
import '../styles/ModelParameters.css';

const ModelParametersManagement = () => {
  const { selectedDatasetType, setSelectedDatasetType } = useGlobalContext();
  
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadFormData, setUploadFormData] = useState({
    name: '',
    description: '',
    datasetType: selectedDatasetType
  });
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState(null);

  // Fetch models for the selected dataset type
  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const modelsData = await modelApi.getModelsByType(selectedDatasetType);
      setModels(modelsData);
    } catch (err) {
      setError(err.message || 'Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  // Handle dataset type change
  const handleDatasetTypeChange = (e) => {
    setSelectedDatasetType(e.target.value);
    setUploadFormData({
      ...uploadFormData,
      datasetType: e.target.value
    });
  };

  // Handle upload form changes
  const handleUploadFormChange = (e) => {
    const { name, value } = e.target;
    setUploadFormData({
      ...uploadFormData,
      [name]: value
    });
  };

  // Handle file selection
  const handleFileChange = (e) => {
    setUploadFile(e.target.files[0]);
  };

  // Handle model parameters upload
  const handleUpload = async (e) => {
    e.preventDefault();
    if (!uploadFile) {
      setUploadStatus({ error: 'Please select a file to upload' });
      return;
    }

    setUploadStatus({ uploading: true });
    setUploadProgress(0);

    try {
      // Create FormData object for file upload
      const formData = new FormData();
      formData.append('file', uploadFile);
      formData.append('name', uploadFormData.name);
      formData.append('description', uploadFormData.description);
      formData.append('datasetType', uploadFormData.datasetType);

      // Use the API to upload the file
      const result = await modelApi.uploadModelFile(
        uploadFile,
        uploadFormData.name,
        uploadFormData.description,
        uploadFormData.datasetType
      );
      
      setUploadProgress(100);
      
      if (result.success) {
        setUploadStatus({ success: 'Model parameters uploaded successfully' });
        setUploadFile(null);
        setUploadFormData({
          name: '',
          description: '',
          datasetType: selectedDatasetType
        });
        // Refresh the models list
        fetchModels();
      } else {
        setUploadStatus({ error: result.error || 'Failed to upload model parameters' });
      }
    } catch (err) {
      setUploadStatus({ error: err.message || 'Failed to upload model parameters' });
    }
  };

  // Handle model download
  const handleDownload = async (model) => {
    try {
      const blob = await modelApi.downloadModel(model.datasetType);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${model.datasetType}_${model.name}.json`;
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      link.remove();
    } catch (err) {
      setError(err.message || 'Failed to download model parameters');
    }
  };
  
  // Handle starting a client with model parameters
  const handleStartClientWithParameters = async (modelId) => {
    try {
      const result = await clientApi.startClientWithParameters(
        selectedDatasetType,
        modelId,
        '127.0.0.1'  // Default server host
      );
      
      // Display notification of success
      alert(`Client started successfully with ID: ${result.clientId}`);
      
    } catch (err) {
      setError(err.message || 'Failed to start client with model parameters');
    }
  };

  // Fetch models when dataset type changes
  useEffect(() => {
    fetchModels();
  }, [selectedDatasetType]);

  return (
    <div className="model-parameters-management">
      <h1>Model Parameters Management</h1>
      
      {error && <ErrorMessage message={error} onRetry={fetchModels} />}
      
      <div className="selection-bar">
        <div className="form-group">
          <label htmlFor="datasetType">Dataset Type:</label>
          <select 
            id="datasetType"
            value={selectedDatasetType}
            onChange={handleDatasetTypeChange}
          >
            <option value="breast_cancer">Breast Cancer</option>
            <option value="parkinsons">Parkinson's</option>
            <option value="reinopath">Diabetic Retinopathy</option>
          </select>
        </div>
      </div>
      
      <div className="model-parameters-sections">
        <div className="models-list-section">
          <Card title="Available Model Parameters">
            {loading ? (
              <LoadingSpinner />
            ) : models.length > 0 ? (
              <table className="models-table">
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
                  {models.map(model => (
                    <tr key={model.id}>
                      <td>{model.name}</td>
                      <td>{formatDate(model.createdAt)}</td>
                      <td>{model.description}</td>
                      <td>
                        <span className={`status-badge ${model.active ? 'active' : 'inactive'}`}>
                          {model.active ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td>
                        <div className="button-group">
                          <button 
                            className="btn btn-primary btn-sm"
                            onClick={() => handleDownload(model)}
                          >
                            Download
                          </button>
                          <button 
                            className="btn btn-success btn-sm"
                            onClick={() => handleStartClientWithParameters(model.id)}
                          >
                            Start Client
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="no-items-message">No model parameters available for this dataset type.</p>
            )}
          </Card>
        </div>
        
        <div className="upload-section">
          <Card title="Upload Model Parameters">
            <form onSubmit={handleUpload}>
              <div className="form-group">
                <label htmlFor="name">Model Name:</label>
                <input 
                  type="text"
                  id="name"
                  name="name"
                  value={uploadFormData.name}
                  onChange={handleUploadFormChange}
                  required
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="description">Description:</label>
                <textarea 
                  id="description"
                  name="description"
                  value={uploadFormData.description}
                  onChange={handleUploadFormChange}
                  rows="3"
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="file">Model Parameters File:</label>
                <input 
                  type="file"
                  id="file"
                  onChange={handleFileChange}
                  accept=".json,.pkl"
                  required
                />
                <small className="help-text">
                  Upload JSON or PKL file containing model parameters
                </small>
              </div>
              
              {uploadStatus?.uploading && (
                <div className="upload-progress">
                  <div className="progress-bar">
                    <div 
                      className="progress-bar-fill" 
                      style={{ width: `${uploadProgress}%` }}
                    ></div>
                  </div>
                  <span>{uploadProgress}%</span>
                </div>
              )}
              
              {uploadStatus?.error && (
                <p className="error-message">{uploadStatus.error}</p>
              )}
              
              {uploadStatus?.success && (
                <p className="success-message">{uploadStatus.success}</p>
              )}
              
              <button 
                type="submit" 
                className="btn btn-primary"
                disabled={uploadStatus?.uploading}
              >
                {uploadStatus?.uploading ? 'Uploading...' : 'Upload Parameters'}
              </button>
            </form>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ModelParametersManagement;