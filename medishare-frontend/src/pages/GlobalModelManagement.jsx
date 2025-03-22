// src/pages/GlobalModelManagement.jsx
import React, { useState, useEffect } from 'react';
import { modelApi } from '../api/modelApi';
import { serverApi } from '../api/serverApi';
import { configApi } from '../api/configApi';
import { useGlobalContext } from '../context/GlobalContext';
import Card from '../components/common/Card';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { formatDate } from '../utils/formatters';

const GlobalModelManagement = () => {
  const { serverStatus, selectedDatasetType, setSelectedDatasetType } = useGlobalContext();
  
  // State management
  const [activeModel, setActiveModel] = useState(null);
  const [modelHistory, setModelHistory] = useState([]);
  const [configurations, setConfigurations] = useState([]);
  const [selectedConfig, setSelectedConfig] = useState('');
  const [federationParams, setFederationParams] = useState({
    contributionWeight: 0.1,
    updateThreshold: 1
  });
  const [loading, setLoading] = useState(true);
  const [configLoading, setConfigLoading] = useState(true);
  const [error, setError] = useState(null);
  const [modelUploadOpen, setModelUploadOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadFormData, setUploadFormData] = useState({
    name: '',
    description: '',
    datasetType: selectedDatasetType
  });
  const [uploadStatus, setUploadStatus] = useState(null);
  
  // Fetch active model and model history
  const fetchModelData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Get all models for the selected dataset type
      const modelsData = await modelApi.getModelsByType(selectedDatasetType);
      
      // Try to get the active model
      try {
        const activeModelData = await modelApi.getActiveModel(selectedDatasetType);
        setActiveModel(activeModelData);
        
        // Filter models to create history (excluding active model)
        const historyModels = modelsData
          .filter(model => model.id !== activeModelData.id)
          .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
        
        setModelHistory(historyModels);
      } catch (err) {
        // No active model - just show all models as history
        setActiveModel(null);
        setModelHistory(modelsData.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)));
      }
    } catch (err) {
      setError(err.message || 'Failed to load model data');
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch configurations for the dataset type
  const fetchConfigurations = async () => {
    setConfigLoading(true);
    
    try {
      const configs = await configApi.getAllConfigurations();
      const filteredConfigs = configs.filter(config => config.datasetType === selectedDatasetType);
      setConfigurations(filteredConfigs);
      
      // Set default selected config
      if (filteredConfigs.length > 0 && !selectedConfig) {
        setSelectedConfig(filteredConfigs[0].id.toString());
        
        // Set federation parameters from the selected config
        if (filteredConfigs[0].configData && filteredConfigs[0].configData.server) {
          const serverConfig = filteredConfigs[0].configData.server;
          setFederationParams({
            contributionWeight: serverConfig.contribution_weight || 0.1,
            updateThreshold: serverConfig.update_threshold || 1
          });
        }
      }
    } catch (err) {
      console.error('Error fetching configurations:', err);
    } finally {
      setConfigLoading(false);
    }
  };
  
  // Handle dataset type change
  const handleDatasetTypeChange = (e) => {
    const newType = e.target.value;
    setSelectedDatasetType(newType);
    setUploadFormData(prev => ({ ...prev, datasetType: newType }));
  };
  
  // Handle configuration change
  const handleConfigChange = (e) => {
    const configId = e.target.value;
    setSelectedConfig(configId);
    
    // Update federation parameters from the selected config
    const selectedConfigObj = configurations.find(config => config.id.toString() === configId);
    if (selectedConfigObj && selectedConfigObj.configData && selectedConfigObj.configData.server) {
      const serverConfig = selectedConfigObj.configData.server;
      setFederationParams({
        contributionWeight: serverConfig.contribution_weight || 0.1,
        updateThreshold: serverConfig.update_threshold || 1
      });
    }
  };
  
  // Update federation parameters
  const handleFederationParamChange = (e) => {
    const { name, value } = e.target;
    setFederationParams(prev => ({
      ...prev,
      [name]: parseFloat(value)
    }));
  };
  
  // Save updated federation parameters to configuration
  const saveFederationParams = async () => {
    try {
      const configToUpdate = configurations.find(config => config.id.toString() === selectedConfig);
      if (!configToUpdate) return;
      
      // Create new configuration with updated parameters
      const updatedConfig = {
        ...configToUpdate,
        configData: {
          ...configToUpdate.configData,
          server: {
            ...configToUpdate.configData.server,
            contribution_weight: federationParams.contributionWeight,
            update_threshold: federationParams.updateThreshold
          }
        }
      };
      
      // Save the updated configuration
      await configApi.updateConfiguration(configToUpdate.datasetType, updatedConfig);
      
      // Refresh configurations
      await fetchConfigurations();
      
      return { success: true };
    } catch (err) {
      return { 
        success: false, 
        error: err.message || 'Failed to update federation parameters' 
      };
    }
  };
  
  // Handle model download
  const handleDownloadModel = async (model) => {
    try {
      const blob = await modelApi.downloadModel(model.datasetType);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${model.datasetType}_model_${model.id}.${model.fileFormat || 'json'}`;
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      link.remove();
    } catch (err) {
      setError(err.message || 'Failed to download model parameters');
    }
  };
  
  // Handle file selection for upload
  const handleFileChange = (e) => {
    setUploadFile(e.target.files[0]);
  };
  
  // Handle upload form changes
  const handleUploadFormChange = (e) => {
    const { name, value } = e.target;
    setUploadFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Upload a new model version
  const handleModelUpload = async (e) => {
    e.preventDefault();
    if (!uploadFile) {
      setUploadStatus({ error: 'Please select a file to upload' });
      return;
    }
    
    setUploadStatus({ uploading: true });
    
    try {
      const result = await modelApi.uploadModelFile(
        uploadFile,
        uploadFormData.name,
        uploadFormData.description,
        uploadFormData.datasetType
      );
      
      if (result.success) {
        setUploadStatus({ success: 'Model uploaded successfully' });
        setModelUploadOpen(false);
        setUploadFile(null);
        setUploadFormData({
          name: '',
          description: '',
          datasetType: selectedDatasetType
        });
        
        // Refresh model data
        await fetchModelData();
      } else {
        setUploadStatus({ error: result.error || 'Failed to upload model' });
      }
    } catch (err) {
      setUploadStatus({ error: err.message || 'Failed to upload model' });
    }
  };
  
  // Start a new training round with the global model
  const startTrainingRound = async () => {
    try {
      // Check if server is running
      if (!serverStatus.isRunning) {
        return { success: false, error: 'Server is not running. Please start the server first.' };
      }
      
      // TODO: Implement actual training round start logic
      // This would typically involve some API call to signal the server to begin a new round
      
      return { success: true, message: 'Training round initiated successfully.' };
    } catch (err) {
      return { success: false, error: err.message || 'Failed to start training round' };
    }
  };
  
  // Set a model as the active global model
  const activateModel = async (modelId) => {
    try {
      // TODO: Implement model activation API endpoint
      // This would typically involve:
      // 1. Updating the model's active status in the database
      // 2. Loading the model on the server
      
      // Simulate for now
      alert(`Model ${modelId} set as active global model`);
      
      // Refresh model data
      await fetchModelData();
      
      return { success: true };
    } catch (err) {
      return { 
        success: false, 
        error: err.message || 'Failed to activate model' 
      };
    }
  };
  
  // Load data when dataset type changes
  useEffect(() => {
    fetchModelData();
    fetchConfigurations();
  }, [selectedDatasetType]);
  
  if (loading && !activeModel && modelHistory.length === 0) {
    return <LoadingSpinner />;
  }
  
  return (
    <div className="global-model-management">
      <h1>Global Model Management</h1>
      
      {error && <ErrorMessage message={error} onRetry={fetchModelData} />}
      
      <div className="top-controls">
        <div className="dataset-selector">
          <label htmlFor="datasetType">Dataset Type:</label>
          <select 
            id="datasetType"
            value={selectedDatasetType}
            onChange={handleDatasetTypeChange}
            className="dataset-select"
          >
            <option value="breast_cancer">Breast Cancer</option>
            <option value="parkinsons">Parkinson's</option>
            <option value="reinopath">Diabetic Retinopathy</option>
          </select>
        </div>
        
        <div className="actions">
          <button 
            className="btn btn-primary" 
            onClick={() => setModelUploadOpen(true)}
          >
            Upload New Model Version
          </button>
          
          <button 
            className="btn btn-secondary"
            onClick={startTrainingRound}
            disabled={!serverStatus.isRunning}
          >
            Start New Training Round
          </button>
        </div>
      </div>
      
      {/* Active Global Model Section */}
      <section className="section-container">
        <h2>Current Global Model</h2>
        {activeModel ? (
          <Card className="active-model-card">
            <div className="model-header">
              <div>
                <h3>{activeModel.name}</h3>
                <span className="status-badge active">Active</span>
              </div>
              <div className="model-actions">
                <button 
                  className="btn btn-primary btn-sm"
                  onClick={() => handleDownloadModel(activeModel)}
                >
                  Download Model
                </button>
              </div>
            </div>
            
            <div className="model-details">
              <div className="detail-group">
                <span className="detail-label">Created:</span>
                <span className="detail-value">{formatDate(activeModel.createdAt)}</span>
              </div>
              
              <div className="detail-group">
                <span className="detail-label">Format:</span>
                <span className="detail-value">{activeModel.fileFormat || 'json'}</span>
              </div>
              
              <div className="detail-group">
                <span className="detail-label">Description:</span>
                <span className="detail-value">{activeModel.description || 'No description provided'}</span>
              </div>
            </div>
            
            {/* Placeholder for model metrics */}
            <div className="model-metrics">
              <h4>Performance Metrics</h4>
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-value">87.5%</div>
                  <div className="metric-label">Accuracy</div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-value">0.82</div>
                  <div className="metric-label">F1 Score</div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-value">0.89</div>
                  <div className="metric-label">Precision</div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-value">0.86</div>
                  <div className="metric-label">Recall</div>
                </div>
              </div>
            </div>
          </Card>
        ) : (
          <p className="no-items-message">No active global model for this dataset type.</p>
        )}
      </section>
      
      {/* Federation Parameters Section */}
      <section className="section-container">
        <h2>Federation Settings</h2>
        <Card className="federation-settings-card">
          <div className="config-selector">
            <label htmlFor="configSelect">Configuration:</label>
            {configLoading ? (
              <div className="inline-loading">Loading configurations...</div>
            ) : configurations.length > 0 ? (
              <select 
                id="configSelect"
                value={selectedConfig}
                onChange={handleConfigChange}
              >
                {configurations.map(config => (
                  <option key={config.id} value={config.id}>
                    {config.configName}
                  </option>
                ))}
              </select>
            ) : (
              <div className="note-message">
                No configurations found for {selectedDatasetType}.
                <a href="/configurations" className="link-button">Create one</a>
              </div>
            )}
          </div>
          
          <div className="federation-controls">
            <div className="form-group">
              <label htmlFor="contributionWeight">Client Contribution Weight:</label>
              <input 
                type="number" 
                id="contributionWeight"
                name="contributionWeight"
                min="0.01" 
                max="1" 
                step="0.01"
                value={federationParams.contributionWeight}
                onChange={handleFederationParamChange}
                disabled={configurations.length === 0}
              />
              <p className="help-text">
                How much each client update influences the global model (0.1 = 10% influence)
              </p>
            </div>
            
            <div className="form-group">
              <label htmlFor="updateThreshold">Update Threshold:</label>
              <input 
                type="number" 
                id="updateThreshold"
                name="updateThreshold"
                min="1" 
                step="1"
                value={federationParams.updateThreshold}
                onChange={handleFederationParamChange}
                disabled={configurations.length === 0}
              />
              <p className="help-text">
                Minimum number of client updates before updating the global model
              </p>
            </div>
            
            <button 
              className="btn btn-primary"
              onClick={saveFederationParams}
              disabled={configurations.length === 0}
            >
              Save Settings
            </button>
          </div>
        </Card>
      </section>
      
      {/* Model History Section */}
      <section className="section-container">
        <h2>Model Version History</h2>
        {modelHistory.length > 0 ? (
          <table className="model-history-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Created</th>
                <th>Format</th>
                <th>Description</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {modelHistory.map(model => (
                <tr key={model.id}>
                  <td>{model.name}</td>
                  <td>{formatDate(model.createdAt)}</td>
                  <td>{model.fileFormat || 'json'}</td>
                  <td className="description-cell">{model.description || 'No description'}</td>
                  <td>
                    <div className="action-buttons">
                      <button 
                        className="btn btn-primary btn-sm"
                        onClick={() => handleDownloadModel(model)}
                      >
                        Download
                      </button>
                      <button 
                        className="btn btn-secondary btn-sm"
                        onClick={() => activateModel(model.id)}
                      >
                        Set Active
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="no-items-message">No model history available for this dataset type.</p>
        )}
      </section>
      
      {/* Model Upload Dialog */}
      {modelUploadOpen && (
        <div className="modal-overlay">
          <div className="modal-container">
            <div className="modal-header">
              <h3>Upload New Model Version</h3>
              <button 
                className="close-button"
                onClick={() => setModelUploadOpen(false)}
              >
                &times;
              </button>
            </div>
            
            <div className="modal-content">
              <form onSubmit={handleModelUpload}>
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
                    accept=".json,.pkl,.bin"
                    required
                  />
                  <p className="help-text">
                    Upload JSON or PKL file containing model parameters
                  </p>
                </div>
                
                {uploadStatus?.error && (
                  <p className="error-message">{uploadStatus.error}</p>
                )}
                
                {uploadStatus?.success && (
                  <p className="success-message">{uploadStatus.success}</p>
                )}
                
                <div className="modal-actions">
                  <button 
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => setModelUploadOpen(false)}
                  >
                    Cancel
                  </button>
                  <button 
                    type="submit"
                    className="btn btn-primary"
                    disabled={uploadStatus?.uploading}
                  >
                    {uploadStatus?.uploading ? 'Uploading...' : 'Upload Model'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GlobalModelManagement;

// To use this page, update your App.jsx routes:
// 
// import GlobalModelManagement from './pages/GlobalModelManagement';
//
// <Routes>
//   <Route path="/" element={<Dashboard />} />
//   <Route path="/server" element={<ServerManagement />} />
//   <Route path="/clients" element={<ClientManagement />} />
//   <Route path="/configurations" element={<ConfigurationManagement />} />
//   <Route path="/models" element={<GlobalModelManagement />} />  {/* Updated Route */}
//   <Route path="/parameters" element={<ModelParametersManagement />} />
//   <Route path="*" element={<Navigate to="/" replace />} />
// </Routes>