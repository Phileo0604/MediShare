// src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { configApi } from '../api/configApi';
import { modelApi } from '../api/modelApi';
import { serverApi } from '../api/serverApi';
import { clientApi } from '../api/clientApi';
import { useGlobalContext } from '../context/GlobalContext';
import ServerStatus from '../components/server/ServerStatus';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import Card from '../components/common/Card';
import { formatDate } from '../utils/formatters';
import '../styles/GlobalModelManagement.css';

const Dashboard = () => {
  const { serverStatus, activeClients, selectedDatasetType, setSelectedDatasetType } = useGlobalContext();
  
  // Dashboard stats
  const [configCount, setConfigCount] = useState(0);
  const [modelCount, setModelCount] = useState(0);
  const [clientCount, setClientCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Global model management state
  const [activeModel, setActiveModel] = useState(null);
  const [modelHistory, setModelHistory] = useState([]);
  const [configurations, setConfigurations] = useState([]);
  const [selectedConfig, setSelectedConfig] = useState('');
  const [federationParams, setFederationParams] = useState({
    contributionWeight: 0.1,
    updateThreshold: 1
  });
  const [configLoading, setConfigLoading] = useState(true);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelError, setModelError] = useState(null);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [modelUploadOpen, setModelUploadOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadFormData, setUploadFormData] = useState({
    name: '',
    description: '',
    datasetType: selectedDatasetType
  });
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [clearingData, setClearingData] = useState(false);
  const [clearStatus, setClearStatus] = useState(null);
  // Fetch dashboard data
  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch configurations
      const configs = await configApi.getAllConfigurations();
      setConfigCount(configs.length || 0);
      
      // Fetch models
      const models = await modelApi.getAllModels();
      setModelCount(models.length || 0);
      
      // Fetch active clients (can use the count from context if available)
      setClientCount(activeClients.length);
      
    } catch (err) {
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch model data
  const fetchModelData = async () => {
    setModelLoading(true);
    setModelError(null);
    
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
      setModelError(err.message || 'Failed to load model data');
    } finally {
      setModelLoading(false);
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
      setModelError(err.message || 'Failed to download model parameters');
    }
  };
  
  // Set a model as the active global model
  const activateModel = async (modelId) => {
    try {
      await modelApi.activateModel(modelId, selectedDatasetType);
      
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
  
  // Start a new training round with the global model
  const startTrainingRound = async () => {
    try {
      // Check if server is running
      if (!serverStatus.isRunning) {
        return { success: false, error: 'Server is not running. Please start the server first.' };
      }
      
      // Call API to start training round
      const result = await modelApi.startTrainingRound(selectedDatasetType);
      return result.success 
        ? { success: true, message: 'Training round initiated successfully.' }
        : { success: false, error: result.error || 'Failed to start training round' };
    } catch (err) {
      return { success: false, error: err.message || 'Failed to start training round' };
    }
  };
  
  // File upload handlers
  const handleFileChange = (e) => {
    setUploadFile(e.target.files[0]);
  };
  
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
  
  // Load data on component mount and when dataset type changes
  useEffect(() => {
    fetchDashboardData();
  }, []);
  
  useEffect(() => {
    fetchModelData();
    fetchConfigurations();
  }, [selectedDatasetType]);
  
  if (loading && !activeModel && modelHistory.length === 0) {
    return <LoadingSpinner />;
  }

  const handleClearAll = async () => {
    setClearingData(true);
    setClearStatus(null);
    
    try {
      // Delete all configurations
      const configResult = await configApi.deleteAllConfigurations();
      
      // Delete all models
      const modelResult = await modelApi.deleteAllModels(selectedDatasetType);
      
      // Refresh data
      await fetchDashboardData();
      await fetchModelData();
      await fetchConfigurations();
      
      setClearStatus({ 
        success: true, 
        message: `Successfully deleted ${configResult.count || 0} configurations and ${modelResult.count || 0} models.` 
      });
    } catch (err) {
      setClearStatus({
        success: false,
        error: err.message || 'Failed to clear data. Please try again.'
      });
    } finally {
      setClearingData(false);
      // Close the dialog after a short delay
      setTimeout(() => setShowConfirmDialog(false), 2000);
    }
  };
  
  return (
    <div className="dashboard">
      <h1>MediShare Dashboard</h1>
      
      {error && <ErrorMessage message={error} />}
      
      {/* Top Controls */}
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
          
          <button 
            className="btn btn-danger"
            onClick={() => setShowConfirmDialog(true)}
          >
            Clear All Data
          </button>
        </div>
      </div>
      
      {/* Dashboard Overview Cards */}
      <div className="dashboard-cards">
        <div className="card">
          <h3>Server Status</h3>
          <div className="card-content">
            <div className="status-indicator">
              <span className={`status-dot ${serverStatus.isRunning ? 'active' : 'inactive'}`}></span>
              <span className="status-text">
                {serverStatus.isRunning ? 'Running' : 'Stopped'}
              </span>
            </div>
            {serverStatus.isRunning && (
              <p>Dataset: {serverStatus.datasetType}</p>
            )}
          </div>
        </div>
        
        <div className="card">
          <h3>Active Clients</h3>
          <div className="card-content">
            <div className="stat-number">{activeClients.length}</div>
            <p>connected clients</p>
          </div>
        </div>
        
        <div className="card">
          <h3>Configurations</h3>
          <div className="card-content">
            <div className="stat-number">{configCount}</div>
            <p>available configurations</p>
          </div>
        </div>
        
        <div className="card">
          <h3>Models</h3>
          <div className="card-content">
            <div className="stat-number">{modelCount}</div>
            <p>registered models</p>
          </div>
        </div>
      </div>
      
      {/* Global Model Section */}
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
      
      {/* Quick Navigation Buttons */}
      <div className="quick-actions">
        <h2>Quick Actions</h2>
        <div className="action-buttons">
          <a href="/server" className="btn btn-primary">
            Manage Server
          </a>
          <a href="/clients" className="btn btn-secondary">
            Manage Clients
          </a>
          <a href="/configurations" className="btn btn-secondary">
            Edit Configurations
          </a>
        </div>
      </div>
      {/* Confirmation Dialog */}
        {showConfirmDialog && (
          <div className="confirm-dialog-backdrop">
            <div className="confirm-dialog">
              <h3>Clear All Data</h3>
              
              <div className="confirm-content">
                <p>Are you sure you want to delete all models and configurations for {selectedDatasetType}?</p>
                <p className="warning-text">This action cannot be undone.</p>
                
                {clearStatus?.success && (
                  <p className="success-message">{clearStatus.message}</p>
                )}
                
                {clearStatus?.error && (
                  <p className="error-message">{clearStatus.error}</p>
                )}
              </div>
              
              <div className="confirm-buttons">
                <button 
                  className="btn btn-secondary"
                  onClick={() => setShowConfirmDialog(false)}
                  disabled={clearingData}
                >
                  Cancel
                </button>
                <button 
                  className="btn btn-danger"
                  onClick={handleClearAll}
                  disabled={clearingData}
                >
                  {clearingData ? 'Clearing...' : 'Delete All Data'}
                </button>
              </div>
            </div>
          </div>
        )}
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

export default Dashboard;