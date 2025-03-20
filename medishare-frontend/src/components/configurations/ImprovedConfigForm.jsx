// src/components/configurations/ImprovedConfigForm.jsx
import React, { useState, useEffect } from 'react';

const ImprovedConfigForm = ({ config, onSubmit, onCancel }) => {
  // Initialize with default values or existing config
  const [formData, setFormData] = useState({
    datasetType: '',
    configName: '',
    configData: {
      dataset: {
        path: '/app/datasets/data.csv',
        target_column: 'diagnosis'
      },
      training: {
        epochs: 10,
        batch_size: 32,
        learning_rate: 0.001
      },
      model: {
        hidden_layers: [64, 32],
        parameters_file: ''
      },
      server: {
        host: '0.0.0.0',
        port: 8080,
        client_host: '127.0.0.1',
        update_threshold: 1,
        contribution_weight: 0.1,
        backup_dir: ''
      },
      client: {
        cycles: 1,
        wait_time: 10,
        retry_interval: 10
      }
    }
  });
  
  const [activeTab, setActiveTab] = useState('basic');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Set form data from config if it exists
  useEffect(() => {
    if (config) {
      setFormData(config);
    }
  }, [config]);
  
  // Handle basic input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };
  
  // Handle nested object changes
  const handleNestedChange = (section, field, value) => {
    setFormData(prev => ({
      ...prev,
      configData: {
        ...prev.configData,
        [section]: {
          ...prev.configData[section],
          [field]: value
        }
      }
    }));
  };
  
  // Handle array input (hidden_layers)
  const handleArrayChange = (e) => {
    const value = e.target.value;
    // Split by comma and convert to numbers
    const arrayValue = value.split(',').map(item => parseInt(item.trim(), 10));
    
    setFormData(prev => ({
      ...prev,
      configData: {
        ...prev.configData,
        model: {
          ...prev.configData.model,
          hidden_layers: arrayValue
        }
      }
    }));
  };
  
  // Generate parameters file path based on dataset type
  useEffect(() => {
    if (formData.datasetType) {
      const parametersFile = `global_models/${formData.datasetType}_model.json`;
      const backupDir = `model_backups/${formData.datasetType}`;
      
      setFormData(prev => ({
        ...prev,
        configData: {
          ...prev.configData,
          model: {
            ...prev.configData.model,
            parameters_file: parametersFile
          },
          server: {
            ...prev.configData.server,
            backup_dir: backupDir
          }
        }
      }));
    }
  }, [formData.datasetType]);
  
  // Form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const result = await onSubmit(formData);
      if (!result.success) {
        setError(result.error);
      }
    } catch (err) {
      setError(err.message || 'An error occurred while saving the configuration');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="config-form">
      <h2>{config ? 'Edit Configuration' : 'Create Configuration'}</h2>
      
      <form onSubmit={handleSubmit}>
        {/* Basic Information - Always visible */}
        <div className="form-section">
          <h3>Basic Information</h3>
          
          <div className="form-group">
            <label htmlFor="datasetType">Dataset Type:</label>
            <select 
              id="datasetType"
              name="datasetType"
              value={formData.datasetType}
              onChange={handleChange}
              disabled={loading || (config !== null)}
              required
            >
              <option value="">Select Dataset Type</option>
              <option value="breast_cancer">Breast Cancer</option>
              <option value="parkinsons">Parkinson's</option>
              <option value="reinopath">Diabetic Retinopathy</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="configName">Configuration Name:</label>
            <input 
              type="text"
              id="configName"
              name="configName"
              value={formData.configName}
              onChange={handleChange}
              disabled={loading}
              required
            />
          </div>
        </div>
        
        {/* Tab Navigation */}
        <div className="tabs">
          <button 
            type="button"
            className={`tab-button ${activeTab === 'basic' ? 'active' : ''}`} 
            onClick={() => setActiveTab('basic')}
          >
            Dataset & Training
          </button>
          <button 
            type="button"
            className={`tab-button ${activeTab === 'model' ? 'active' : ''}`} 
            onClick={() => setActiveTab('model')}
          >
            Model
          </button>
          <button 
            type="button"
            className={`tab-button ${activeTab === 'server' ? 'active' : ''}`} 
            onClick={() => setActiveTab('server')}
          >
            Server
          </button>
          <button 
            type="button"
            className={`tab-button ${activeTab === 'client' ? 'active' : ''}`} 
            onClick={() => setActiveTab('client')}
          >
            Client
          </button>
        </div>
        
        {/* Dataset and Training Tab */}
        {activeTab === 'basic' && (
          <>
            <div className="form-section">
              <h3>Dataset Configuration</h3>
              
              <div className="form-group">
                <label htmlFor="dataset_path">Dataset Path:</label>
                <input 
                  type="text"
                  id="dataset_path"
                  value={formData.configData.dataset.path}
                  onChange={(e) => handleNestedChange('dataset', 'path', e.target.value)}
                  disabled={loading}
                  required
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="target_column">Target Column:</label>
                <input 
                  type="text"
                  id="target_column"
                  value={formData.configData.dataset.target_column}
                  onChange={(e) => handleNestedChange('dataset', 'target_column', e.target.value)}
                  disabled={loading}
                  required
                />
              </div>
            </div>
            
            <div className="form-section">
              <h3>Training Configuration</h3>
              
              <div className="form-group">
                <label htmlFor="epochs">Epochs:</label>
                <input 
                  type="number"
                  id="epochs"
                  value={formData.configData.training.epochs}
                  onChange={(e) => handleNestedChange('training', 'epochs', parseInt(e.target.value, 10))}
                  min="1"
                  disabled={loading}
                  required
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="batch_size">Batch Size:</label>
                <input 
                  type="number"
                  id="batch_size"
                  value={formData.configData.training.batch_size}
                  onChange={(e) => handleNestedChange('training', 'batch_size', parseInt(e.target.value, 10))}
                  min="1"
                  disabled={loading}
                  required
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="learning_rate">Learning Rate:</label>
                <input 
                  type="number"
                  id="learning_rate"
                  value={formData.configData.training.learning_rate}
                  onChange={(e) => handleNestedChange('training', 'learning_rate', parseFloat(e.target.value))}
                  step="0.0001"
                  min="0.0001"
                  max="1"
                  disabled={loading}
                  required
                />
              </div>
            </div>
          </>
        )}
        
        {/* Model Tab */}
        {activeTab === 'model' && (
          <div className="form-section">
            <h3>Model Configuration</h3>
            
            <div className="form-group">
              <label htmlFor="hidden_layers">Hidden Layers (comma-separated):</label>
              <input 
                type="text"
                id="hidden_layers"
                value={formData.configData.model.hidden_layers.join(', ')}
                onChange={handleArrayChange}
                disabled={loading}
                required
              />
              <small className="help-text">Enter layer sizes separated by commas (e.g., 64, 32)</small>
            </div>
            
            <div className="form-group">
              <label htmlFor="parameters_file">Parameters File Path:</label>
              <input 
                type="text"
                id="parameters_file"
                value={formData.configData.model.parameters_file}
                onChange={(e) => handleNestedChange('model', 'parameters_file', e.target.value)}
                disabled={loading}
                required
              />
              <small className="help-text">This path is auto-generated based on dataset type</small>
            </div>
          </div>
        )}
        
        {/* Server Tab */}
        {activeTab === 'server' && (
          <div className="form-section">
            <h3>Server Configuration</h3>
            
            <div className="form-row">
              <div className="form-group half">
                <label htmlFor="server_host">Host:</label>
                <input 
                  type="text"
                  id="server_host"
                  value={formData.configData.server.host}
                  onChange={(e) => handleNestedChange('server', 'host', e.target.value)}
                  disabled={loading}
                  required
                />
              </div>
              
              <div className="form-group half">
                <label htmlFor="server_port">Port:</label>
                <input 
                  type="number"
                  id="server_port"
                  value={formData.configData.server.port}
                  onChange={(e) => handleNestedChange('server', 'port', parseInt(e.target.value, 10))}
                  min="1"
                  max="65535"
                  disabled={loading}
                  required
                />
              </div>
            </div>
            
            <div className="form-group">
              <label htmlFor="client_host">Client Host:</label>
              <input 
                type="text"
                id="client_host"
                value={formData.configData.server.client_host}
                onChange={(e) => handleNestedChange('server', 'client_host', e.target.value)}
                disabled={loading}
                required
              />
            </div>
            
            <div className="form-row">
              <div className="form-group half">
                <label htmlFor="update_threshold">Update Threshold:</label>
                <input 
                  type="number"
                  id="update_threshold"
                  value={formData.configData.server.update_threshold}
                  onChange={(e) => handleNestedChange('server', 'update_threshold', parseInt(e.target.value, 10))}
                  min="1"
                  disabled={loading}
                  required
                />
                <small className="help-text">Minimum clients before model update</small>
              </div>
              
              <div className="form-group half">
                <label htmlFor="contribution_weight">Contribution Weight:</label>
                <input 
                  type="number"
                  id="contribution_weight"
                  value={formData.configData.server.contribution_weight}
                  onChange={(e) => handleNestedChange('server', 'contribution_weight', parseFloat(e.target.value))}
                  step="0.01"
                  min="0"
                  max="1"
                  disabled={loading}
                  required
                />
                <small className="help-text">How much each client influences the model (0-1)</small>
              </div>
            </div>
            
            <div className="form-group">
              <label htmlFor="backup_dir">Backup Directory:</label>
              <input 
                type="text"
                id="backup_dir"
                value={formData.configData.server.backup_dir}
                onChange={(e) => handleNestedChange('server', 'backup_dir', e.target.value)}
                disabled={loading}
                required
              />
              <small className="help-text">This path is auto-generated based on dataset type</small>
            </div>
          </div>
        )}
        
        {/* Client Tab */}
        {activeTab === 'client' && (
          <div className="form-section">
            <h3>Client Configuration</h3>
            
            <div className="form-row">
              <div className="form-group third">
                <label htmlFor="cycles">Cycles:</label>
                <input 
                  type="number"
                  id="cycles"
                  value={formData.configData.client.cycles}
                  onChange={(e) => handleNestedChange('client', 'cycles', parseInt(e.target.value, 10))}
                  min="1"
                  disabled={loading}
                  required
                />
                <small className="help-text">Training iterations</small>
              </div>
              
              <div className="form-group third">
                <label htmlFor="wait_time">Wait Time (seconds):</label>
                <input 
                  type="number"
                  id="wait_time"
                  value={formData.configData.client.wait_time}
                  onChange={(e) => handleNestedChange('client', 'wait_time', parseInt(e.target.value, 10))}
                  min="1"
                  disabled={loading}
                  required
                />
                <small className="help-text">Between cycles</small>
              </div>
              
              <div className="form-group third">
                <label htmlFor="retry_interval">Retry Interval (seconds):</label>
                <input 
                  type="number"
                  id="retry_interval"
                  value={formData.configData.client.retry_interval}
                  onChange={(e) => handleNestedChange('client', 'retry_interval', parseInt(e.target.value, 10))}
                  min="1"
                  disabled={loading}
                  required
                />
                <small className="help-text">After failure</small>
              </div>
            </div>
          </div>
        )}
        
        {error && <div className="error-message">{error}</div>}
        
        <div className="form-actions">
          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={onCancel}
            disabled={loading}
          >
            Cancel
          </button>
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? 'Saving...' : (config ? 'Update Configuration' : 'Create Configuration')}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ImprovedConfigForm;