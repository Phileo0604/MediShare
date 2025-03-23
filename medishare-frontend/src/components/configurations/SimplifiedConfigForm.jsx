// src/components/configurations/SimplifiedConfigForm.jsx
import React, { useState, useEffect } from 'react';
import { useModels } from '../../hooks/useModels';

const SimplifiedConfigForm = ({ config, onSubmit, onCancel, autoGenerateParameterPath = true }) => {
  const [selectedDatasetType, setSelectedDatasetType] = useState('');
  const { models, loading: modelsLoading, fetchModels } = useModels(selectedDatasetType);
  
  // Initialize with default values or existing config
  const [formData, setFormData] = useState({
    datasetType: '',
    configName: '',
    configData: {
      dataset: {
        // Add placeholder dataset info that will be populated based on dataset type
        path: '',
        target_column: ''
      },
      model: {
        modelId: '',
        parameters_file: ''
      },
      server: {
        host: '0.0.0.0',
        port: 8080,
        client_host: '127.0.0.1',
        update_threshold: 1,
        contribution_weight: 0.1
      },
      client: {
        cycles: 1,
        retry_interval: 10
      }
    }
  });

  // Update dataset info based on dataset type
  useEffect(() => {
    if (formData.datasetType) {
      let datasetPath, targetColumn;
      
      switch (formData.datasetType) {
        case 'breast_cancer':
          datasetPath = 'datasets/breast_cancer_data.csv';
          targetColumn = 'diagnosis';
          break;
        case 'parkinsons':
          datasetPath = 'datasets/parkinsons_data.csv';
          targetColumn = 'UPDRS';
          break;
        case 'reinopath':
          datasetPath = 'datasets/reinopath_data.csv';
          targetColumn = 'class';
          break;
        default:
          datasetPath = `datasets/${formData.datasetType}_data.csv`;
          targetColumn = 'target';
          break;
      }
      
      setFormData(prev => ({
        ...prev,
        configData: {
          ...prev.configData,
          dataset: {
            path: datasetPath,
            target_column: targetColumn
          }
        }
      }));
      
      // Refetch models when dataset type changes
      fetchModels();
    }
  }, [formData.datasetType]);
  
  // Set form data from config if it exists
  useEffect(() => {
    if (config) {
      // Handle both old and new format
      if (config.configData) {
        // New format
        setFormData(config);
        setSelectedDatasetType(config.datasetType);
      } else if (config.configJson) {
        // Format where configData is in configJson
        try {
          const configData = typeof config.configJson === 'string'
            ? JSON.parse(config.configJson)
            : config.configJson;
          
          setFormData({
            datasetType: config.datasetType,
            configName: config.configName,
            configData: configData
          });
          setSelectedDatasetType(config.datasetType);
        } catch (error) {
          console.error("Error parsing config JSON:", error);
        }
      } else {
        // Old format - convert to new format
        const newFormat = {
          datasetType: config.datasetType || config.dataset_type,
          configName: config.configName || config.name || 'Configuration',
          configData: {
            model: config.model || {},
            server: config.server || {},
            client: config.client || {}
          }
        };
        setFormData(newFormat);
        setSelectedDatasetType(newFormat.datasetType);
      }
    }
  }, [config]);
  
  // Handle basic input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    if (name === 'datasetType') {
      setSelectedDatasetType(value);
    }
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
  
  // Select the newest model when models are loaded
  useEffect(() => {
    if (models && models.length > 0 && !formData.configData?.model?.modelId) {
      // Sort models by ID descending (assuming newer models have higher IDs)
      const sortedModels = [...models].sort((a, b) => b.id - a.id);
      
      // Select the newest model for this dataset type
      const newestModel = sortedModels.find(model => model.datasetType === selectedDatasetType);
      
      if (newestModel) {
        handleNestedChange('model', 'modelId', newestModel.id.toString());
        console.log(`Auto-selected newest model: ${newestModel.name} (ID: ${newestModel.id})`);
      }
    }
  }, [models, selectedDatasetType]);
  
  // Update parameters file based on dataset type and model ID
  useEffect(() => {
    if (autoGenerateParameterPath && selectedDatasetType) {
      let parametersFile;
      
      // If a model ID is selected, get the file path from the model
      if (formData.configData?.model?.modelId && models) {
        const selectedModel = models.find(model => model.id.toString() === formData.configData.model.modelId);
        if (selectedModel && selectedModel.filePath) {
          // Extract just the filename part, not the full path
          const fileName = selectedModel.filePath.split('/').pop();
          parametersFile = `global_models/${fileName}`;
        }
      } else {
        // Otherwise, use the default path for the dataset type
        switch (selectedDatasetType) {
          case 'breast_cancer':
            parametersFile = 'global_models/breast_cancer_model.json';
            break;
          case 'parkinsons':
            parametersFile = 'global_models/parkinsons_model.pkl';
            break;
          case 'reinopath':
            parametersFile = 'global_models/reinopath_model.pkl';
            break;
          default:
            parametersFile = `global_models/${selectedDatasetType}_model.json`;
            break;
        }
      }
      
      handleNestedChange('model', 'parameters_file', parametersFile);
    }
  }, [selectedDatasetType, formData.configData?.model?.modelId, models, autoGenerateParameterPath]);
  
  // Update model selection when a model is chosen
  const handleModelSelect = (e) => {
    const modelId = e.target.value;
    handleNestedChange('model', 'modelId', modelId);
  };
  
  // Refresh models list
  const refreshModels = () => {
    if (selectedDatasetType) {
      fetchModels();
    }
  };
  
  // Form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      // Make sure configData is properly structured for API
      const submissionData = {
        ...formData,
        // Ensure configData is an object not a string
        configData: typeof formData.configData === 'string'
          ? JSON.parse(formData.configData)
          : formData.configData
      };
      
      const result = await onSubmit(submissionData);
      return result;
    } catch (err) {
      console.error('Form submission error:', err);
      return { success: false, error: err.message || 'An unexpected error occurred' };
    }
  };
  
  return (
    <div className="simplified-config-form">
      <form onSubmit={handleSubmit}>
        <div className="form-section">
          <h3>Basic Information</h3>
          
          <div className="form-group">
            <label htmlFor="datasetType">Dataset Type:</label>
            <select 
              id="datasetType"
              name="datasetType"
              value={formData.datasetType}
              onChange={handleChange}
              disabled={config !== null}
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
              placeholder="Enter a name for this configuration"
              required
            />
          </div>
        </div>
        
        <div className="form-section">
          <h3>Model Selection</h3>
          
          <div className="model-selection-header">
            <label htmlFor="modelId">Select Pre-trained Model:</label>
            <button 
              type="button" 
              className="btn btn-secondary btn-sm"
              onClick={refreshModels}
              disabled={!selectedDatasetType || modelsLoading}
            >
              {modelsLoading ? 'Refreshing...' : 'Refresh Models'}
            </button>
          </div>
          
          <select
            id="modelId"
            value={formData.configData?.model?.modelId || ''}
            onChange={handleModelSelect}
            disabled={modelsLoading || !formData.datasetType}
            required
            className="model-select"
          >
            <option value="">Select a Model</option>
            {!modelsLoading && models && models
              .filter(model => model.datasetType === formData.datasetType)
              .map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} {model.createdAt ? `(${new Date(model.createdAt).toLocaleString()})` : ''}
                </option>
              ))
            }
          </select>
          {modelsLoading && <p className="form-helper">Loading models...</p>}
          {!modelsLoading && models && models.filter(model => model.datasetType === formData.datasetType).length === 0 && (
            <p className="form-helper">No models available for this dataset type. Please upload one first.</p>
          )}
          
          {autoGenerateParameterPath ? (
            <div className="form-group">
              <label htmlFor="parameters_file">Parameters File Path:</label>
              <input 
                type="text"
                id="parameters_file"
                value={formData.configData?.model?.parameters_file || ''}
                className="read-only-input"
                readOnly
              />
              <p className="form-helper">Path is automatically generated based on model selection</p>
            </div>
          ) : (
            <div className="form-group">
              <label htmlFor="parameters_file">Parameters File Path:</label>
              <input 
                type="text"
                id="parameters_file"
                value={formData.configData?.model?.parameters_file || ''}
                onChange={(e) => handleNestedChange('model', 'parameters_file', e.target.value)}
                placeholder="Path where model parameters will be stored"
                required
              />
              <p className="form-helper">Path where model parameters will be stored on the server</p>
            </div>
          )}
        </div>
        
        <div className="form-section">
          <h3>Server Configuration</h3>
          
          <div className="form-row">
            <div className="form-group half">
              <label htmlFor="server_host">Host:</label>
              <input 
                type="text"
                id="server_host"
                value={formData.configData?.server?.host || '0.0.0.0'}
                onChange={(e) => handleNestedChange('server', 'host', e.target.value)}
                placeholder="Server host address"
                required
              />
            </div>
            
            <div className="form-group half">
              <label htmlFor="server_port">Port:</label>
              <input 
                type="number"
                id="server_port"
                value={formData.configData?.server?.port || 8080}
                onChange={(e) => handleNestedChange('server', 'port', parseInt(e.target.value, 10))}
                min="1"
                max="65535"
                placeholder="Server port"
                required
              />
            </div>
          </div>
          
          <div className="form-group">
            <label htmlFor="client_host">Client Host:</label>
            <input 
              type="text"
              id="client_host"
              value={formData.configData?.server?.client_host || '127.0.0.1'}
              onChange={(e) => handleNestedChange('server', 'client_host', e.target.value)}
              placeholder="Client connection address"
              required
            />
            <p className="form-helper">Address clients will use to connect to the server</p>
          </div>
          
          <div className="form-row">
            <div className="form-group half">
              <label htmlFor="update_threshold">Update Threshold:</label>
              <input 
                type="number"
                id="update_threshold"
                value={formData.configData?.server?.update_threshold || 1}
                onChange={(e) => handleNestedChange('server', 'update_threshold', parseInt(e.target.value, 10))}
                min="1"
                placeholder="Model update threshold"
                required
              />
              <p className="form-helper">Minimum client count before updating the global model</p>
            </div>
            
            <div className="form-group half">
              <label htmlFor="contribution_weight">Contribution Weight:</label>
              <input 
                type="number"
                id="contribution_weight"
                value={formData.configData?.server?.contribution_weight || 0.1}
                onChange={(e) => handleNestedChange('server', 'contribution_weight', parseFloat(e.target.value))}
                step="0.01"
                min="0"
                max="1"
                placeholder="Client contribution weight"
                required
              />
              <p className="form-helper">Weight of new client contributions (0-1)</p>
            </div>
          </div>
        </div>
        
        <div className="form-section">
          <h3>Client Configuration</h3>
          
          <div className="form-row">
            <div className="form-group half">
              <label htmlFor="cycles">Default Cycles:</label>
              <input 
                type="number"
                id="cycles"
                value={formData.configData?.client?.cycles || 1}
                onChange={(e) => handleNestedChange('client', 'cycles', parseInt(e.target.value, 10))}
                min="1"
                placeholder="Default training cycles"
                required
              />
              <p className="form-helper">Default number of training cycles</p>
            </div>
            
            <div className="form-group half">
              <label htmlFor="retry_interval">Retry Interval (seconds):</label>
              <input 
                type="number"
                id="retry_interval"
                value={formData.configData?.client?.retry_interval || 10}
                onChange={(e) => handleNestedChange('client', 'retry_interval', parseInt(e.target.value, 10))}
                min="1"
                placeholder="Connection retry interval"
                required
              />
              <p className="form-helper">Seconds to wait between connection attempts</p>
            </div>
          </div>
        </div>
        
        <div className="form-actions">
          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={onCancel}
          >
            Cancel
          </button>
          <button 
            type="submit" 
            className="btn btn-primary"
          >
            {config ? 'Update Configuration' : 'Create Configuration'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SimplifiedConfigForm;