// src/utils/validators.js
/**
 * Validate a configuration object
 * @param {Object} config - Configuration object
 * @returns {Object} Validation result with isValid and errors
 */
export const validateConfiguration = (config) => {
    const errors = {};
    
    // Check required fields
    if (!config.datasetType) {
      errors.datasetType = 'Dataset type is required';
    }
    
    if (!config.configName) {
      errors.configName = 'Configuration name is required';
    }
    
    // Check dataset configuration
    if (!config.configData?.dataset?.path) {
      errors.datasetPath = 'Dataset path is required';
    }
    
    if (!config.configData?.dataset?.target_column) {
      errors.targetColumn = 'Target column is required';
    }
    
    // Check training configuration
    if (!config.configData?.training?.epochs || config.configData.training.epochs < 1) {
      errors.epochs = 'Epochs must be at least 1';
    }
    
    if (!config.configData?.training?.batch_size || config.configData.training.batch_size < 1) {
      errors.batchSize = 'Batch size must be at least 1';
    }
    
    if (!config.configData?.training?.learning_rate || config.configData.training.learning_rate <= 0) {
      errors.learningRate = 'Learning rate must be greater than 0';
    }
    
    // Check model configuration
    if (!config.configData?.model?.hidden_layers || !Array.isArray(config.configData.model.hidden_layers) || config.configData.model.hidden_layers.length === 0) {
      errors.hiddenLayers = 'At least one hidden layer is required';
    }
    
    if (!config.configData?.model?.parameters_file) {
      errors.parametersFile = 'Parameters file is required';
    }
    
    return {
      isValid: Object.keys(errors).length === 0,
      errors
    };
  };
  
  /**
   * Validate a server host and port
   * @param {string} host - Server host
   * @param {number} port - Server port
   * @returns {Object} Validation result with isValid and error
   */
  export const validateServerAddress = (host, port) => {
    let error = null;
    
    if (!host) {
      error = 'Server host is required';
    } else if (!port) {
      error = 'Server port is required';
    } else if (port < 1 || port > 65535) {
      error = 'Port must be between 1 and 65535';
    }
    
    return {
      isValid: error === null,
      error
    };
  };