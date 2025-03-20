package com.medishare.api.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Component
public class ConfigurationUtils {

    private final ObjectMapper objectMapper;

    public ConfigurationUtils(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    /**
     * Normalize configuration to handle both old and new formats.
     * 
     * @param config Either a flat config or a config with configData nested structure
     * @return Normalized configuration with consistent access pattern
     */
    public Map<String, Object> normalizeConfig(Map<String, Object> config) {
        Map<String, Object> normalized = new HashMap<>();
        
        // Check if we're dealing with new format (with configData)
        Map<String, Object> source;
        if (config.containsKey("configData")) {
            source = (Map<String, Object>) config.get("configData");
            normalized.put("datasetType", config.getOrDefault("datasetType", "breast_cancer"));
            normalized.put("configName", config.getOrDefault("configName", "Default Config"));
        } else {
            source = config;
            normalized.put("datasetType", config.getOrDefault("dataset_type", "breast_cancer"));
            normalized.put("configName", config.getOrDefault("name", "Default Config"));
        }
        
        // Handle dataset section
        Map<String, Object> datasetMap = (Map<String, Object>) source.getOrDefault("dataset", new HashMap<>());
        if (datasetMap.isEmpty()) {
            datasetMap = new HashMap<>();
            datasetMap.put("path", source.getOrDefault("dataset_path", "data.csv"));
            datasetMap.put("target_column", source.getOrDefault("target_column", "target"));
        }
        normalized.put("dataset", datasetMap);
        
        // Handle training section
        Map<String, Object> trainingMap = (Map<String, Object>) source.getOrDefault("training", new HashMap<>());
        if (trainingMap.isEmpty()) {
            trainingMap = new HashMap<>();
            trainingMap.put("epochs", source.getOrDefault("epochs", 10));
            trainingMap.put("batch_size", source.getOrDefault("batch_size", 32));
            trainingMap.put("learning_rate", source.getOrDefault("learning_rate", 0.001));
        }
        normalized.put("training", trainingMap);
        
        // Handle model section
        Map<String, Object> modelMap = (Map<String, Object>) source.getOrDefault("model", new HashMap<>());
        if (modelMap.isEmpty()) {
            modelMap = new HashMap<>();
            
            // Get the dataset type for parameters file path
            String datasetType = normalized.get("datasetType").toString();
            
            modelMap.put("hidden_layers", source.getOrDefault("hidden_layers", new Integer[]{64, 32}));
            modelMap.put("parameters_file", source.getOrDefault("parameters_file", 
                    "global_models/" + datasetType + "_model.json"));
        }
        normalized.put("model", modelMap);
        
        // Handle server section
        Map<String, Object> serverMap = (Map<String, Object>) source.getOrDefault("server", new HashMap<>());
        if (serverMap.isEmpty()) {
            serverMap = new HashMap<>();
            serverMap.put("host", source.getOrDefault("host", "0.0.0.0"));
            serverMap.put("port", source.getOrDefault("port", 8080));
            serverMap.put("client_host", source.getOrDefault("client_host", "127.0.0.1"));
            serverMap.put("update_threshold", source.getOrDefault("update_threshold", 1));
            serverMap.put("contribution_weight", source.getOrDefault("contribution_weight", 0.1));
        }
        normalized.put("server", serverMap);
        
        // Handle client section
        Map<String, Object> clientMap = (Map<String, Object>) source.getOrDefault("client", new HashMap<>());
        if (clientMap.isEmpty()) {
            clientMap = new HashMap<>();
            clientMap.put("cycles", source.getOrDefault("cycles", 1));
            clientMap.put("wait_time", source.getOrDefault("wait_time", 10));
            clientMap.put("retry_interval", source.getOrDefault("retry_interval", 10));
        }
        normalized.put("client", clientMap);
        
        return normalized;
    }
    
    /**
     * Convert a normalized configuration to the new format
     */
    public Map<String, Object> toNewFormat(Map<String, Object> normalized) {
        Map<String, Object> result = new HashMap<>();
        
        result.put("datasetType", normalized.get("datasetType"));
        result.put("configName", normalized.get("configName"));
        
        Map<String, Object> configData = new HashMap<>();
        configData.put("model", normalized.get("model"));
        configData.put("server", normalized.get("server"));
        configData.put("client", normalized.get("client"));
        
        // Only include dataset and training if they were in the original
        if (normalized.containsKey("dataset")) {
            configData.put("dataset", normalized.get("dataset"));
        }
        if (normalized.containsKey("training")) {
            configData.put("training", normalized.get("training"));
        }
        
        result.put("configData", configData);
        return result;
    }
    
    /**
     * Parse and normalize a JSON string configuration
     */
    public Map<String, Object> parseAndNormalize(String configJson) {
        try {
            Map<String, Object> config = objectMapper.readValue(configJson, Map.class);
            return normalizeConfig(config);
        } catch (Exception e) {
            throw new RuntimeException("Failed to parse configuration JSON", e);
        }
    }
    
    /**
     * Create a minimal configuration for the given dataset type
     */
    public Map<String, Object> createMinimalConfig(String datasetType) {
        Map<String, Object> config = new HashMap<>();
        config.put("datasetType", datasetType);
        config.put("configName", datasetType + " Default Config");
        
        Map<String, Object> configData = new HashMap<>();
        
        // Server section
        Map<String, Object> server = new HashMap<>();
        server.put("host", "0.0.0.0");
        server.put("port", 8080);
        server.put("client_host", "127.0.0.1");
        server.put("update_threshold", 1);
        server.put("contribution_weight", 0.1);
        configData.put("server", server);
        
        // Model section
        Map<String, Object> model = new HashMap<>();
        if (datasetType.equals("breast_cancer")) {
            model.put("parameters_file", "global_models/breast_cancer_model.json");
        } else if (datasetType.equals("parkinsons")) {
            model.put("parameters_file", "global_models/parkinsons_model.pkl");
        } else if (datasetType.equals("reinopath")) {
            model.put("parameters_file", "global_models/reinopath_model.pkl");
        } else {
            model.put("parameters_file", "global_models/" + datasetType + "_model.json");
        }
        configData.put("model", model);
        
        // Client section
        Map<String, Object> client = new HashMap<>();
        client.put("cycles", 1);
        client.put("wait_time", 10);
        client.put("retry_interval", 10);
        configData.put("client", client);
        
        config.put("configData", configData);
        
        return config;
    }
}