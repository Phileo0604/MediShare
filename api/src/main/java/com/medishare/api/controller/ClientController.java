package com.medishare.api.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.medishare.api.model.Configuration;
import com.medishare.api.model.Model;
import com.medishare.api.repository.ConfigurationRepository;
import com.medishare.api.repository.ModelRepository;
import com.medishare.api.service.ClientService;
import com.medishare.api.util.ConfigurationUtils;
import com.medishare.api.util.FileSystemUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/client")
public class ClientController {
    private final ClientService clientService;
    private final FileSystemUtil fileSystemUtil;
    private final ConfigurationRepository configRepository;
    private final ModelRepository modelRepository;
    private final ObjectMapper objectMapper;
    
    @Autowired
    private ConfigurationUtils configUtils;

    @Autowired
    public ClientController(
            ClientService clientService, 
            FileSystemUtil fileSystemUtil,
            ConfigurationRepository configRepository,
            ModelRepository modelRepository,
            ObjectMapper objectMapper) {
        this.clientService = clientService;
        this.fileSystemUtil = fileSystemUtil;
        this.configRepository = configRepository;
        this.modelRepository = modelRepository;
        this.objectMapper = objectMapper;
    }
    
    @PostMapping("/start")
    public ResponseEntity<?> startClient(
            @RequestParam String datasetType,
            @RequestParam(required = false) Long configId,
            @RequestParam(defaultValue = "1") int cycles,
            @RequestParam(defaultValue = "127.0.0.1") String serverHost,
            @RequestParam(required = false) String clientId) { // Added optional clientId parameter
        try {
            String configPath;
            String modelFilePath = null;
            
            // If configId is provided, use that specific configuration
            if (configId != null) {
                Optional<Configuration> configOpt = configRepository.findById(configId);
                if (configOpt.isPresent()) {
                    Configuration config = configOpt.get();
                    
                    // Verify the dataset type matches
                    if (!config.getDatasetType().equals(datasetType)) {
                        return ResponseEntity.badRequest().body(
                            "Configuration dataset type does not match the requested dataset type"
                        );
                    }
                    
                    // Get the config JSON from the database
                    String configJson = config.getConfigJson();
                    
                    // Try to extract model ID from the configuration
                    try {
                        Map<String, Object> configMap = objectMapper.readValue(configJson, Map.class);
                        Map<String, Object> modelConfig = (Map<String, Object>) configMap.get("model");
                        
                        if (modelConfig != null && modelConfig.containsKey("modelId")) {
                            // Get model by ID
                            Long modelId = Long.valueOf(modelConfig.get("modelId").toString());
                            Optional<Model> modelOpt = modelRepository.findById(modelId);
                            
                            if (modelOpt.isPresent()) {
                                // Use the model file path
                                modelFilePath = modelOpt.get().getFilePath();
                            } else {
                                return ResponseEntity.badRequest().body("Model not found: " + modelId);
                            }
                        }
                    } catch (Exception e) {
                        return ResponseEntity.badRequest().body(
                            "Failed to parse configuration JSON: " + e.getMessage()
                        );
                    }
                    
                    // Write the config to a temporary file
                    String tempConfigPath = fileSystemUtil.writeTempConfigFile(
                        config.getDatasetType(), 
                        config.getConfigJson()
                    );
                    configPath = tempConfigPath;
                } else {
                    return ResponseEntity.badRequest().body("Configuration not found: " + configId);
                }
            } else {
                // Use default config path
                configPath = fileSystemUtil.getConfigFilePath(datasetType);
                
                // Check if the configuration file exists
                if (!fileSystemUtil.checkFileExists(configPath)) {
                    return ResponseEntity.badRequest().body(
                        "Configuration file not found for dataset type: " + datasetType
                    );
                }
            }
            
            // Generate a client ID if not provided
            String finalClientId = clientId;
            if (finalClientId == null || finalClientId.isEmpty()) {
                finalClientId = "client_" + System.currentTimeMillis();
            }
            
            // Start the client
            boolean started = clientService.startClient(
                modelFilePath, // This can be null if we're using training mode
                configPath, 
                datasetType, 
                finalClientId, 
                serverHost, 
                cycles
            );
            
            if (started) {
                Map<String, Object> response = new HashMap<>();
                response.put("status", "started");
                response.put("clientId", finalClientId);
                response.put("datasetType", datasetType);
                response.put("cycles", cycles);
                response.put("serverHost", serverHost);
                response.put("startTime", System.currentTimeMillis());
                if (modelFilePath != null) {
                    response.put("usingModelParameters", true);
                }
                
                return ResponseEntity.ok(response);
            } else {
                return ResponseEntity.internalServerError().body("Failed to start client");
            }
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error starting client: " + e.getMessage());
        }
    }
    
    @PostMapping("/start-with-parameters")
    public ResponseEntity<?> startClientWithParameters(
            @RequestParam String datasetType,
            @RequestParam Long modelId,
            @RequestParam(defaultValue = "127.0.0.1") String serverHost,
            @RequestParam(required = false) String clientId) { // Added optional clientId parameter
        try {
            // Get the model from the database
            Optional<Model> modelOpt = modelRepository.findById(modelId);
            if (!modelOpt.isPresent()) {
                return ResponseEntity.badRequest().body("Model not found: " + modelId);
            }
            
            Model model = modelOpt.get();
            
            // Verify the dataset type matches
            if (!model.getDatasetType().equals(datasetType)) {
                return ResponseEntity.badRequest().body(
                    "Model dataset type does not match the requested dataset type"
                );
            }
            
            // Get the model file path
            String modelFilePath = model.getFilePath();
            
            // Generate a client ID if not provided
            String finalClientId = clientId;
            if (finalClientId == null || finalClientId.isEmpty()) {
                finalClientId = "client_" + System.currentTimeMillis();
            }
            
            // We need a config path as well, using the default one for the dataset type
            String configPath = fileSystemUtil.getConfigFilePath(datasetType);
            if (!fileSystemUtil.checkFileExists(configPath)) {
                // Try to create a minimal config
                Map<String, Object> minimalConfig = createMinimalConfig(datasetType);
                fileSystemUtil.writeConfigToFile(datasetType, minimalConfig);
                configPath = fileSystemUtil.getConfigFilePath(datasetType);
            }
            
            // Start the client with parameters
            boolean started = clientService.startClient(
                modelFilePath, 
                configPath,
                datasetType, 
                finalClientId, 
                serverHost, 
                1 // Default to 1 cycle when using parameters
            );
            
            if (started) {
                Map<String, Object> response = new HashMap<>();
                response.put("status", "started");
                response.put("clientId", finalClientId);
                response.put("datasetType", datasetType);
                response.put("modelId", modelId);
                response.put("serverHost", serverHost);
                response.put("startTime", System.currentTimeMillis());
                response.put("usingModelParameters", true);
                
                return ResponseEntity.ok(response);
            } else {
                return ResponseEntity.internalServerError().body("Failed to start client");
            }
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error starting client: " + e.getMessage());
        }
    }
    
    @GetMapping("/status/{clientId}")
    public ResponseEntity<?> getClientStatus(@PathVariable String clientId) {
        try {
            String status = clientService.getClientStatus(clientId);
            Map<String, String> response = new HashMap<>();
            response.put("clientId", clientId);
            response.put("status", status);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error getting client status: " + e.getMessage());
        }
    }
    
    @PostMapping("/stop/{clientId}")
    public ResponseEntity<?> stopClient(@PathVariable String clientId) {
        try {
            boolean stopped = clientService.stopClient(clientId);
            if (stopped) {
                return ResponseEntity.ok("Client stopped successfully");
            } else {
                return ResponseEntity.internalServerError().body("Failed to stop client");
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error stopping client: " + e.getMessage());
        }
    }
    
    @GetMapping("/history/{datasetType}")
    public ResponseEntity<?> getClientHistory(@PathVariable String datasetType) {
        try {
            // Get client history for the specified dataset type
            List<Map<String, Object>> clientHistory = clientService.getClientHistory(datasetType);
            return ResponseEntity.ok(clientHistory);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                "Error retrieving client history: " + e.getMessage()
            );
        }
    }
    
    /**
     * Delete a client history entry
     */
    @DeleteMapping("/history/{clientId}")
    public ResponseEntity<?> deleteClientHistory(@PathVariable String clientId) {
        try {
            // Try to stop the client first if it's running
            try {
                String status = clientService.getClientStatus(clientId);
                if (status.equals("running")) {
                    clientService.stopClient(clientId);
                }
            } catch (Exception e) {
                // Ignore errors if the client is not running or can't be stopped
                System.err.println("Error stopping client before deletion: " + e.getMessage());
            }
            
            // Now try to remove from history
            boolean removed = false;
            synchronized (clientService) {
                List<Map<String, Object>> allHistory = clientService.getAllClientHistory();
                int sizeBefore = allHistory.size();
                
                // Filter the specific entry out
                allHistory.removeIf(entry -> entry.get("clientId").equals(clientId));
                
                removed = allHistory.size() < sizeBefore;
            }
            
            if (removed) {
                return ResponseEntity.ok("Client history deleted successfully");
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error deleting client history: " + e.getMessage());
        }
    }
    
    /**
     * Refresh client statuses
     */
    @PostMapping("/refresh-status")
    public ResponseEntity<?> refreshClientStatuses(@RequestParam String datasetType) {
        try {
            List<String> activeClientIds = clientService.getActiveClientIds();
            
            // Update status for each potentially active client
            for (String clientId : activeClientIds) {
                try {
                    String status = clientService.getClientStatus(clientId);
                    // Status update is handled inside getClientStatus
                } catch (Exception e) {
                    // Log but continue with other clients
                    System.err.println("Error updating status for client " + clientId + ": " + e.getMessage());
                }
            }
            
            // Return updated client history
            List<Map<String, Object>> updatedHistory = clientService.getClientHistory(datasetType);
            return ResponseEntity.ok(updatedHistory);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error refreshing client statuses: " + e.getMessage());
        }
    }
    
    /**
     * Create a minimal configuration for the given dataset type
     * This is used when starting with parameters but no config exists
     */
    private Map<String, Object> createMinimalConfig(String datasetType) {
        return configUtils.createMinimalConfig(datasetType);
    }
}