package com.medishare.api.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import com.fasterxml.jackson.databind.ObjectMapper;

@Service
public class ServerService {
    
    @Value("${app.python.script-path}")
    private String pythonScriptPath;
    
    private Process serverProcess;
    
    private String currentDatasetType = null;
    
    public boolean startServerWithDefaultConfig() {
        try {
            // Kill any existing process
            stopServer();
            
            // Create or use a default server config
            String defaultConfigPath = "config/server_config.json";
            Path configFilePath = Paths.get(defaultConfigPath);
            
            // Check if default config exists, if not create it
            if (!Files.exists(configFilePath)) {
                createDefaultServerConfig(defaultConfigPath);
            }
            
            System.out.println("Starting server with default config: " + defaultConfigPath);
            
            return startServer(defaultConfigPath);
        } catch (Exception e) {
            System.err.println("Error starting server with default config: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    private void createDefaultServerConfig(String configPath) throws Exception {
        // Create a minimal server configuration that supports multiple dataset types
        Map<String, Object> serverConfig = new HashMap<>();
        
        // Server section
        Map<String, Object> serverSection = new HashMap<>();
        serverSection.put("host", "0.0.0.0");
        serverSection.put("port", 8080);
        serverSection.put("client_host", "127.0.0.1");
        serverSection.put("update_threshold", 1);
        serverSection.put("contribution_weight", 0.1);
        serverSection.put("backup_dir", "model_backups");
        serverConfig.put("server", serverSection);
        
        // Dataset types the server will support
        serverConfig.put("supported_datasets", Arrays.asList("breast_cancer", "parkinsons", "reinopath"));
        
        // Dataset-specific configuration sections
        Map<String, Object> datasetConfigs = new HashMap<>();
        
        // Breast cancer configuration
        Map<String, Object> breastCancerConfig = new HashMap<>();
        breastCancerConfig.put("parameters_file", "global_models/breast_cancer_model.json");
        breastCancerConfig.put("hidden_layers", Arrays.asList(64, 32));
        breastCancerConfig.put("task", "classification");
        datasetConfigs.put("breast_cancer", breastCancerConfig);
        
        // Parkinsons configuration
        Map<String, Object> parkinsonsConfig = new HashMap<>();
        parkinsonsConfig.put("parameters_file", "global_models/parkinsons_model.pkl");
        parkinsonsConfig.put("hidden_layers", Arrays.asList(128, 64, 32));
        parkinsonsConfig.put("task", "regression");
        datasetConfigs.put("parkinsons", parkinsonsConfig);
        
        // Reinopath configuration
        Map<String, Object> reinopathConfig = new HashMap<>();
        reinopathConfig.put("parameters_file", "global_models/reinopath_model.pkl");
        reinopathConfig.put("hidden_layers", Arrays.asList(128, 64, 32));
        reinopathConfig.put("task", "classification");
        datasetConfigs.put("reinopath", reinopathConfig);
        
        serverConfig.put("dataset_configs", datasetConfigs);
        
        // Create parent directories if they don't exist
        Files.createDirectories(Paths.get(configPath).getParent());
        
        // Write config to file
        ObjectMapper objectMapper = new ObjectMapper();
        Files.write(Paths.get(configPath), objectMapper.writeValueAsBytes(serverConfig));
        
        System.out.println("Created default server config at: " + configPath);
    }
    
    public boolean startServer(String configPath) {
        try {
            // Kill any existing process
            stopServer();
            
            System.out.println("Starting server with config: " + configPath);
            System.out.println("Python script path: " + pythonScriptPath);
            
            // Check if Python script exists
            if (!Files.exists(Paths.get(pythonScriptPath))) {
                System.err.println("Python script not found at: " + pythonScriptPath);
                return false;
            }
            
            // Try to extract dataset type from config path or content
            try {
                // Extract from path: config_breast_cancer.json
                String filename = Paths.get(configPath).getFileName().toString();
                if (filename.contains("breast_cancer")) {
                    currentDatasetType = "breast_cancer";
                } else if (filename.contains("parkinsons")) {
                    currentDatasetType = "parkinsons";
                } else if (filename.contains("reinopath")) {
                    currentDatasetType = "reinopath";
                } else {
                    // Try to read from the file
                    String configContent = Files.readString(Paths.get(configPath));
                    ObjectMapper mapper = new ObjectMapper();
                    Map<String, Object> config = mapper.readValue(configContent, Map.class);
                    
                    // Check if there's a dataset_type or datasetType field
                    if (config.containsKey("dataset_type")) {
                        currentDatasetType = (String) config.get("dataset_type");
                    } else if (config.containsKey("datasetType")) {
                        currentDatasetType = (String) config.get("datasetType");
                    }
                }
                
                // Write dataset type to server info file for future reference
                if (currentDatasetType != null) {
                    Map<String, Object> serverInfo = new HashMap<>();
                    serverInfo.put("datasetType", currentDatasetType);
                    serverInfo.put("configPath", configPath);
                    serverInfo.put("startTime", System.currentTimeMillis());
                    
                    ObjectMapper mapper = new ObjectMapper();
                    Files.writeString(Paths.get("server_info.json"), mapper.writeValueAsString(serverInfo));
                }
            } catch (Exception e) {
                System.err.println("Warning: Could not determine dataset type: " + e.getMessage());
                // Continue - this is not critical
            }
            
            // Execute your Python server script as a process
            ProcessBuilder processBuilder = new ProcessBuilder("python", pythonScriptPath, 
                    "--mode", "server", "--config", configPath);
            
            // Add environment variables if needed
            // processBuilder.environment().put("PYTHONPATH", "/path/to/python/modules");
            
            // Redirect error stream to output stream
            processBuilder.redirectErrorStream(true);
            
            // Start the process
            serverProcess = processBuilder.start();
            
            // Start a thread to read and log output
            new Thread(() -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(serverProcess.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println("Server: " + line);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
            
            // Wait a bit to see if it crashes immediately
            Thread.sleep(1000);
            
            if (!serverProcess.isAlive()) {
                int exitCode = serverProcess.exitValue();
                System.err.println("Server process exited immediately with code: " + exitCode);
                return false;
            }
            
            // Write PID file for later management
            Files.writeString(Paths.get("server.pid"), String.valueOf(serverProcess.pid()));
            System.out.println("Server started successfully with PID: " + serverProcess.pid());
            
            return true;
        } catch (Exception e) {
            System.err.println("Error starting server: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    public boolean stopServer() {
        try {
            // Try to kill by process reference first
            if (serverProcess != null && serverProcess.isAlive()) {
                serverProcess.destroy();
                
                // Clean up server info
                try {
                    Files.deleteIfExists(Paths.get("server_info.json"));
                } catch (Exception e) {
                    // Ignore
                }
                
                return true;
            }
            
            // Try to kill by PID file as fallback
            Path pidFile = Paths.get("server.pid");
            if (Files.exists(pidFile)) {
                String pid = Files.readString(pidFile);
                Runtime.getRuntime().exec("kill " + pid);
                Files.delete(pidFile);
                
                // Clean up server info
                try {
                    Files.deleteIfExists(Paths.get("server_info.json"));
                } catch (Exception e) {
                    // Ignore
                }
                
                return true;
            }
            
            return false;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    public String getServerStatus() {
        try {
            // Check if we have a process reference
            if (serverProcess != null && serverProcess.isAlive()) {
                return "running";
            }
            
            // Check PID file as fallback
            Path pidFile = Paths.get("server.pid");
            if (Files.exists(pidFile)) {
                String pid = Files.readString(pidFile);
                Process checkProcess = Runtime.getRuntime().exec("ps -p " + pid);
                int exitCode = checkProcess.waitFor();
                
                return exitCode == 0 ? "running" : "stopped";
            }
            
            return "stopped";
        } catch (Exception e) {
            e.printStackTrace();
            return "unknown";
        }
    }
    
    /**
     * Check if the server is running for a specific dataset type
     * @param datasetType The dataset type to check
     * @return True if the server is running for this dataset type
     */
    public boolean isRunningForDatasetType(String datasetType) {
        // Get current server status
        String status = getServerStatus();
        
        // If the server is not running, it's definitely not running for this dataset type
        if (!"running".equals(status)) {
            return false;
        }
        
        try {
            // Check the current dataset type
            Path serverInfoPath = Paths.get("server_info.json");
            if (Files.exists(serverInfoPath)) {
                String serverInfo = Files.readString(serverInfoPath);
                ObjectMapper mapper = new ObjectMapper();
                Map<String, Object> serverData = mapper.readValue(serverInfo, Map.class);
                String currentDatasetType = (String) serverData.get("datasetType");
                
                return datasetType.equals(currentDatasetType);
            } else if (currentDatasetType != null) {
                // Use the in-memory dataset type if available
                return datasetType.equals(currentDatasetType);
            }
            
            // Cannot determine the dataset type, assume it's not the right one
            return false;
        } catch (Exception e) {
            System.err.println("Error checking server dataset type: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Get the current dataset type the server is running with
     * @return The current dataset type, or null if not running or unknown
     */
    public String getCurrentDatasetType() {
        // If server is not running, return null
        if (!"running".equals(getServerStatus())) {
            return null;
        }
        
        try {
            // Check the server info file first
            Path serverInfoPath = Paths.get("server_info.json");
            if (Files.exists(serverInfoPath)) {
                String serverInfo = Files.readString(serverInfoPath);
                ObjectMapper mapper = new ObjectMapper();
                Map<String, Object> serverData = mapper.readValue(serverInfo, Map.class);
                return (String) serverData.get("datasetType");
            } else if (currentDatasetType != null) {
                // Fall back to in-memory value
                return currentDatasetType;
            }
            
            return null;
        } catch (Exception e) {
            System.err.println("Error getting current dataset type: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Initiate a new federated learning training round
     * @param datasetType The dataset type to start a training round for
     * @return True if the training round was successfully initiated
     */
    public boolean initiateTrainingRound(String datasetType) {
        try {
            // Method 1: Send a direct command to the server process
            // This assumes the server has an API or mechanism to accept commands
            
            // Create a signal file that the Python script can monitor
            Path signalPath = Paths.get("training_round_" + datasetType + ".signal");
            Files.writeString(signalPath, String.valueOf(System.currentTimeMillis()));
            
            System.out.println("Created training round signal file: " + signalPath);
            
            // Method 2: Send a request to a Python server endpoint
            // This is an alternative approach if your Python server exposes an HTTP API
            try {
                String serverUrl = "http://localhost:8080/start_round";
                HttpClient client = HttpClient.newHttpClient();
                HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(serverUrl))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString("{\"datasetType\":\"" + datasetType + "\"}"))
                    .build();
                
                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                
                // Check if the request was successful
                if (response.statusCode() == 200) {
                    System.out.println("Successfully sent training round initiation request to server");
                    return true;
                } else {
                    System.out.println("Server responded with status " + response.statusCode() + ": " + response.body());
                    // Fall back to signal file approach
                    return Files.exists(signalPath);
                }
            } catch (Exception e) {
                System.out.println("HTTP request to server failed: " + e.getMessage());
                System.out.println("Falling back to signal file approach");
                // Fall back to signal file approach
                return Files.exists(signalPath);
            }
        } catch (Exception e) {
            System.err.println("Error initiating training round: " + e.getMessage());
            return false;
        }
    }
}