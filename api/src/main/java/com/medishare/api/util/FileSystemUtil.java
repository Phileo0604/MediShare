package com.medishare.api.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.UUID;

@Component
public class FileSystemUtil {
    private final Path configStorageLocation;
    private final Path modelStorageLocation;
    private final Path tempStorageLocation;
    private final ObjectMapper objectMapper;
    
    @Autowired
    public FileSystemUtil(
            @Value("${app.config-storage.location}") String configStorageLocation,
            @Value("${app.model-storage.location}") String modelStorageLocation,
            @Value("${app.temp-storage.location:temp}") String tempStorageLocation,
            ObjectMapper objectMapper) {
        this.configStorageLocation = Paths.get(configStorageLocation);
        this.modelStorageLocation = Paths.get(modelStorageLocation);
        this.tempStorageLocation = Paths.get(tempStorageLocation);
        this.objectMapper = objectMapper;
        
        // Create directories if they don't exist
        try {
            Files.createDirectories(this.configStorageLocation);
            Files.createDirectories(this.modelStorageLocation);
            Files.createDirectories(this.tempStorageLocation);
        } catch (Exception e) {
            throw new RuntimeException("Could not create storage directories", e);
        }
    }
    
    public void writeConfigToFile(String datasetType, Map<String, Object> configData) throws Exception {
        Path configFile = configStorageLocation.resolve(datasetType + "_config.json");
        String jsonContent = objectMapper.writeValueAsString(configData);
        Files.writeString(configFile, jsonContent);
    }
    
    /**
     * Write a temporary config file for a specific use (like starting a client with a specific config)
     * 
     * @param datasetType The dataset type
     * @param configJson The configuration JSON string
     * @return The path to the temporary file
     * @throws Exception If there's an error writing the file
     */
    // Add this method to FileSystemUtil.java
    public String writeTempConfigFile(String datasetType, String configJson) throws Exception {
        // Use the ConfigurationUtils to parse and normalize the config
        ConfigurationUtils configUtils = new ConfigurationUtils(objectMapper);
        Map<String, Object> normalizedConfig = configUtils.parseAndNormalize(configJson);
        
        // Create a unique filename for this temporary config
        String filename = datasetType + "_" + System.currentTimeMillis() + ".json";
        Path tempConfigPath = configStorageLocation.resolve(filename);
        
        // Convert the normalized config to the new format
        Map<String, Object> newFormatConfig = configUtils.toNewFormat(normalizedConfig);
        
        // Write the config to the file
        String jsonContent = objectMapper.writeValueAsString(newFormatConfig);
        Files.writeString(tempConfigPath, jsonContent);
        
        return tempConfigPath.toString();
    }
    
    public Map<String, Object> readConfigFromFile(String datasetType) throws Exception {
        Path configFile = configStorageLocation.resolve(datasetType + "_config.json");
        if (Files.exists(configFile)) {
            String content = Files.readString(configFile);
            return objectMapper.readValue(content, Map.class);
        }
        throw new RuntimeException("Config file not found: " + configFile);
    }
    
    public String getConfigFilePath(String datasetType) {
        return configStorageLocation.resolve(datasetType + "_config.json").toString();
    }
    
    public Path getModelFilePath(String datasetType, String modelType) {
        return modelStorageLocation.resolve(datasetType + "_" + modelType + ".json");
    }
    
    public boolean checkFileExists(String filePath) {
        return Files.exists(Paths.get(filePath));
    }
    
    public void deleteConfigFile(String datasetType) throws Exception {
        Path configFile = configStorageLocation.resolve(datasetType + "_config.json");
        
        if (Files.exists(configFile)) {
            Files.delete(configFile);
            System.out.println("Deleted configuration file: " + configFile);
        }
    }
    
    /**
     * Clean up temporary files older than a specified age
     * 
     * @param maxAgeHours Maximum age in hours
     * @return Number of files removed
     */
    public int cleanupTempFiles(int maxAgeHours) {
        try {
            final long maxAgeMillis = maxAgeHours * 60 * 60 * 1000L;
            final long cutoffTime = System.currentTimeMillis() - maxAgeMillis;
            
            return (int) Files.list(tempStorageLocation)
                .filter(path -> Files.isRegularFile(path))
                .filter(path -> {
                    try {
                        return Files.getLastModifiedTime(path).toMillis() < cutoffTime;
                    } catch (Exception e) {
                        return false;
                    }
                })
                .map(path -> {
                    try {
                        Files.delete(path);
                        return true;
                    } catch (Exception e) {
                        return false;
                    }
                })
                .filter(deleted -> deleted)
                .count();
        } catch (Exception e) {
            return 0;
        }
    }
}