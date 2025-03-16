package com.medishare.api.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

@Component
public class FileSystemUtil {
    private final Path configStorageLocation;
    private final Path modelStorageLocation;
    private final ObjectMapper objectMapper;
    
    @Autowired
    public FileSystemUtil(
            @Value("${app.config-storage.location}") String configStorageLocation,
            @Value("${app.model-storage.location}") String modelStorageLocation,
            ObjectMapper objectMapper) {
        this.configStorageLocation = Paths.get(configStorageLocation);
        this.modelStorageLocation = Paths.get(modelStorageLocation);
        this.objectMapper = objectMapper;
        
        // Create directories if they don't exist
        try {
            Files.createDirectories(this.configStorageLocation);
            Files.createDirectories(this.modelStorageLocation);
        } catch (Exception e) {
            throw new RuntimeException("Could not create storage directories", e);
        }
    }
    
    public void writeConfigToFile(String datasetType, Map<String, Object> configData) throws Exception {
        Path configFile = configStorageLocation.resolve(datasetType + "_config.json");
        String jsonContent = objectMapper.writeValueAsString(configData);
        Files.writeString(configFile, jsonContent);
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
}