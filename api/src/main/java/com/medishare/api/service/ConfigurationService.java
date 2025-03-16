package com.medishare.api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.medishare.api.dto.ConfigurationDTO;
import com.medishare.api.model.Configuration;
import com.medishare.api.repository.ConfigurationRepository;
import com.medishare.api.util.FileSystemUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
public class ConfigurationService {
    private final ConfigurationRepository configRepository;
    private final ObjectMapper objectMapper;
    private final FileSystemUtil fileSystemUtil;
    
    @Autowired
    public ConfigurationService(ConfigurationRepository configRepository, 
                               ObjectMapper objectMapper,
                               FileSystemUtil fileSystemUtil) {
        this.configRepository = configRepository;
        this.objectMapper = objectMapper;
        this.fileSystemUtil = fileSystemUtil;
    }
    
    public Configuration createConfiguration(ConfigurationDTO configDTO) throws Exception {
        Configuration config = new Configuration();
        config.setDatasetType(configDTO.getDatasetType());
        config.setConfigName(configDTO.getConfigName());
        config.setConfigJson(objectMapper.writeValueAsString(configDTO.getConfigData()));
        config.setCreatedAt(LocalDateTime.now());
        config.setUpdatedAt(LocalDateTime.now());
        config.setActive(true);
        
        // Write to file system as well for Python code to read
        fileSystemUtil.writeConfigToFile(configDTO.getDatasetType(), configDTO.getConfigData());
        
        return configRepository.save(config);
    }
    
    public Optional<Configuration> getConfigurationByDatasetType(String datasetType) {
        return configRepository.findByDatasetType(datasetType);
    }
    
    public List<Configuration> getAllConfigurations() {
        return configRepository.findAll();
    }
    
    public Configuration updateConfiguration(String datasetType, ConfigurationDTO configDTO) throws Exception {
        Optional<Configuration> existingConfig = configRepository.findByDatasetType(datasetType);
        
        if (existingConfig.isPresent()) {
            Configuration config = existingConfig.get();
            config.setConfigName(configDTO.getConfigName());
            config.setConfigJson(objectMapper.writeValueAsString(configDTO.getConfigData()));
            config.setUpdatedAt(LocalDateTime.now());
            
            // Update file system config
            fileSystemUtil.writeConfigToFile(datasetType, configDTO.getConfigData());
            
            return configRepository.save(config);
        } else {
            throw new RuntimeException("Configuration not found for dataset type: " + datasetType);
        }
    }

    public int deleteAllConfigurations() {
        // Get the list of configurations before deleting
        List<Configuration> configurations = configRepository.findAll();
        
        // Delete configurations from repository
        configRepository.deleteAll();
        
        // Remove corresponding config files from file system
        for (Configuration config : configurations) {
            try {
                fileSystemUtil.deleteConfigFile(config.getDatasetType());
            } catch (Exception e) {
                // Log the error but continue processing other files
                System.err.println("Error deleting config file for " + config.getDatasetType() + ": " + e.getMessage());
            }
        }
        
        return configurations.size();
    }
    
    public boolean deleteConfigurationByDatasetType(String datasetType) {
        Optional<Configuration> existingConfig = configRepository.findByDatasetType(datasetType);
        
        if (existingConfig.isPresent()) {
            // Delete from repository
            configRepository.delete(existingConfig.get());
            
            // Delete corresponding config file from file system
            try {
                fileSystemUtil.deleteConfigFile(datasetType);
            } catch (Exception e) {
                System.err.println("Error deleting config file for " + datasetType + ": " + e.getMessage());
            }
            
            return true;
        }
        
        return false;
    }
}