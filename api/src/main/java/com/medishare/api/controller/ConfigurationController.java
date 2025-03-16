package com.medishare.api.controller;

import com.medishare.api.dto.ConfigurationDTO;
import com.medishare.api.model.Configuration;
import com.medishare.api.service.ConfigurationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/config")
public class ConfigurationController {
    private final ConfigurationService configService;
    
    @Autowired
    public ConfigurationController(ConfigurationService configService) {
        this.configService = configService;
    }
    
    @PostMapping("/create")
    public ResponseEntity<?> createConfiguration(@RequestBody ConfigurationDTO configDTO) {
        try {
            Configuration config = configService.createConfiguration(configDTO);
            return ResponseEntity.ok(config);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error creating configuration: " + e.getMessage());
        }
    }
    
    @GetMapping("/all")
    public ResponseEntity<List<Configuration>> getAllConfigurations() {
        List<Configuration> configs = configService.getAllConfigurations();
        return ResponseEntity.ok(configs);
    }
    
    @GetMapping("/{datasetType}")
    public ResponseEntity<?> getConfigurationByDatasetType(@PathVariable String datasetType) {
        Optional<Configuration> config = configService.getConfigurationByDatasetType(datasetType);
        return config.map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PutMapping("/{datasetType}")
    public ResponseEntity<?> updateConfiguration(
            @PathVariable String datasetType,
            @RequestBody ConfigurationDTO configDTO) {
        try {
            // Ensure the dataset type in the path matches the one in the DTO
            if (!datasetType.equals(configDTO.getDatasetType())) {
                return ResponseEntity.badRequest().body("Dataset type mismatch");
            }
            
            Configuration updatedConfig = configService.updateConfiguration(datasetType, configDTO);
            return ResponseEntity.ok(updatedConfig);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error updating configuration: " + e.getMessage());
        }
    }

    @DeleteMapping("/all")
    public ResponseEntity<?> deleteAllConfigurations() {
        try {
            int deletedCount = configService.deleteAllConfigurations();
            return ResponseEntity.ok("Deleted " + deletedCount + " configurations");
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error deleting configurations: " + e.getMessage());
        }
    }

    @DeleteMapping("/{datasetType}")
    public ResponseEntity<?> deleteConfigurationByDatasetType(@PathVariable String datasetType) {
        try {
            boolean deleted = configService.deleteConfigurationByDatasetType(datasetType);
            if (deleted) {
                return ResponseEntity.ok("Configuration for " + datasetType + " deleted successfully");
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error deleting configuration: " + e.getMessage());
        }
    }
}