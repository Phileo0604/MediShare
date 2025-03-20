package com.medishare.api.controller;

import com.medishare.api.dto.ModelDTO;
import com.medishare.api.model.Model;
import com.medishare.api.service.ModelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/model")
public class ModelUploadController {
    
    @Value("${app.model-storage.location}")
    private String modelStorageLocation;
    
    private final ModelService modelService;
    
    @Autowired
    public ModelUploadController(ModelService modelService) {
        this.modelService = modelService;
    }
    
    @PostMapping("/upload")
    public ResponseEntity<?> uploadModelParameters(
            @RequestParam("file") MultipartFile file,
            @RequestParam("name") String name,
            @RequestParam("description") String description,
            @RequestParam("datasetType") String datasetType) {
        
        try {
            // Create directories if they don't exist
            Path storageDir = Paths.get(modelStorageLocation);
            if (!Files.exists(storageDir)) {
                Files.createDirectories(storageDir);
            }
            
            // Generate unique filename to prevent overwrites
            String originalFilename = file.getOriginalFilename();
            String fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
            String uniqueFilename = datasetType + "_" + UUID.randomUUID().toString() + fileExtension;
            
            // Define the full path where the file will be stored
            Path filePath = storageDir.resolve(uniqueFilename);
            
            // Save the file
            Files.copy(file.getInputStream(), filePath);
            
            // Create a ModelDTO with file format
            ModelDTO modelDTO = new ModelDTO();
            modelDTO.setName(name);
            modelDTO.setDescription(description);
            modelDTO.setDatasetType(datasetType);
            
            // Determine file format from extension
            String fileFormat = fileExtension.toLowerCase().substring(1); // Remove the dot
            
            // Register the model with file format
            Model savedModel = modelService.registerModel(modelDTO, filePath.toString(), fileFormat);
            
            // Return success response
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("model", savedModel);
            response.put("message", "Model parameters uploaded successfully");
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                Map.of(
                    "success", false,
                    "error", "Failed to upload model parameters: " + e.getMessage()
                )
            );
        }
    }
}