package com.medishare.api.controller;

import com.medishare.api.dto.ModelDTO;
import com.medishare.api.model.Model;
import com.medishare.api.model.TrainingJob;
import com.medishare.api.service.ModelService;
import com.medishare.api.service.ModelTrainingService;
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
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/training")
public class ModelTrainingController {
    
    @Value("${app.dataset-storage.temp-location:temp_datasets}")
    private String tempDatasetStorageLocation;
    
    private final ModelTrainingService trainingService;
    private final ModelService modelService;
    
    @Autowired
    public ModelTrainingController(ModelTrainingService trainingService, ModelService modelService) {
        this.trainingService = trainingService;
        this.modelService = modelService;
    }
    
    @PostMapping("/start")
    public ResponseEntity<?> startModelTraining(
            @RequestParam("file") MultipartFile file,
            @RequestParam("datasetType") String datasetType,
            @RequestParam("modelName") String modelName,
            @RequestParam("description") String description,
            @RequestParam(value = "epochs", defaultValue = "10") int epochs,
            @RequestParam(value = "batchSize", defaultValue = "32") int batchSize,
            @RequestParam(value = "learningRate", defaultValue = "0.001") double learningRate) {
        
        try {
            // Create temporary directory for dataset if it doesn't exist
            Path datasetDir = Paths.get(tempDatasetStorageLocation);
            if (!Files.exists(datasetDir)) {
                Files.createDirectories(datasetDir);
            }
            
            // Generate unique filename
            String originalFilename = file.getOriginalFilename();
            String fileExtension = "";
            if (originalFilename != null && originalFilename.contains(".")) {
                fileExtension = originalFilename.substring(originalFilename.lastIndexOf("."));
            }
            String uniqueFilename = "dataset_" + UUID.randomUUID().toString() + fileExtension;
            
            // Define the full path where the file will be temporarily stored
            Path filePath = datasetDir.resolve(uniqueFilename);
            
            // Save the file
            Files.copy(file.getInputStream(), filePath);
            System.out.println("Dataset saved to: " + filePath.toString());
            
            // Generate a unique job ID
            String jobId = UUID.randomUUID().toString();
            
            // Create training configuration
            Map<String, Object> trainingConfig = new HashMap<>();
            trainingConfig.put("datasetType", datasetType);
            trainingConfig.put("modelName", modelName);
            trainingConfig.put("description", description);
            trainingConfig.put("epochs", epochs);
            trainingConfig.put("batchSize", batchSize);
            trainingConfig.put("learningRate", learningRate);
            
            // Add dataset configuration
            Map<String, Object> datasetConfig = new HashMap<>();
            datasetConfig.put("path", filePath.toString());
            
            // Select appropriate target column based on dataset type
            if (datasetType.equalsIgnoreCase("breast_cancer")) {
                datasetConfig.put("target_column", "diagnosis");
            } else if (datasetType.equalsIgnoreCase("parkinsons")) {
                datasetConfig.put("target_column", "UPDRS");
            } else if (datasetType.equalsIgnoreCase("reinopath")) {
                datasetConfig.put("target_column", "class");
            } else {
                datasetConfig.put("target_column", "target");
            }
            
            trainingConfig.put("dataset", datasetConfig);
            
            // Start training asynchronously
            trainingService.startTraining(jobId, datasetType, filePath.toString(), trainingConfig);
            
            // Return success response with job ID
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("jobId", jobId);
            response.put("message", "Model training started successfully");
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.badRequest().body(
                Map.of(
                    "success", false,
                    "error", "Failed to start model training: " + e.getMessage()
                )
            );
        }
    }
    
    @GetMapping("/status/{jobId}")
    public ResponseEntity<?> getTrainingStatus(@PathVariable String jobId) {
        try {
            TrainingJob job = trainingService.getTrainingStatus(jobId);
            
            if (job == null) {
                return ResponseEntity.notFound().build();
            }
            
            // Create response with job status
            Map<String, Object> response = new HashMap<>();
            response.put("jobId", job.getJobId());
            response.put("status", job.getStatus());
            response.put("progress", job.getProgress());
            response.put("datasetType", job.getDatasetType());
            response.put("createdAt", job.getCreatedAt());
            response.put("updatedAt", job.getUpdatedAt());
            
            if (job.getModelId() != null) {
                response.put("modelId", job.getModelId());
                
                // Add model information if available
                try {
                    Model model = modelService.getModelById(job.getModelId()).orElse(null);
                    if (model != null) {
                        Map<String, Object> modelInfo = new HashMap<>();
                        modelInfo.put("id", model.getId());
                        modelInfo.put("name", model.getName());
                        modelInfo.put("description", model.getDescription());
                        modelInfo.put("createdAt", model.getCreatedAt());
                        modelInfo.put("active", model.isActive());
                        response.put("model", modelInfo);
                    }
                } catch (Exception e) {
                    // Ignore error getting model info
                }
            }
            
            if (job.getErrorMessage() != null) {
                response.put("error", job.getErrorMessage());
            }
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                Map.of(
                    "success", false,
                    "error", "Failed to get training status: " + e.getMessage()
                )
            );
        }
    }
    
    @PostMapping("/cancel/{jobId}")
    public ResponseEntity<?> cancelTraining(@PathVariable String jobId) {
        try {
            boolean cancelled = trainingService.cancelTraining(jobId);
            
            if (cancelled) {
                return ResponseEntity.ok(Map.of(
                    "success", true,
                    "message", "Training job cancelled successfully"
                ));
            } else {
                return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "error", "Training job not found or already completed"
                ));
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                Map.of(
                    "success", false,
                    "error", "Failed to cancel training: " + e.getMessage()
                )
            );
        }
    }
    
    @GetMapping("/jobs")
    public ResponseEntity<?> getAllTrainingJobs() {
        try {
            List<Map<String, Object>> jobs = trainingService.getAllTrainingJobs().stream()
                .map(job -> {
                    Map<String, Object> jobInfo = new HashMap<>();
                    jobInfo.put("jobId", job.getJobId());
                    jobInfo.put("status", job.getStatus());
                    jobInfo.put("progress", job.getProgress());
                    jobInfo.put("datasetType", job.getDatasetType());
                    jobInfo.put("createdAt", job.getCreatedAt());
                    jobInfo.put("updatedAt", job.getUpdatedAt());
                    
                    if (job.getModelId() != null) {
                        jobInfo.put("modelId", job.getModelId());
                    }
                    
                    if (job.getErrorMessage() != null) {
                        jobInfo.put("error", job.getErrorMessage());
                    }
                    
                    return jobInfo;
                })
                .collect(Collectors.toList());
            
            return ResponseEntity.ok(jobs);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                Map.of(
                    "success", false,
                    "error", "Failed to retrieve training jobs: " + e.getMessage()
                )
            );
        }
    }
    
    @GetMapping("/active")
    public ResponseEntity<?> getActiveTrainingJobs() {
        try {
            List<Map<String, Object>> jobs = trainingService.getActiveTrainingJobs().stream()
                .map(job -> {
                    Map<String, Object> jobInfo = new HashMap<>();
                    jobInfo.put("jobId", job.getJobId());
                    jobInfo.put("status", job.getStatus());
                    jobInfo.put("progress", job.getProgress());
                    jobInfo.put("datasetType", job.getDatasetType());
                    jobInfo.put("createdAt", job.getCreatedAt());
                    
                    return jobInfo;
                })
                .collect(Collectors.toList());
            
            return ResponseEntity.ok(jobs);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                Map.of(
                    "success", false,
                    "error", "Failed to retrieve active training jobs: " + e.getMessage()
                )
            );
        }
    }
}