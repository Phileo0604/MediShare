package com.medishare.api.controller;

import com.medishare.api.dto.ModelDTO;
import com.medishare.api.model.Model;
import com.medishare.api.service.ModelService;
import com.medishare.api.service.ServerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/models")
public class ModelController {
    private final ModelService modelService;
    private final ServerService serverService;
    
    @Autowired
    public ModelController(ModelService modelService, ServerService serverService) {
        this.modelService = modelService;
        this.serverService = serverService;
    }
    
    @GetMapping
    public ResponseEntity<List<Model>> getAllModels() {
        List<Model> models = modelService.getAllModels();
        return ResponseEntity.ok(models);
    }
    
    @GetMapping("/{datasetType}")
    public ResponseEntity<List<Model>> getModelsByDatasetType(@PathVariable String datasetType) {
        List<Model> models = modelService.getModelsByDatasetType(datasetType);
        return ResponseEntity.ok(models);
    }
    
    @GetMapping("/byId/{id}")
    public ResponseEntity<?> getModelById(@PathVariable Long id) {
        Optional<Model> model = modelService.getModelById(id);
        if (model.isPresent()) {
            return ResponseEntity.ok(model.get());
        } else {
            return ResponseEntity.notFound().build();
        }
    }
    
    @GetMapping("/{datasetType}/active")
    public ResponseEntity<?> getActiveModel(@PathVariable String datasetType) {
        Optional<Model> model = modelService.getActiveModel(datasetType);
        return model.map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/{datasetType}/download")
    public ResponseEntity<Resource> downloadModel(@PathVariable String datasetType) {
        try {
            Resource resource = modelService.loadModelAsResource(datasetType);
            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .header(HttpHeaders.CONTENT_DISPOSITION, 
                            "attachment; filename=\"" + resource.getFilename() + "\"")
                    .body(resource);
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }
    
    @DeleteMapping("/all/{datasetType}")
        public ResponseEntity<?> deleteAllModelsByDatasetType(@PathVariable String datasetType) {
            try {
                int deletedCount = modelService.deleteAllModelsByDatasetType(datasetType);
                return ResponseEntity.ok(Map.of(
                    "success", true,
                    "count", deletedCount,
                    "message", "Deleted " + deletedCount + " models for " + datasetType
                ));
            } catch (Exception e) {
                return ResponseEntity.badRequest().body("Error deleting models: " + e.getMessage());
            }
        }

        @DeleteMapping("/all")
        public ResponseEntity<?> deleteAllModels() {
            try {
                int deletedCount = modelService.deleteAllModels();
                return ResponseEntity.ok(Map.of(
                    "success", true,
                    "count", deletedCount,
                    "message", "Deleted " + deletedCount + " models"
                ));
            } catch (Exception e) {
                return ResponseEntity.badRequest().body("Error deleting models: " + e.getMessage());
            }
        }

    @PostMapping("/register")
    public ResponseEntity<?> registerModel(@RequestBody ModelDTO modelDTO, @RequestParam String filePath) {
        try {
            Model model = modelService.registerModel(modelDTO, filePath);
            return ResponseEntity.ok(model);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error registering model: " + e.getMessage());
        }
    }
    
    /**
     * Activate a model as the global model for a dataset type
     */
    @PostMapping("/activate")
    public ResponseEntity<?> activateModel(@RequestBody Map<String, Object> requestData) {
        try {
            Long modelId = Long.valueOf(requestData.get("modelId").toString());
            String datasetType = (String) requestData.get("datasetType");
            
            Optional<Model> modelOpt = modelService.getModelById(modelId);
            if (modelOpt.isPresent()) {
                Model model = modelOpt.get();
                
                // Ensure the model is for the right dataset type
                if (!model.getDatasetType().equals(datasetType)) {
                    return ResponseEntity.badRequest().body(
                        Map.of(
                            "success", false,
                            "error", "Model dataset type does not match requested dataset type"
                        )
                    );
                }
                
                // Set this model as active (the service will handle deactivating others)
                Model activatedModel = modelService.setModelActive(model);
                
                return ResponseEntity.ok(
                    Map.of(
                        "success", true,
                        "model", activatedModel
                    )
                );
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                Map.of(
                    "success", false,
                    "error", "Error activating model: " + e.getMessage()
                )
            );
        }
    }

    /**
     * Get performance metrics for a model
     * This is a placeholder implementation - you'll need to extend it with actual metrics
     */
    @GetMapping("/{id}/metrics")
    public ResponseEntity<?> getModelMetrics(@PathVariable Long id) {
        try {
            Optional<Model> modelOpt = modelService.getModelById(id);
            if (modelOpt.isPresent()) {
                // This is where you would calculate or retrieve actual metrics
                // For now, we'll just return placeholder data
                Map<String, Object> metrics = new HashMap<>();
                metrics.put("accuracy", 0.875);
                metrics.put("f1Score", 0.82);
                metrics.put("precision", 0.89);
                metrics.put("recall", 0.86);
                
                return ResponseEntity.ok(metrics);
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error getting model metrics: " + e.getMessage());
        }
    }

    /**
     * Start a new federated learning training round
     */
    @PostMapping("/training-round")
    public ResponseEntity<?> startTrainingRound(@RequestBody Map<String, Object> requestData) {
        try {
            String datasetType = (String) requestData.get("datasetType");
            
            // 1. Check if server is running for this dataset type
            if (!serverService.isRunningForDatasetType(datasetType)) {
                return ResponseEntity.badRequest().body(
                    Map.of(
                        "success", false,
                        "error", "Server not running for dataset type: " + datasetType
                    )
                );
            }
            
            // 2. Initiate a new training round
            boolean initiated = serverService.initiateTrainingRound(datasetType);
            
            if (initiated) {
                return ResponseEntity.ok(
                    Map.of(
                        "success", true,
                        "message", "Training round initiated successfully",
                        "timestamp", System.currentTimeMillis()
                    )
                );
            } else {
                return ResponseEntity.internalServerError().body(
                    Map.of(
                        "success", false,
                        "error", "Failed to initiate training round"
                    )
                );
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                Map.of(
                    "success", false,
                    "error", "Error starting training round: " + e.getMessage()
                )
            );
        }
    }
}