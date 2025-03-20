package com.medishare.api.controller;

import com.medishare.api.dto.ModelDTO;
import com.medishare.api.model.Model;
import com.medishare.api.service.ModelService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/models")
public class ModelController {
    private final ModelService modelService;
    
    @Autowired
    public ModelController(ModelService modelService) {
        this.modelService = modelService;
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
    
    @PostMapping("/register")
    public ResponseEntity<?> registerModel(@RequestBody ModelDTO modelDTO, @RequestParam String filePath) {
        try {
            Model model = modelService.registerModel(modelDTO, filePath);
            return ResponseEntity.ok(model);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error registering model: " + e.getMessage());
        }
    }
}