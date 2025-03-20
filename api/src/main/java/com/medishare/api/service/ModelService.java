package com.medishare.api.service;

import com.medishare.api.dto.ModelDTO;
import com.medishare.api.model.Model;
import com.medishare.api.repository.ModelRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
public class ModelService {
    private final ModelRepository modelRepository;
    private final Path modelStorageLocation;
    
    @Autowired
    public ModelService(ModelRepository modelRepository, 
                        @Value("${app.model-storage.location}") String modelStorageLocation) {
        this.modelRepository = modelRepository;
        this.modelStorageLocation = Paths.get(modelStorageLocation);
    }
    
    public List<Model> getAllModels() {
        return modelRepository.findAll();
    }
    
    public Optional<Model> getModelById(Long id) {
        return modelRepository.findById(id);
    }
    
    public List<Model> getModelsByDatasetType(String datasetType) {
        return modelRepository.findByDatasetType(datasetType);
    }
    
    public Optional<Model> getActiveModel(String datasetType) {
        return modelRepository.findByDatasetTypeAndActive(datasetType, true);
    }
    
    public Resource loadModelAsResource(String datasetType) throws Exception {
        Optional<Model> modelOpt = getActiveModel(datasetType);
        if (modelOpt.isPresent()) {
            Path filePath = Paths.get(modelOpt.get().getFilePath());
            Resource resource = new UrlResource(filePath.toUri());
            if (resource.exists()) {
                return resource;
            }
        }
        throw new RuntimeException("Model not found for dataset type: " + datasetType);
    }
    
    // Original method that can still be used by other parts of the app
    public Model registerModel(ModelDTO modelDTO, String filePath) {
        return registerModel(modelDTO, filePath, "json"); // Default file format
    }
    
    // Method that accepts file format
    public Model registerModel(ModelDTO modelDTO, String filePath, String fileFormat) {
        // Deactivate previous active models for this dataset type
        List<Model> activeModels = modelRepository.findByDatasetTypeAndActive(modelDTO.getDatasetType(), true)
                .stream().toList();
        
        for (Model activeModel : activeModels) {
            activeModel.setActive(false);
            modelRepository.save(activeModel);
        }
        
        // Create new model
        Model model = new Model();
        model.setDatasetType(modelDTO.getDatasetType());
        model.setName(modelDTO.getName());
        model.setDescription(modelDTO.getDescription());
        model.setFilePath(filePath);
        model.setFileFormat(fileFormat); // Set the file format
        model.setCreatedAt(LocalDateTime.now());
        model.setActive(true);
        
        return modelRepository.save(model);
    }
}