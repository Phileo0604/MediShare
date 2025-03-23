package com.medishare.api.service;

import com.medishare.api.dto.ModelDTO;
import com.medishare.api.model.Model;
import com.medishare.api.repository.ModelRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

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
    
    /**
     * Set a model as the active model for its dataset type
     * 
     * @param model The model to set as active
     * @return The updated model
     */
    public Model setModelActive(Model model) {
        // First, deactivate all models for this dataset type
        List<Model> activeModels = modelRepository.findByDatasetTypeAndActive(model.getDatasetType(), true)
                .stream().toList();
        
        for (Model activeModel : activeModels) {
            // Skip if this is the model we're trying to activate
            if (activeModel.getId().equals(model.getId())) {
                continue;
            }
            activeModel.setActive(false);
            modelRepository.save(activeModel);
        }
        
        // Now activate the requested model
        model.setActive(true);
        return modelRepository.save(model);
    }

    // Add these methods to the ModelService class

// Add these methods to your ModelService class

/**
 * Delete all models
 * 
 * @return The number of models deleted
 */
@Transactional
public int deleteAllModels() {
    // Get the count of models before deleting
    long count = modelRepository.count();
    
    // Delete all models from repository
    modelRepository.deleteAll();
    
    // Return the count of deleted models
    return (int) count;
}

/**
 * Delete all models for a specific dataset type
 * 
 * @param datasetType The dataset type to delete models for
 * @return The number of models deleted
 */
@Transactional
public int deleteAllModelsByDatasetType(String datasetType) {
    // Get the list of models for this dataset type
    List<Model> models = modelRepository.findByDatasetType(datasetType);
    int count = models.size();
    
    // Delete all models for this dataset type
    modelRepository.deleteByDatasetType(datasetType);
    
    // Return the count of deleted models
    return count;
}

    /**
     * Activate a model by ID
     * 
     * @param modelId The ID of the model to activate
     * @param datasetType The dataset type for validation
     * @return The activated model
     * @throws RuntimeException if the model is not found or dataset type doesn't match
     */
    public Model activateModelById(Long modelId, String datasetType) {
        Optional<Model> modelOpt = modelRepository.findById(modelId);
        
        if (modelOpt.isPresent()) {
            Model model = modelOpt.get();
            
            // Verify the dataset type matches
            if (!model.getDatasetType().equals(datasetType)) {
                throw new RuntimeException("Model dataset type does not match the requested dataset type");
            }
            
            // Set the model as active
            return setModelActive(model);
        } else {
            throw new RuntimeException("Model not found with ID: " + modelId);
        }
    }
}