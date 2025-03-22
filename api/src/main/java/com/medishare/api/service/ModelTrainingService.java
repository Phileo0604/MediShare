package com.medishare.api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.medishare.api.dto.ModelDTO;
import com.medishare.api.model.Model;
import com.medishare.api.model.TrainingJob;
import com.medishare.api.repository.ModelRepository;
import com.medishare.api.repository.TrainingJobRepository;
import com.medishare.api.util.FileSystemUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Service
public class ModelTrainingService {
    
    @Value("${app.python.train-script-path:train_model.py}")
    private String pythonTrainScriptPath;
    
    @Value("${app.model-storage.location:global_models}")
    private String modelStorageLocation;
    
    @Value("${app.temp-storage.location:temp}")
    private String tempStorageLocation;
    
    private final TrainingJobRepository trainingJobRepository;
    private final ModelRepository modelRepository;
    private final ModelService modelService;
    private final FileSystemUtil fileSystemUtil;
    private final ObjectMapper objectMapper;
    
    // Store running training processes by job ID
    private final Map<String, Process> trainingProcesses = new ConcurrentHashMap<>();
    
    @Autowired
    public ModelTrainingService(
        TrainingJobRepository trainingJobRepository,
        ModelRepository modelRepository,
        ModelService modelService,
        FileSystemUtil fileSystemUtil,
        ObjectMapper objectMapper
    ) {
        this.trainingJobRepository = trainingJobRepository;
        this.modelRepository = modelRepository;
        this.modelService = modelService;
        this.fileSystemUtil = fileSystemUtil;
        this.objectMapper = objectMapper;
    }
    
    /**
     * Start a training job asynchronously
     */
    @Async
    public void startTraining(String jobId, String datasetType, String datasetPath, Map<String, Object> config) {
        // Create and save training job record
        TrainingJob job = new TrainingJob();
        job.setJobId(jobId);
        job.setDatasetType(datasetType);
        job.setStatus("PENDING");
        job.setProgress(0);
        job.setDatasetPath(datasetPath);
        job.setCreatedAt(LocalDateTime.now());
        
        // Store config for reference
        try {
            job.setConfigJson(objectMapper.writeValueAsString(config));
        } catch (Exception e) {
            job.setConfigJson("{}");
        }
        
        trainingJobRepository.save(job);
        
        try {
            // Create a temporary config file for the training script
            Path tempConfigDir = Paths.get(tempStorageLocation);
            if (!Files.exists(tempConfigDir)) {
                Files.createDirectories(tempConfigDir);
            }
            
            String tempConfigPath = Paths.get(tempStorageLocation, "config_" + jobId + ".json").toString();
            try {
                // Write config to a temporary file
                Files.write(Paths.get(tempConfigPath), objectMapper.writeValueAsBytes(config));
                System.out.println("Created temporary config file: " + tempConfigPath);
            } catch (Exception e) {
                System.err.println("Error creating config file: " + e.getMessage());
                job.setStatusWithTimestamp("FAILED");
                job.setErrorMessage("Failed to create configuration file: " + e.getMessage());
                trainingJobRepository.save(job);
                return;
            }
            
            // Build command with appropriate arguments for Python training script
            java.util.List<String> command = new java.util.ArrayList<>();
            command.add("python");
            command.add(pythonTrainScriptPath);
            command.add("--config"); 
            command.add(tempConfigPath);
            command.add("--dataset-type");
            command.add(datasetType);
            command.add("--dataset-path");
            command.add(datasetPath);
            
            // Add optional parameters if specified
            if (config.containsKey("epochs")) {
                command.add("--epochs");
                command.add(String.valueOf(config.get("epochs")));
            }
            if (config.containsKey("batchSize")) {
                command.add("--batch-size");
                command.add(String.valueOf(config.get("batchSize")));
            }
            if (config.containsKey("learningRate")) {
                command.add("--learning-rate");
                command.add(String.valueOf(config.get("learningRate")));
            }
            
            command.add("--job-id");
            command.add(jobId);
            
            // Update job status
            job.setStatusWithTimestamp("RUNNING");
            trainingJobRepository.save(job);
            
            System.out.println("Starting training process with command: " + String.join(" ", command));
            
            // Start process
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            
            Process process = processBuilder.start();
            trainingProcesses.put(jobId, process);
            
            // Create a StringBuilder to store full output
            StringBuilder outputLog = new StringBuilder();
            
            // Read output to track progress and update status
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                
                // Regular expressions for parsing key outputs
                Pattern progressPattern = Pattern.compile("PROGRESS:\\s*(\\d+)");
                Pattern modelIdPattern = Pattern.compile("MODEL_ID:\\s*(\\d+)");
                Pattern errorPattern = Pattern.compile("Error:.*");
                
                while ((line = reader.readLine()) != null) {
                    System.out.println("Training Job " + jobId + ": " + line);
                    outputLog.append(line).append("\n");
                    
                    // Parse progress updates
                    Matcher progressMatcher = progressPattern.matcher(line);
                    if (progressMatcher.find()) {
                        try {
                            int progress = Integer.parseInt(progressMatcher.group(1));
                            job.setProgress(progress);
                            trainingJobRepository.save(job);
                        } catch (NumberFormatException e) {
                            // Ignore parse errors
                        }
                    }
                    
                    // Parse model ID when training completes
                    Matcher modelIdMatcher = modelIdPattern.matcher(line);
                    if (modelIdMatcher.find()) {
                        try {
                            Long modelId = Long.parseLong(modelIdMatcher.group(1));
                            job.setModelId(modelId);
                            trainingJobRepository.save(job);
                        } catch (NumberFormatException e) {
                            // Ignore parse errors
                        }
                    }
                    
                    // Check for error messages
                    Matcher errorMatcher = errorPattern.matcher(line);
                    if (errorMatcher.find()) {
                        job.setErrorMessage(line);
                        trainingJobRepository.save(job);
                    }
                }
            }
            
            // Wait for process to complete
            int exitCode = process.waitFor();
            
            // Update job status based on exit code
            if (exitCode == 0) {
                job.setStatusWithTimestamp("COMPLETED");
                job.setProgress(100);
                
                // Try to find the actual model file in the output
                Pattern modelPathPattern = Pattern.compile("Model parameters saved to ([\\w./\\-_]+)");
                Matcher modelPathMatcher = modelPathPattern.matcher(outputLog);
                
                String modelPath = null;
                if (modelPathMatcher.find()) {
                    modelPath = modelPathMatcher.group(1);
                }
                
                // Register the model in the database if a model ID was provided
                if (job.getModelId() != null && 
                    config.containsKey("modelName") && 
                    config.containsKey("description")) {
                    
                    try {
                        String modelName = (String) config.get("modelName");
                        String description = (String) config.get("description");
                        
                        ModelDTO modelDTO = new ModelDTO();
                        modelDTO.setName(modelName);
                        modelDTO.setDescription(description);
                        modelDTO.setDatasetType(datasetType);
                        
                        // Set appropriate file format based on dataset type
                        String fileFormat;
                        if (datasetType.equalsIgnoreCase("parkinsons") || 
                            datasetType.equalsIgnoreCase("reinopath")) {
                            fileFormat = "pkl";
                        } else {
                            fileFormat = "json";
                        }
                        modelDTO.setFileFormat(fileFormat);
                        
                        // Register model with model service
                        if (modelPath != null) {
                            Model model = modelService.registerModel(modelDTO, modelPath);
                            job.setModelId(model.getId());
                        }
                    } catch (Exception e) {
                        System.err.println("Error registering model: " + e.getMessage());
                        job.setErrorMessage("Model was trained but registration failed: " + e.getMessage());
                    }
                }
            } else {
                job.setStatusWithTimestamp("FAILED");
                job.setErrorMessage("Training process exited with code: " + exitCode);
            }
            
            // Clean up temporary config file
            try {
                Files.deleteIfExists(Paths.get(tempConfigPath));
                System.out.println("Deleted temporary config file: " + tempConfigPath);
            } catch (Exception e) {
                System.err.println("Error deleting temp config file: " + e.getMessage());
            }
            
        } catch (Exception e) {
            job.setStatusWithTimestamp("FAILED");
            job.setErrorMessage(e.getMessage());
        } finally {
            // Clean up dataset file if it's in a temp location
            if (datasetPath.contains("/temp/") || datasetPath.contains("\\temp\\")) {
                try {
                    Files.deleteIfExists(Paths.get(datasetPath));
                    System.out.println("Deleted temporary dataset file: " + datasetPath);
                } catch (Exception e) {
                    System.err.println("Error deleting dataset file: " + e.getMessage());
                }
            }
            
            // Remove from active processes map
            trainingProcesses.remove(jobId);
            
            // Save final job status
            trainingJobRepository.save(job);
        }
    }
    
    /**
     * Get current status of a training job
     */
    public TrainingJob getTrainingStatus(String jobId) {
        return trainingJobRepository.findById(jobId).orElse(null);
    }
    
    /**
     * Cancel a training job if it's running
     */
    public boolean cancelTraining(String jobId) {
        TrainingJob job = trainingJobRepository.findById(jobId).orElse(null);
        
        if (job == null || !job.getStatus().equals("RUNNING")) {
            return false;
        }
        
        // Try to kill process
        Process process = trainingProcesses.get(jobId);
        if (process != null && process.isAlive()) {
            process.destroy();
            trainingProcesses.remove(jobId);
            
            // Update job status
            job.setStatusWithTimestamp("CANCELLED");
            trainingJobRepository.save(job);
            
            // Clean up dataset file if it's in a temp location
            if (job.getDatasetPath() != null && 
                (job.getDatasetPath().contains("/temp/") || job.getDatasetPath().contains("\\temp\\"))) {
                try {
                    Files.deleteIfExists(Paths.get(job.getDatasetPath()));
                } catch (Exception e) {
                    System.err.println("Error deleting dataset file: " + e.getMessage());
                }
            }
            
            return true;
        }
        
        return false;
    }
    
    /**
     * Get all training jobs
     */
    public List<TrainingJob> getAllTrainingJobs() {
        return trainingJobRepository.findAll();
    }
    
    /**
     * Get active training jobs (with a status of RUNNING)
     */
    public List<TrainingJob> getActiveTrainingJobs() {
        return trainingJobRepository.findByStatus("RUNNING");
    }
}