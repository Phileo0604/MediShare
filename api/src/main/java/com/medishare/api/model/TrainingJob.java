package com.medishare.api.model;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "training_jobs")
public class TrainingJob {
    @Id
    @Column(length = 36)
    private String jobId;
    
    @Column(nullable = false)
    private String datasetType;
    
    @Column(nullable = false)
    private String status;
    
    @Column
    private int progress;
    
    @Column
    private String datasetPath;
    
    @Column(columnDefinition = "TEXT")
    private String configJson;
    
    @Column(columnDefinition = "TEXT")
    private String errorMessage;
    
    @Column
    private Long modelId;
    
    @Column(nullable = false)
    private LocalDateTime createdAt;
    
    @Column
    private LocalDateTime updatedAt;
    
    /**
     * Set status and update the timestamp
     */
    public void setStatusWithTimestamp(String status) {
        this.status = status;
        this.updatedAt = LocalDateTime.now();
    }
}