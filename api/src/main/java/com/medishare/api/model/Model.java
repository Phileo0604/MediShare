package com.medishare.api.model;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "models")
public class Model {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private String datasetType;
    
    @Column(nullable = false)
    private String name;
    
    @Column(nullable = false)
    private String filePath;
    
    @Column(nullable = false)
    private String fileFormat;
    
    @Column(columnDefinition = "TEXT")
    private String description;
    
    private LocalDateTime createdAt;
    
    private boolean active;
    
    // Getter and setter for fileFormat
    public String getFileFormat() {
        return fileFormat;
    }
    
    public void setFileFormat(String fileFormat) {
        this.fileFormat = fileFormat;
    }
}