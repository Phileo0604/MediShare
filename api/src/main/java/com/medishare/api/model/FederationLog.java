// src/main/java/com/medishare/api/model/FederationLog.java
package com.medishare.api.model;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;
import com.vladmihalcea.hibernate.type.json.JsonBinaryType;
import org.hibernate.annotations.Type;

@Data
@Entity
@Table(name = "federation_logs")
public class FederationLog {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private String eventType;
    
    @Column(nullable = false)
    private String datasetType;
    
    private String clientId;
    
    private Long userId;
    
    @Column(columnDefinition = "JSONB")
    private String details;
    
    @Column(nullable = false)
    private LocalDateTime createdAt;

    
}