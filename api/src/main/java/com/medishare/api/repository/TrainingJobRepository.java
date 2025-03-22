package com.medishare.api.repository;

import com.medishare.api.model.TrainingJob;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface TrainingJobRepository extends JpaRepository<TrainingJob, String> {
    List<TrainingJob> findByStatus(String status);
    List<TrainingJob> findByDatasetType(String datasetType);
    List<TrainingJob> findByDatasetTypeAndStatus(String datasetType, String status);
}