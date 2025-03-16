package com.medishare.api.repository;

import com.medishare.api.model.Model;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional;

public interface ModelRepository extends JpaRepository<Model, Long> {
    List<Model> findByDatasetType(String datasetType);
    Optional<Model> findByDatasetTypeAndActive(String datasetType, boolean active);
}