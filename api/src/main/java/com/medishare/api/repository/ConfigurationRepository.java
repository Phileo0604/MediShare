package com.medishare.api.repository;

import com.medishare.api.model.Configuration;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface ConfigurationRepository extends JpaRepository<Configuration, Long> {
    Optional<Configuration> findByDatasetType(String datasetType);
    Optional<Configuration> findByDatasetTypeAndIsActive(String datasetType, boolean isActive);
}