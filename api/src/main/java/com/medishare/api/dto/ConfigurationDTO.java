package com.medishare.api.dto;

import lombok.Data;
import java.util.Map;

@Data
public class ConfigurationDTO {
    private String datasetType;
    private String configName;
    private Map<String, Object> configData;
}