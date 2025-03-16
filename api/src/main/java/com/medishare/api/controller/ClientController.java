package com.medishare.api.controller;

import com.medishare.api.service.ClientService;
import com.medishare.api.util.FileSystemUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/client")
public class ClientController {
    private final ClientService clientService;
    private final FileSystemUtil fileSystemUtil;
    
    @Autowired
    public ClientController(ClientService clientService, FileSystemUtil fileSystemUtil) {
        this.clientService = clientService;
        this.fileSystemUtil = fileSystemUtil;
    }
    
    @PostMapping("/start")
    public ResponseEntity<?> startClient(
            @RequestParam String datasetType,
            @RequestParam(defaultValue = "1") int cycles,
            @RequestParam(defaultValue = "127.0.0.1") String serverHost) {
        try {
            // Get the path to the configuration file
            String configPath = fileSystemUtil.getConfigFilePath(datasetType);
            
            // Check if the configuration file exists
            if (!fileSystemUtil.checkFileExists(configPath)) {
                return ResponseEntity.badRequest().body("Configuration file not found for dataset type: " + datasetType);
            }
            
            // Generate a client ID
            String clientId = "client_" + System.currentTimeMillis();
            
            // Start the client
            boolean started = clientService.startClient(configPath, datasetType, clientId, serverHost, cycles);
            
            if (started) {
                Map<String, String> response = new HashMap<>();
                response.put("status", "started");
                response.put("clientId", clientId);
                response.put("datasetType", datasetType);
                
                return ResponseEntity.ok(response);
            } else {
                return ResponseEntity.internalServerError().body("Failed to start client");
            }
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error starting client: " + e.getMessage());
        }
    }
    
    @GetMapping("/status/{clientId}")
    public ResponseEntity<?> getClientStatus(@PathVariable String clientId) {
        try {
            String status = clientService.getClientStatus(clientId);
            Map<String, String> response = new HashMap<>();
            response.put("clientId", clientId);
            response.put("status", status);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error getting client status: " + e.getMessage());
        }
    }
    
    @PostMapping("/stop/{clientId}")
    public ResponseEntity<?> stopClient(@PathVariable String clientId) {
        try {
            boolean stopped = clientService.stopClient(clientId);
            if (stopped) {
                return ResponseEntity.ok("Client stopped successfully");
            } else {
                return ResponseEntity.internalServerError().body("Failed to stop client");
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error stopping client: " + e.getMessage());
        }
    }
}