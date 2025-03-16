package com.medishare.api.controller;

import com.medishare.api.service.ServerService;
import com.medishare.api.util.FileSystemUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/server")
public class ServerController {
    private final ServerService serverService;
    private final FileSystemUtil fileSystemUtil;
    
    @Autowired
    public ServerController(ServerService serverService, FileSystemUtil fileSystemUtil) {
        this.serverService = serverService;
        this.fileSystemUtil = fileSystemUtil;
    }
    
    @PostMapping("/start")
    public ResponseEntity<?> startServer(@RequestParam String datasetType) {
        try {
            // Get the path to the configuration file
            String configPath = fileSystemUtil.getConfigFilePath(datasetType);
            
            // Check if the configuration file exists
            if (!fileSystemUtil.checkFileExists(configPath)) {
                return ResponseEntity.badRequest().body("Configuration file not found for dataset type: " + datasetType);
            }
            
            // Start the server
            boolean started = serverService.startServer(configPath);
            
            if (started) {
                return ResponseEntity.ok("Server started successfully for " + datasetType);
            } else {
                return ResponseEntity.internalServerError().body("Failed to start server");
            }
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Error starting server: " + e.getMessage());
        }
    }
    
    @PostMapping("/stop")
    public ResponseEntity<String> stopServer() {
        boolean stopped = serverService.stopServer();
        if (stopped) {
            return ResponseEntity.ok("Server stopped successfully");
        } else {
            return ResponseEntity.internalServerError().body("Failed to stop server");
        }
    }
    
    @GetMapping("/status")
    public ResponseEntity<Map<String, String>> getServerStatus() {
        String status = serverService.getServerStatus();
        Map<String, String> statusInfo = new HashMap<>();
        statusInfo.put("status", status);
        return ResponseEntity.ok(statusInfo);
    }
}