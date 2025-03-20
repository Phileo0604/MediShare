package com.medishare.api.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.io.File;

@Service
public class ClientService {
    
    @Value("${app.python.script-path}")
    private String pythonScriptPath;
    
    // Store client processes by client ID
    private final Map<String, Process> clientProcesses = new ConcurrentHashMap<>();
    
    // Store client history
    private final List<Map<String, Object>> clientHistory = Collections.synchronizedList(new ArrayList<>());
    
    public boolean startClient(String modelFilePath, String configPath, String datasetType, String clientId, String serverHost, int cycles) {
        try {
            // Kill any existing process with the same client ID
            stopClient(clientId);
            
            // Build command with appropriate arguments
            List<String> command = new ArrayList<>();
            command.add("python");
            command.add(pythonScriptPath);
            command.add("--mode"); 
            command.add("client");
            command.add("--config");
            command.add(configPath);
            command.add("--dataset-type");
            command.add(datasetType);
            command.add("--client-id");
            command.add(clientId);
            command.add("--server-host");
            command.add(serverHost);
            command.add("--cycles");
            command.add(String.valueOf(cycles));
            
            // If model file path is provided, use parameter-only mode
            if (modelFilePath != null && !modelFilePath.isEmpty()) {
                command.add("--parameter-path");
                command.add(modelFilePath);
                command.add("--skip-training");
                command.add("--skip-dataset");  // Add this to skip dataset loading
            }
            
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            
            // Start the process
            Process process = processBuilder.start();
            
            // Store the process
            clientProcesses.put(clientId, process);
            
            // Start a thread to read and log output
            new Thread(() -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println("Client " + clientId + ": " + line);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
            
            // Wait a bit to see if it crashes immediately
            Thread.sleep(1000);
            
            if (!process.isAlive()) {
                clientProcesses.remove(clientId);
                return false;
            }
            
            // Create a file with client info
            Map<String, String> clientInfo = new HashMap<>();
            clientInfo.put("pid", String.valueOf(process.pid()));
            clientInfo.put("datasetType", datasetType);
            clientInfo.put("configPath", configPath);
            clientInfo.put("serverHost", serverHost);
            clientInfo.put("cycles", String.valueOf(cycles));
            if (modelFilePath != null) {
                clientInfo.put("modelFilePath", modelFilePath);
                clientInfo.put("skipTraining", "true");
            }
            
            Path clientInfoPath = Paths.get("client_" + clientId + ".info");
            Files.writeString(clientInfoPath, clientInfo.toString());
            
            // Add to client history
            Map<String, Object> historyEntry = new HashMap<>();
            historyEntry.put("clientId", clientId);
            historyEntry.put("datasetType", datasetType);
            historyEntry.put("serverHost", serverHost);
            historyEntry.put("cycles", cycles);
            historyEntry.put("startTime", System.currentTimeMillis());
            historyEntry.put("status", "started");
            if (modelFilePath != null) {
                historyEntry.put("usingModelParameters", true);
                historyEntry.put("modelFilePath", modelFilePath);
            }
            
            // Add to history
            clientHistory.add(historyEntry);
            
            // Limit history size to prevent memory issues
            while (clientHistory.size() > 100) {
                clientHistory.remove(0);
            }
            
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    public boolean stopClient(String clientId) {
        try {
            // Try to kill by process reference first
            Process process = clientProcesses.get(clientId);
            if (process != null && process.isAlive()) {
                process.destroy();
                clientProcesses.remove(clientId);
                
                // Clean up client info file
                Path clientInfoPath = Paths.get("client_" + clientId + ".info");
                if (Files.exists(clientInfoPath)) {
                    Files.delete(clientInfoPath);
                }
                
                // Update history entry
                updateClientHistoryStatus(clientId, "stopped");
                
                return true;
            }
            
            // Try to kill by PID file as fallback
            Path clientInfoPath = Paths.get("client_" + clientId + ".info");
            if (Files.exists(clientInfoPath)) {
                String clientInfoContent = Files.readString(clientInfoPath);
                // Very simple parsing of the toString() output of the HashMap
                if (clientInfoContent.contains("pid=")) {
                    String pidStr = clientInfoContent.substring(
                            clientInfoContent.indexOf("pid=") + 4, 
                            clientInfoContent.indexOf(",", clientInfoContent.indexOf("pid="))
                    );
                    Runtime.getRuntime().exec("kill " + pidStr);
                    Files.delete(clientInfoPath);
                    
                    // Update history entry
                    updateClientHistoryStatus(clientId, "stopped");
                    
                    return true;
                }
            }
            
            return false;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    public String getClientStatus(String clientId) {
        try {
            // Check if we have a process reference
            Process process = clientProcesses.get(clientId);
            if (process != null) {
                if (process.isAlive()) {
                    return "running";
                } else {
                    int exitCode = process.exitValue();
                    clientProcesses.remove(clientId);
                    
                    // Update history entry
                    updateClientHistoryStatus(clientId, "completed with exit code " + exitCode);
                    
                    return "completed with exit code " + exitCode;
                }
            }
            
            // Check client info file as fallback
            Path clientInfoPath = Paths.get("client_" + clientId + ".info");
            if (Files.exists(clientInfoPath)) {
                String clientInfoContent = Files.readString(clientInfoPath);
                if (clientInfoContent.contains("pid=")) {
                    String pidStr = clientInfoContent.substring(
                            clientInfoContent.indexOf("pid=") + 4, 
                            clientInfoContent.indexOf(",", clientInfoContent.indexOf("pid="))
                    );
                    
                    Process checkProcess = Runtime.getRuntime().exec("ps -p " + pidStr);
                    int exitCode = checkProcess.waitFor();
                    
                    String status = exitCode == 0 ? "running" : "stopped";
                    
                    // Update history entry if stopped
                    if (status.equals("stopped")) {
                        updateClientHistoryStatus(clientId, "stopped");
                    }
                    
                    return status;
                }
            }
            
            return "not found";
        } catch (Exception e) {
            e.printStackTrace();
            return "error: " + e.getMessage();
        }
    }
    
    private void updateClientHistoryStatus(String clientId, String status) {
        synchronized (clientHistory) {
            for (Map<String, Object> entry : clientHistory) {
                if (entry.get("clientId").equals(clientId)) {
                    entry.put("status", status);
                    if (status.equals("stopped") || status.startsWith("completed")) {
                        entry.put("endTime", System.currentTimeMillis());
                    }
                    break;
                }
            }
        }
    }
    
    public List<Map<String, Object>> getClientHistory(String datasetType) {
        List<Map<String, Object>> filteredHistory = new ArrayList<>();
        
        synchronized (clientHistory) {
            for (Map<String, Object> entry : clientHistory) {
                if (entry.get("datasetType").equals(datasetType)) {
                    filteredHistory.add(new HashMap<>(entry));  // Create a copy to avoid concurrency issues
                }
            }
        }
        
        // Sort by start time descending (newest first)
        filteredHistory.sort((a, b) -> {
            Long timeA = (Long) a.get("startTime");
            Long timeB = (Long) b.get("startTime");
            return timeB.compareTo(timeA);
        });
        
        return filteredHistory;
    }
    
    public List<Map<String, Object>> getAllClientHistory() {
        List<Map<String, Object>> historyCopy = new ArrayList<>();
        
        synchronized (clientHistory) {
            for (Map<String, Object> entry : clientHistory) {
                historyCopy.add(new HashMap<>(entry));  // Create a copy to avoid concurrency issues
            }
        }
        
        // Sort by start time descending (newest first)
        historyCopy.sort((a, b) -> {
            Long timeA = (Long) a.get("startTime");
            Long timeB = (Long) b.get("startTime");
            return timeB.compareTo(timeA);
        });
        
        return historyCopy;
    }
    
    /**
     * Get a list of currently active client IDs
     * @return List of active client IDs
     */
    public List<String> getActiveClientIds() {
        List<String> activeClients = new ArrayList<>();
        
        // Check process map first
        for (Map.Entry<String, Process> entry : clientProcesses.entrySet()) {
            if (entry.getValue().isAlive()) {
                activeClients.add(entry.getKey());
            }
        }
        
        // Check for client info files as fallback
        try {
            File currentDir = new File(".");
            String[] files = currentDir.list((dir, name) -> name.startsWith("client_") && name.endsWith(".info"));
            
            if (files != null) {
                for (String file : files) {
                    String clientId = file.substring(7, file.length() - 5);  // Remove "client_" prefix and ".info" suffix
                    if (!activeClients.contains(clientId)) {
                        String status = getClientStatus(clientId);
                        if (status.equals("running")) {
                            activeClients.add(clientId);
                        }
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        return activeClients;
    }
}