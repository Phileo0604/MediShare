package com.medishare.api.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class ClientService {
    
    @Value("${app.python.script-path}")
    private String pythonScriptPath;
    
    // Store client processes by client ID
    private final Map<String, Process> clientProcesses = new ConcurrentHashMap<>();
    
    public boolean startClient(String configPath, String datasetType, String clientId, String serverHost, int cycles) {
        try {
            // Kill any existing process with the same client ID
            stopClient(clientId);
            
            // Execute the Python client script as a process
            ProcessBuilder processBuilder = new ProcessBuilder(
                "python", 
                pythonScriptPath, 
                "--mode", "client", 
                "--config", configPath,
                "--dataset-type", datasetType,
                "--client-id", clientId,
                "--server-host", serverHost,
                "--cycles", String.valueOf(cycles)
            );
            
            processBuilder.redirectErrorStream(true);
            
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
            
            Path clientInfoPath = Paths.get("client_" + clientId + ".info");
            Files.writeString(clientInfoPath, clientInfo.toString());
            
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
                    
                    return exitCode == 0 ? "running" : "stopped";
                }
            }
            
            return "not found";
        } catch (Exception e) {
            e.printStackTrace();
            return "error: " + e.getMessage();
        }
    }
}