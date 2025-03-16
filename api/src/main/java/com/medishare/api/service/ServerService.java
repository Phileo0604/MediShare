package com.medishare.api.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@Service
public class ServerService {
    
    @Value("${app.python.script-path}")
    private String pythonScriptPath;
    
    private Process serverProcess;
    
    public boolean startServer(String configPath) {
        try {
            // Kill any existing process
            stopServer();
            
            System.out.println("Starting server with config: " + configPath);
            System.out.println("Python script path: " + pythonScriptPath);
            
            // Check if Python script exists
            if (!Files.exists(Paths.get(pythonScriptPath))) {
                System.err.println("Python script not found at: " + pythonScriptPath);
                return false;
            }
            
            // Execute your Python server script as a process
            ProcessBuilder processBuilder = new ProcessBuilder("python", pythonScriptPath, 
                    "--mode", "server", "--config", configPath);
            
            // Add environment variables if needed
            // processBuilder.environment().put("PYTHONPATH", "/path/to/python/modules");
            
            // Redirect error stream to output stream
            processBuilder.redirectErrorStream(true);
            
            // Start the process
            serverProcess = processBuilder.start();
            
            // Start a thread to read and log output
            new Thread(() -> {
                try {
                    BufferedReader reader = new BufferedReader(new InputStreamReader(serverProcess.getInputStream()));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println("Server: " + line);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
            
            // Wait a bit to see if it crashes immediately
            Thread.sleep(1000);
            
            if (!serverProcess.isAlive()) {
                int exitCode = serverProcess.exitValue();
                System.err.println("Server process exited immediately with code: " + exitCode);
                return false;
            }
            
            // Write PID file for later management
            Files.writeString(Paths.get("server.pid"), String.valueOf(serverProcess.pid()));
            System.out.println("Server started successfully with PID: " + serverProcess.pid());
            
            return true;
        } catch (Exception e) {
            System.err.println("Error starting server: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
    
    public boolean stopServer() {
        try {
            // Try to kill by process reference first
            if (serverProcess != null && serverProcess.isAlive()) {
                serverProcess.destroy();
                return true;
            }
            
            // Try to kill by PID file as fallback
            Path pidFile = Paths.get("server.pid");
            if (Files.exists(pidFile)) {
                String pid = Files.readString(pidFile);
                Runtime.getRuntime().exec("kill " + pid);
                Files.delete(pidFile);
                return true;
            }
            
            return false;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    
    public String getServerStatus() {
        try {
            // Check if we have a process reference
            if (serverProcess != null && serverProcess.isAlive()) {
                return "running";
            }
            
            // Check PID file as fallback
            Path pidFile = Paths.get("server.pid");
            if (Files.exists(pidFile)) {
                String pid = Files.readString(pidFile);
                Process checkProcess = Runtime.getRuntime().exec("ps -p " + pid);
                int exitCode = checkProcess.waitFor();
                
                return exitCode == 0 ? "running" : "stopped";
            }
            
            return "stopped";
        } catch (Exception e) {
            e.printStackTrace();
            return "unknown";
        }
    }
}