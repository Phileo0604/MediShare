// src/main/java/com/medishare/api/controller/LogsController.java
package com.medishare.api.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@RestController
@RequestMapping("/api/logs")
public class LogsController {
    
    // Using default values in case the properties aren't found
    @Value("${app.log-files.server:server.log}")
    private String serverLogPath;
    
    @Value("${app.log-files.client:client*.log}")
    private String clientLogPattern;
    
    @GetMapping
    public ResponseEntity<?> getLogs(@RequestParam(defaultValue = "100") int lines) {
        try {
            List<LogEntry> logs = new ArrayList<>();
            
            // Add server logs
            File serverLogFile = new File(serverLogPath);
            if (serverLogFile.exists()) {
                logs.addAll(readLogs(serverLogPath, "Server", lines));
            }
            
            // Add client logs - this will read client logs from all matching files
            try {
                Path dir = Paths.get(".");
                try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "client*.log")) {
                    for (Path path : stream) {
                        logs.addAll(readLogs(path.toString(), "Client", lines));
                    }
                }
            } catch (Exception e) {
                // Just log the error but continue
                System.err.println("Error reading client logs: " + e.getMessage());
            }
            
            // Sort logs by timestamp (newest first)
            logs.sort((a, b) -> b.getTimestamp().compareTo(a.getTimestamp()));
            
            // Limit to requested number of lines
            if (logs.size() > lines) {
                logs = logs.subList(0, lines);
            }
            
            return ResponseEntity.ok(logs);
        } catch (Exception e) {
            return ResponseEntity.ok(List.of(new LogEntry(
                java.time.LocalDateTime.now().toString(), 
                "Error", 
                "Error reading logs: " + e.getMessage()
            )));
        }
    }
    
    private List<LogEntry> readLogs(String filePath, String source, int maxLines) throws Exception {
        List<LogEntry> logs = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null && logs.size() < maxLines) {
                LogEntry entry = parseLine(line, source);
                if (entry != null) {
                    logs.add(entry);
                }
            }
        } catch (Exception e) {
            logs.add(new LogEntry(
                java.time.LocalDateTime.now().toString(),
                source,
                "Error reading log file " + filePath + ": " + e.getMessage()
            ));
        }
        
        return logs;
    }
    
    private LogEntry parseLine(String line, String defaultSource) {
        try {
            // Pattern to extract timestamp and source
            Pattern pattern = Pattern.compile("(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2})\\s+(\\w+(?:_\\w+)*):\\s+(.*)");
            Matcher matcher = pattern.matcher(line);
            
            if (matcher.find()) {
                String timestamp = matcher.group(1);
                String source = defaultSource != null ? defaultSource : matcher.group(2);
                String message = matcher.group(3);
                
                return new LogEntry(timestamp, source, message);
            }
            
            // If the pattern doesn't match, just use the whole line as the message
            return new LogEntry(
                java.time.LocalDateTime.now().toString(),
                defaultSource,
                line
            );
        } catch (Exception e) {
            // Fallback for any parsing errors
            return new LogEntry(
                java.time.LocalDateTime.now().toString(),
                defaultSource,
                line
            );
        }
    }
    
    // Inner class for log entries
    static class LogEntry {
        private String timestamp;
        private String source;
        private String message;
        
        public LogEntry(String timestamp, String source, String message) {
            this.timestamp = timestamp;
            this.source = source;
            this.message = message;
        }
        
        public String getTimestamp() {
            return timestamp;
        }
        
        public String getSource() {
            return source;
        }
        
        public String getMessage() {
            return message;
        }
    }
}