// src/services/LogStore.js
class LogStore {
    constructor() {
      this.logs = [];
      this.maxSize = 1000; // Maximum number of log entries to keep
    }
  
    addLog(source, message) {
      const timestamp = new Date().toISOString().slice(0, 19).replace('T', ' ');
      this.logs.unshift({ timestamp, source, message }); // Add at the beginning for newest first
      
      // Trim logs if they exceed max size
      if (this.logs.length > this.maxSize) {
        this.logs = this.logs.slice(0, this.maxSize);
      }
    }
  
    getLogs(maxLines = 100, filter = '') {
      let filteredLogs = this.logs;
      
      // Apply filter if provided
      if (filter) {
        const lowerFilter = filter.toLowerCase();
        filteredLogs = filteredLogs.filter(log => 
          log.message.toLowerCase().includes(lowerFilter) || 
          log.source.toLowerCase().includes(lowerFilter)
        );
      }
      
      // Return requested number of logs
      return filteredLogs.slice(0, maxLines);
    }
  
    clear() {
      this.logs = [];
    }
  }
  
  // Create a singleton instance
  const logStore = new LogStore();
  
  // Add some initial logs to make the UI more appealing
  logStore.addLog('System', 'MediShare log capture initialized');
  logStore.addLog('Server', 'Waiting for server to start...');
  
  export default logStore;