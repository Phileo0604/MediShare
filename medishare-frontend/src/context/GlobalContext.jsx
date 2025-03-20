// src/context/GlobalContext.jsx
import React, { createContext, useContext, useState, useEffect } from 'react';
import { serverApi } from '../api/serverApi';

// Create the context
const GlobalContext = createContext();

// Custom hook to use the context
export const useGlobalContext = () => useContext(GlobalContext);

// Provider component
export const GlobalProvider = ({ children }) => {
  // Server state
  const [serverStatus, setServerStatus] = useState({
    isRunning: false,
    datasetType: null,
    startTime: null,
    activeClients: 0,
    loading: true,
    error: null
  });

  // Active clients
  const [activeClients, setActiveClients] = useState([]);

  // Fetch server status
  const fetchServerStatus = async () => {
    try {
      setServerStatus(prev => ({ ...prev, loading: true, error: null }));
      const data = await serverApi.getServerStatus();
      
      // Basic server status
      const serverStatusUpdate = {
        isRunning: data.status === 'running',
        datasetType: data.datasetType || null,
        startTime: data.startTime || null,
        loading: false,
        error: null
      };
      
      // If server is running, fetch active clients from logs
      if (serverStatusUpdate.isRunning) {
        try {
          // Get all CLIENT_STARTED logs without matching STOPPED logs
          const activeClientLogs = await logsApi.getLogsByEventType('CLIENT_STARTED');
          const stoppedClientLogs = await logsApi.getLogsByEventType('CLIENT_STOPPED');
          
          // Filter out clients that have been stopped
          const stoppedClientIds = stoppedClientLogs.map(log => log.clientId);
          const activeClients = activeClientLogs
            .filter(log => !stoppedClientIds.includes(log.clientId))
            .map(log => {
              const details = log.details ? JSON.parse(log.details) : {};
              return {
                clientId: log.clientId,
                datasetType: log.datasetType,
                startTime: log.createdAt,
                serverHost: details.serverHost || 'unknown',
                cycles: details.cycles || 1,
                completedCycles: 0 // We can't know this without additional logging
              };
            });
          
          serverStatusUpdate.activeClients = activeClients.length;
          setActiveClients(activeClients);
        } catch (err) {
          console.error('Error fetching active clients from logs:', err);
        }
      } else {
        serverStatusUpdate.activeClients = 0;
        setActiveClients([]);
      }
      
      setServerStatus(serverStatusUpdate);
    } catch (error) {
      setServerStatus(prev => ({ 
        ...prev, 
        loading: false, 
        error: error.message || 'Failed to fetch server status' 
      }));
    }
  };

  // Start the server
  const startServer = async (datasetType, configId = null) => {
    try {
      setServerStatus(prev => ({ ...prev, loading: true, error: null }));
      await serverApi.startServer(datasetType, configId);
      await fetchServerStatus();
      return true;
    } catch (error) {
      setServerStatus(prev => ({ 
        ...prev, 
        loading: false, 
        error: error.message || 'Failed to start server' 
      }));
      return false;
    }
  };

  // Stop the server
  const stopServer = async () => {
    try {
      setServerStatus(prev => ({ ...prev, loading: true, error: null }));
      await serverApi.stopServer();
      await fetchServerStatus();
      return true;
    } catch (error) {
      setServerStatus(prev => ({ 
        ...prev, 
        loading: false, 
        error: error.message || 'Failed to stop server' 
      }));
      return false;
    }
  };

  // Selected dataset type
  const [selectedDatasetType, setSelectedDatasetType] = useState('breast_cancer');

  // Fetch status on component mount
  useEffect(() => {
    fetchServerStatus();
    // Set up an interval to refresh the server status every 10 seconds
    const interval = setInterval(fetchServerStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  // Values to be provided by context
  const value = {
    serverStatus,
    fetchServerStatus,
    startServer,
    stopServer,
    activeClients,
    selectedDatasetType,
    setSelectedDatasetType
  };

  return (
    <GlobalContext.Provider value={value}>
      {children}
    </GlobalContext.Provider>
  );
};

export default GlobalContext;