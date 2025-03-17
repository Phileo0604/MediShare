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
      setServerStatus({
        isRunning: data.isRunning,
        datasetType: data.datasetType,
        startTime: data.startTime,
        activeClients: data.activeClients?.length || 0,
        loading: false,
        error: null
      });
      if (data.activeClients) {
        setActiveClients(data.activeClients);
      }
    } catch (error) {
      setServerStatus(prev => ({ 
        ...prev, 
        loading: false, 
        error: error.message || 'Failed to fetch server status' 
      }));
    }
  };

  // Start the server
  const startServer = async (datasetType) => {
    try {
      setServerStatus(prev => ({ ...prev, loading: true, error: null }));
      await serverApi.startServer(datasetType);
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