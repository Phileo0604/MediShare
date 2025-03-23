// src/pages/ClientManagement.jsx
import React, { useState, useEffect } from 'react';
import ClientForm from '../components/client/ClientForm';
import ClientHistory from '../components/client/ClientHistory';
import LogViewer from '../components/logs/LogViewer';
import { useClientHistory } from '../hooks/useClientHistory';
import { useGlobalContext } from '../context/GlobalContext';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { clientApi } from '../api/clientApi';
import '../styles/ClientManagement.css';

const ClientManagement = () => {
  const { selectedDatasetType } = useGlobalContext();
  const { 
    clientHistory, 
    loading: historyLoading, 
    error: historyError,
    fetchClientHistory,
    deleteClientHistory,
    refreshClientStatuses
  } = useClientHistory(selectedDatasetType);
  
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5 seconds default
  
  // Handle change in refresh interval
  const handleRefreshIntervalChange = (e) => {
    setRefreshInterval(parseInt(e.target.value, 10));
  };
  
  const handleClientStarted = async (clientInfo) => {
    // After a client is started, refresh the history
    await fetchClientHistory();
  };
  
  const handleDeleteClient = async (clientId, stopFirst = false) => {
    await deleteClientHistory(clientId, stopFirst);
  };
  
  return (
    <div className="client-management">
      <h1>Client Management</h1>
      
      <div className="client-form-section">
        <ClientForm 
          onClientStarted={handleClientStarted}
        />
      </div>
      
      <div className="client-history-section">
        <ClientHistory 
          clientHistory={clientHistory}
          loading={historyLoading}
          error={historyError}
          onRefresh={refreshClientStatuses}
          onDelete={handleDeleteClient}
        />
      </div>
      
      {/* Logs Section */}
      <div className="logs-section">
        <div className="logs-header">
          <h2>Client Logs</h2>
          <div className="refresh-control">
            <label htmlFor="refreshInterval">Refresh Interval:</label>
            <select 
              id="refreshInterval" 
              value={refreshInterval}
              onChange={handleRefreshIntervalChange}
            >
              <option value="1000">1 second</option>
              <option value="5000">5 seconds</option>
              <option value="10000">10 seconds</option>
              <option value="30000">30 seconds</option>
              <option value="60000">1 minute</option>
            </select>
          </div>
        </div>
        <LogViewer 
          maxLines={200} 
          refreshInterval={refreshInterval} 
          sourceFilter="client" // Optional: Pre-filter to only show client logs
        />
      </div>
    </div>
  );
};

export default ClientManagement;