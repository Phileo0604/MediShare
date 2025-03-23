// src/hooks/useClientHistory.js
import { useState, useEffect } from 'react';
import { clientApi } from '../api/clientApi';

export const useClientHistory = (datasetType) => {
  const [clientHistory, setClientHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchClientHistory = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const history = await clientApi.getClientHistory(datasetType);
      setClientHistory(history);
      setLoading(false);
      return true;
    } catch (err) {
      setError(err.message || 'Failed to load client history');
      setLoading(false);
      return false;
    }
  };

  // Delete a client history entry
  const deleteClientHistory = async (clientId, stopFirst = false) => {
    try {
      // If client is running, stop it first
      if (stopFirst) {
        await clientApi.stopClient(clientId);
      }
      
      // Now delete from history
      await clientApi.deleteClientHistory(clientId);
      
      // Update the local state to remove the deleted entry
      setClientHistory(prevHistory => 
        prevHistory.filter(client => client.clientId !== clientId)
      );
      
      return true;
    } catch (err) {
      setError(err.message || `Failed to delete client ${clientId}`);
      return false;
    }
  };

  // Refresh all client statuses
  const refreshClientStatuses = async () => {
    setError(null);
    
    try {
      // Option 1: Backend endpoint to refresh all statuses (if available)
      // const updatedHistory = await clientApi.refreshClientStatuses(datasetType);
      // setClientHistory(updatedHistory);
      
      // Option 2: Manual refresh by fetching individual statuses
      const updatedHistory = await Promise.all(
        clientHistory.map(async (client) => {
          try {
            // If not running, no need to check status
            if (client.status !== 'running') {
              return client;
            }
            
            const statusResponse = await clientApi.getClientStatus(client.clientId);
            return {
              ...client,
              status: statusResponse.status
            };
          } catch (error) {
            // If we can't get status, assume there's an issue
            return client;
          }
        })
      );
      
      setClientHistory(updatedHistory);
      return true;
    } catch (err) {
      setError(err.message || 'Failed to refresh client statuses');
      return false;
    }
  };

  // Fetch history when dataset type changes
  useEffect(() => {
    if (datasetType) {
      fetchClientHistory();
    }
  }, [datasetType]);

  return {
    clientHistory,
    loading,
    error,
    fetchClientHistory,
    deleteClientHistory,
    refreshClientStatuses
  };
};