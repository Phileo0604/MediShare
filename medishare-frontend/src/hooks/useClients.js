// src/hooks/useClients.js
import { useState, useEffect } from 'react';
import { clientApi } from '../api/clientApi';

export const useClients = () => {
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Start a new client
  const startClient = async (datasetType, cycles, serverHost) => {
    setLoading(true);
    setError(null);
    
    try {
      const newClient = await clientApi.startClient(datasetType, cycles, serverHost);
      setClients(prev => [...prev, newClient]);
      setLoading(false);
      return { success: true, client: newClient };
    } catch (err) {
      setError(err.message || 'Failed to start client');
      setLoading(false);
      return { success: false, error: err.message };
    }
  };

  // Stop a client
  const stopClient = async (clientId) => {
    setLoading(true);
    setError(null);
    
    try {
      await clientApi.stopClient(clientId);
      setClients(prev => prev.filter(client => client.clientId !== clientId));
      setLoading(false);
      return { success: true };
    } catch (err) {
      setError(err.message || 'Failed to stop client');
      setLoading(false);
      return { success: false, error: err.message };
    }
  };

  // Update client statuses
  const updateClientStatuses = async () => {
    if (clients.length === 0) return;
    
    try {
      const updatedClients = await Promise.all(
        clients.map(async (client) => {
          try {
            const status = await clientApi.getClientStatus(client.clientId);
            return { ...client, ...status };
          } catch {
            // If we can't get status, assume client is gone
            return null;
          }
        })
      );
      
      // Filter out null values
      setClients(updatedClients.filter(Boolean));
    } catch (err) {
      console.error('Error updating client statuses:', err);
    }
  };

  // Set up status update interval
  useEffect(() => {
    if (clients.length > 0) {
      const interval = setInterval(updateClientStatuses, 5000);
      return () => clearInterval(interval);
    }
  }, [clients]);

  return {
    clients,
    loading,
    error,
    startClient,
    stopClient,
    updateClientStatuses
  };
};