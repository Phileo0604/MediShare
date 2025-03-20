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
    fetchClientHistory
  };
};