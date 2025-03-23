// src/components/client/ClientHistory.jsx
import React, { useState } from 'react';
import { formatDate } from '../../utils/formatters';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorMessage from '../common/ErrorMessage';
import { clientApi } from '../../api/clientApi';

const ClientHistory = ({ clientHistory, loading, error, onRefresh, onDelete }) => {
  const [deletingClientId, setDeletingClientId] = useState(null);
  const [refreshingStatus, setRefreshingStatus] = useState(false);

  if (loading) {
    return <LoadingSpinner />;
  }
  
  if (error) {
    return <ErrorMessage message={error} />;
  }
  
  if (!clientHistory || clientHistory.length === 0) {
    return <p className="no-items-message">No client history found.</p>;
  }

  const getStatusClass = (status) => {
    if (status === 'running') return 'active';
    if (status === 'stopped') return 'inactive';
    if (status.startsWith('completed')) return 'completed';
    return 'inactive';
  };
  
  const calculateDuration = (startTime, endTime) => {
    if (!endTime) return 'In progress';
    
    const durationMs = endTime - startTime;
    const seconds = Math.floor(durationMs / 1000);
    
    if (seconds < 60) {
      return `${seconds} seconds`;
    }
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes < 60) {
      return `${minutes} min ${remainingSeconds} sec`;
    }
    
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    
    return `${hours} hr ${remainingMinutes} min`;
  };

  const handleRefreshStatuses = async () => {
    if (refreshingStatus || !onRefresh) return;
    
    setRefreshingStatus(true);
    await onRefresh();
    setRefreshingStatus(false);
  };

  const handleDeleteClient = async (clientId) => {
    if (window.confirm(`Are you sure you want to delete this client history entry?`)) {
      setDeletingClientId(clientId);
      await onDelete(clientId);
      setDeletingClientId(null);
    }
  };

  // Helper to render client name (if any) or ID
  const renderClientName = (record) => {
    const nameDisplay = record.clientName || record.clientId;
    
    // Check if client ID was generated from a name
    const isNamedClient = record.clientId.includes('_') && 
                         !record.clientId.startsWith('client_') &&
                         !record.clientName;
    
    if (isNamedClient) {
      // Convert snake_case to readable format
      return record.clientId.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ');
    }
    
    return nameDisplay;
  };

  return (
    <div className="client-history">
      <div className="history-header">
        <h3>Client History</h3>
        <button 
          className="btn btn-secondary"
          onClick={handleRefreshStatuses}
          disabled={refreshingStatus}
        >
          {refreshingStatus ? 'Refreshing...' : 'Refresh Status'}
        </button>
      </div>

      <table className="history-table">
        <thead>
          <tr>
            <th>Client Name/ID</th>
            <th>Started</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Server Host</th>
            <th>Cycles</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {clientHistory.map((record) => (
            <tr key={record.clientId + record.startTime}>
              <td>{renderClientName(record)}</td>
              <td>{formatDate(record.startTime)}</td>
              <td>
                <span className={`status-badge ${getStatusClass(record.status)}`}>
                  {record.status}
                </span>
              </td>
              <td>{calculateDuration(record.startTime, record.endTime)}</td>
              <td>{record.serverHost}</td>
              <td>{record.cycles}</td>
              <td>
                <div className="action-buttons">
                  {record.status === 'running' && (
                    <button 
                      className="btn btn-warning btn-sm"
                      onClick={() => onDelete(record.clientId, true)}
                      disabled={deletingClientId === record.clientId}
                    >
                      Stop
                    </button>
                  )}
                  <button 
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDeleteClient(record.clientId)}
                    disabled={deletingClientId === record.clientId}
                  >
                    {deletingClientId === record.clientId ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ClientHistory;