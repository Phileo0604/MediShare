// src/components/client/ClientHistory.jsx
import React from 'react';
import { formatDate } from '../../utils/formatters';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorMessage from '../common/ErrorMessage';

const ClientHistory = ({ clientHistory, loading, error }) => {
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

  return (
    <div className="client-history">
      <table className="history-table">
        <thead>
          <tr>
            <th>Client ID</th>
            <th>Started</th>
            <th>Status</th>
            <th>Duration</th>
            <th>Server Host</th>
            <th>Cycles</th>
          </tr>
        </thead>
        <tbody>
          {clientHistory.map((record) => (
            <tr key={record.clientId + record.startTime}>
              <td>{record.clientId}</td>
              <td>{formatDate(record.startTime)}</td>
              <td>
                <span className={`status-badge ${getStatusClass(record.status)}`}>
                  {record.status}
                </span>
              </td>
              <td>{calculateDuration(record.startTime, record.endTime)}</td>
              <td>{record.serverHost}</td>
              <td>{record.cycles}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ClientHistory;