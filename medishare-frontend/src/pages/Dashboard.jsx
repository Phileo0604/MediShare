// src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { configApi } from '../api/configApi';
import { modelApi } from '../api/modelApi';
import { useGlobalContext } from '../context/GlobalContext';
import ServerStatus from '../components/server/ServerStatus';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';

const Dashboard = () => {
  const { serverStatus, activeClients } = useGlobalContext();
  
  const [configCount, setConfigCount] = useState(0);
  const [modelCount, setModelCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Fetch dashboard data
  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch configurations
      const configs = await configApi.getAllConfigurations();
      setConfigCount(configs.length || 0);
      
      // Fetch models
      const models = await modelApi.getAllModels();
      setModelCount(models.length || 0);
      
    } catch (err) {
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchDashboardData();
  }, []);
  
  if (loading) {
    return <LoadingSpinner />;
  }
  
  if (error) {
    return <ErrorMessage message={error} onRetry={fetchDashboardData} />;
  }
  
  return (
    <div className="dashboard">
      <h1>MediShare Dashboard</h1>
      
      <div className="dashboard-cards">
        <div className="card">
          <h3>Server Status</h3>
          <div className="card-content">
            <div className="status-indicator">
              <span className={`status-dot ${serverStatus.isRunning ? 'active' : 'inactive'}`}></span>
              <span className="status-text">
                {serverStatus.isRunning ? 'Running' : 'Stopped'}
              </span>
            </div>
            {serverStatus.isRunning && (
              <p>Dataset: {serverStatus.datasetType}</p>
            )}
          </div>
        </div>
        
        <div className="card">
          <h3>Active Clients</h3>
          <div className="card-content">
            <div className="stat-number">{activeClients.length}</div>
            <p>connected clients</p>
          </div>
        </div>
        
        <div className="card">
          <h3>Configurations</h3>
          <div className="card-content">
            <div className="stat-number">{configCount}</div>
            <p>available configurations</p>
          </div>
        </div>
        
        <div className="card">
          <h3>Models</h3>
          <div className="card-content">
            <div className="stat-number">{modelCount}</div>
            <p>registered models</p>
          </div>
        </div>
      </div>
      
      <div className="quick-actions">
        <h2>Quick Actions</h2>
        <div className="action-buttons">
          <a href="/server" className="btn btn-primary">
            Manage Server
          </a>
          <a href="/clients" className="btn btn-secondary">
            Manage Clients
          </a>
          <a href="/configurations" className="btn btn-secondary">
            Edit Configurations
          </a>
          <a href="/models" className="btn btn-secondary">
            View Models
          </a>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;


