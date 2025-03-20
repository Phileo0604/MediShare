// src/components/common/Header.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import { useGlobalContext } from '../../context/GlobalContext';

const Header = () => {
  const { serverStatus } = useGlobalContext();
  
  return (
    <header className="app-header">
      <div className="logo">
        <Link to="/">
          <h1>MediShare</h1>
        </Link>
      </div>
      
      <div className="server-status-indicator">
        <span className={`status-dot ${serverStatus.isRunning ? 'active' : 'inactive'}`}></span>
        <span className="status-text">
          Server: {serverStatus.isRunning ? 'Online' : 'Offline'}
        </span>
        {serverStatus.isRunning && (
          <span className="dataset-type">({serverStatus.datasetType})</span>
        )}
      </div>
      
      <nav className="main-nav">
        <ul>
          <li>
            <Link to="/">Dashboard</Link>
          </li>
          <li>
            <Link to="/server">Server</Link>
          </li>
          <li>
            <Link to="/clients">Clients</Link>
          </li>
          <li>
            <Link to="/configurations">Configurations</Link>
          </li>
          <li>
            <Link to="/models">Models</Link>
          </li>
          <li>
            <Link to="/parameters">Parameters</Link>
          </li>
        </ul>
      </nav>
    </header>
  );
};

export default Header;