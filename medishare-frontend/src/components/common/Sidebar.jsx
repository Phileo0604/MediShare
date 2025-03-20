// src/components/common/Sidebar.jsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import { useGlobalContext } from '../../context/GlobalContext';

const Sidebar = () => {
  const { serverStatus } = useGlobalContext();
  
  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        <ul>
          <li>
            <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="icon dashboard-icon"></i>
              <span>Dashboard</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/server" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="icon server-icon"></i>
              <span>Server</span>
              <span className={`status-indicator ${serverStatus.isRunning ? 'active' : 'inactive'}`}></span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/clients" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="icon client-icon"></i>
              <span>Clients</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/configurations" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="icon config-icon"></i>
              <span>Configurations</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/models" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="icon model-icon"></i>
              <span>Models</span>
            </NavLink>
          </li>
          <li>
            <NavLink to="/parameters" className={({ isActive }) => isActive ? 'active' : ''}>
              <i className="icon parameters-icon"></i>
              <span>Model Parameters</span>
            </NavLink>
          </li>
        </ul>
      </nav>
      
      <div className="sidebar-footer">
        <div className="version">v1.0.0</div>
      </div>
    </aside>
  );
};

export default Sidebar;