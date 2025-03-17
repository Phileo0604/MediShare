// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { GlobalProvider } from './context/GlobalContext';

// Import pages
import Dashboard from './pages/Dashboard';
import ServerManagement from './pages/ServerManagement';
import ClientManagement from './pages/ClientManagement';
import ConfigurationManagement from './pages/ConfigurationManagement';
import ModelManagement from './pages/ModelManagement';

// Import common components
import Header from './components/common/Header';
import Sidebar from './components/common/Sidebar';
import Footer from './components/common/Footer';

// Import styles
import './index.css';

function App() {
  return (
    <GlobalProvider>
      <Router>
        <div className="app-container">
          <Header />
          <div className="main-content">
            <Sidebar />
            <div className="page-content">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/server" element={<ServerManagement />} />
                <Route path="/clients" element={<ClientManagement />} />
                <Route path="/configurations" element={<ConfigurationManagement />} />
                <Route path="/models" element={<ModelManagement />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </div>
          </div>
          <Footer />
        </div>
      </Router>
    </GlobalProvider>
  );
}

export default App;



