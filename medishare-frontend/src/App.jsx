// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { GlobalProvider } from './context/GlobalContext';

// Import pages
import Dashboard from './pages/Dashboard';
import ServerManagement from './pages/ServerManagement';
import ClientManagement from './pages/ClientManagement';
import ConfigurationManagement from './pages/ConfigurationManagement';
// No longer need to import the ModelManagement page
import ModelParametersManagement from './pages/ModelParametersManagement';

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
                {/* Removed the /models route */}
                <Route path="/parameters" element={<ModelParametersManagement />} />
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