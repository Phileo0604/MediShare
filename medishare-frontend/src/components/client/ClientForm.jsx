// src/components/client/ClientForm.jsx
import React, { useState } from 'react';
import { clientApi } from '../../api/clientApi';
import { useGlobalContext } from '../../context/GlobalContext';

const ClientForm = ({ onClientStarted }) => {
  const { selectedDatasetType, setSelectedDatasetType } = useGlobalContext();
  
  const [formData, setFormData] = useState({
    datasetType: selectedDatasetType,
    cycles: 3,
    serverHost: '127.0.0.1'
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    if (name === 'datasetType') {
      setSelectedDatasetType(value);
    }
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const result = await clientApi.startClient(
        formData.datasetType, 
        formData.cycles, 
        formData.serverHost
      );
      
      if (result && result.clientId) {
        if (onClientStarted) {
          onClientStarted(result);
        }
        // Reset form
        setFormData({
          datasetType: selectedDatasetType,
          cycles: 3,
          serverHost: '127.0.0.1'
        });
      } else {
        setError('Failed to start client. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'Failed to start client');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="client-form">
      <h2>Start New Client</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="datasetType">Dataset Type:</label>
          <select 
            id="datasetType"
            name="datasetType"
            value={formData.datasetType}
            onChange={handleChange}
            disabled={loading}
          >
            <option value="breast_cancer">Breast Cancer</option>
            <option value="parkinsons">Parkinson's</option>
            <option value="reinopath">Reinopath</option>
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="cycles">Training Cycles:</label>
          <input 
            type="number"
            id="cycles"
            name="cycles"
            min="1"
            max="10"
            value={formData.cycles}
            onChange={handleChange}
            disabled={loading}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="serverHost">Server Host:</label>
          <input 
            type="text"
            id="serverHost"
            name="serverHost"
            value={formData.serverHost}
            onChange={handleChange}
            disabled={loading}
          />
        </div>
        
        <button 
          type="submit" 
          className="btn btn-primary"
          disabled={loading}
        >
          {loading ? 'Starting...' : 'Start Client'}
        </button>
      </form>
      
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default ClientForm;