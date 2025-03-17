// src/components/models/ModelUpload.jsx
import React, { useState } from 'react';

const ModelUpload = ({ datasetType, onRegister }) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    filePath: ''
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(false);
    
    const modelData = {
      datasetType: datasetType,
      name: formData.name,
      description: formData.description
    };
    
    try {
      const result = await onRegister(modelData, formData.filePath);
      
      if (result.success) {
        setSuccess(true);
        // Reset form
        setFormData({
          name: '',
          description: '',
          filePath: ''
        });
      } else {
        setError(result.error || 'Failed to register model');
      }
    } catch (err) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="model-upload card">
      <h3>Register New Model</h3>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="name">Model Name:</label>
          <input 
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            required
            disabled={loading}
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="description">Description:</label>
          <textarea 
            id="description"
            name="description"
            value={formData.description}
            onChange={handleChange}
            rows="3"
            disabled={loading}
          ></textarea>
        </div>
        
        <div className="form-group">
          <label htmlFor="filePath">Model File Path:</label>
          <input 
            type="text"
            id="filePath"
            name="filePath"
            value={formData.filePath}
            onChange={handleChange}
            placeholder="Path to model file on server"
            required
            disabled={loading}
          />
          <small className="help-text">
            Enter the path to the model file on the server
          </small>
        </div>
        
        <div className="form-group">
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? 'Registering...' : 'Register Model'}
          </button>
        </div>
      </form>
      
      {error && <p className="error-message">{error}</p>}
      {success && <p className="success-message">Model registered successfully!</p>}
    </div>
  );
};

export default ModelUpload;