// src/components/common/SimpleFileSelector.jsx
import React, { useState } from 'react';

const SimpleFileSelector = ({ label, value, onChange, accept }) => {
  // State to track the displayed path
  const [displayPath, setDisplayPath] = useState(value || '');
  
  // Handle file selection
  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const filePath = e.target.files[0].path || e.target.files[0].name;
      setDisplayPath(filePath);
      
      // Call the onChange handler with the path
      if (onChange) {
        onChange(filePath);
      }
    }
  };
  
  // Allow manual path entry
  const handleInputChange = (e) => {
    const newPath = e.target.value;
    setDisplayPath(newPath);
    
    if (onChange) {
      onChange(newPath);
    }
  };
  
  return (
    <div className="file-selector form-group">
      <label>{label || 'Select File'}:</label>
      <div className="file-input-container">
        <input
          type="text"
          value={displayPath}
          onChange={handleInputChange}
          placeholder="Enter file path or browse"
          className="file-path-input"
        />
        <label className="file-browse-button">
          Browse
          <input
            type="file"
            accept={accept}
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
        </label>
      </div>
    </div>
  );
};

export default SimpleFileSelector;