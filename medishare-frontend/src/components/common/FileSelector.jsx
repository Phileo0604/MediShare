// src/components/common/FileSelector.jsx
import React, { useRef, useState } from 'react';

const FileSelector = ({ 
  value, 
  onChange, 
  label = "Select File", 
  accept = "*", 
  name = "file",
  placeholder = "No file selected",
  className = ""
}) => {
  const [fileName, setFileName] = useState(value || "");
  const fileInputRef = useRef(null);

  const handleClick = () => {
    // Programmatically click the hidden file input
    fileInputRef.current.click();
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    
    if (file) {
      // Set the file name for display
      setFileName(file.name);
      
      // Call the onChange handler with the file path
      // In a real application, you might want to use the File object itself
      // or upload it to the server and use the returned path
      onChange({
        target: {
          name: name,
          value: file.name // In real usage, this would be the server path
        }
      });
    }
  };

  return (
    <div className={`file-selector ${className}`}>
      <div className="file-selector-input">
        <input
          type="text"
          value={fileName || value || ""}
          placeholder={placeholder}
          readOnly
          onClick={handleClick}
        />
        <button
          type="button"
          className="btn btn-secondary file-select-button"
          onClick={handleClick}
        >
          {label}
        </button>
      </div>
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={handleChange}
      />
    </div>
  );
};

export default FileSelector;