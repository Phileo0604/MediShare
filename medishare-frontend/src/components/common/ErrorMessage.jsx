// src/components/common/ErrorMessage.jsx
import React from 'react';

const ErrorMessage = ({ message, onRetry }) => {
  return (
    <div className="error-container">
      <div className="error-message">
        <p><strong>Error:</strong> {message}</p>
        {onRetry && (
          <button 
            className="btn btn-secondary btn-sm"
            onClick={onRetry}
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;