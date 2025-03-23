// src/utils/formatters.js
/**
 * Format a date string to a readable format
 * @param {string} dateString - ISO date string
 * @returns {string} Formatted date string
 */
export const formatDate = (dateString) => {
  if (!dateString) return 'N/A';
  
  try {
    const date = new Date(dateString);
    
    // Check if date is valid
    if (isNaN(date.getTime())) {
      return 'N/A';
    }
    
    // Format: "Mar 21, 2025, 14:30"
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit'
    }).format(date);
  } catch (err) {
    console.error('Error formatting date:', err);
    return 'N/A';
  }
};

/**
 * Format a dataset type to a more readable form
 * @param {string} datasetType - Dataset type string (e.g., "breast_cancer")
 * @returns {string} Formatted dataset type
 */
export const formatDatasetType = (datasetType) => {
  if (!datasetType) return 'Unknown';
  
  // Convert underscore case to sentence case
  const words = datasetType.split('_').map(word => 
    word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
  );
  
  return words.join(' ');
};

/**
 * Format a number to a percentage
 * @param {number} value - Value to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted percentage
 */
export const formatPercentage = (value, decimals = 2) => {
  if (value === undefined || value === null) return 'N/A';
  
  try {
    return `${value.toFixed(decimals)}%`;
  } catch (err) {
    console.error('Error formatting percentage:', err);
    return `${value}%`;
  }
};

/**
 * Format a number with commas for thousands
 * @param {number} value - Value to format
 * @returns {string} Formatted number
 */
export const formatNumber = (value) => {
  if (value === undefined || value === null) return 'N/A';
  
  try {
    return value.toLocaleString();
  } catch (err) {
    console.error('Error formatting number:', err);
    return `${value}`;
  }
};