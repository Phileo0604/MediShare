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
      return date.toLocaleString();
    } catch (err) {
      console.error('Error formatting date:', err);
      return dateString;
    }
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