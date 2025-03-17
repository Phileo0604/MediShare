// src/components/common/Footer.jsx
import React from 'react';

const Footer = () => {
  return (
    <footer className="app-footer">
      <p>&copy; {new Date().getFullYear()} MediShare - Federated Learning for Medical Applications</p>
    </footer>
  );
};

export default Footer;