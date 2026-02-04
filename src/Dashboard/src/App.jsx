import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import team_logo from "./pictures/peter_parkers_logo.jpg";
import './App.css';
import Detailed from './components/Detailed.jsx';

function AppContent() {
  const location = useLocation();

  const hideHeader =
    location.pathname === "/detailed";

  return (
    <div className="app-container">
      {!hideHeader && (
        <header className="app-header">
          <div className="header-logo">
            <img src={team_logo} alt="Team Logo" className="logo-img" />
          </div>

          <h1 className="header-title">Parking Lot Scanner</h1>

          <nav className="header-nav">
            <Link to="/detailed" className="nav-link">
              Click To View
            </Link>
          </nav>
        </header>
      )}

      {/* Routing Section */}
      <Routes>
        <Route path="/detailed" element={<Detailed />} />
      </Routes>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}
