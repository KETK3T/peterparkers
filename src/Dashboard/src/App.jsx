import React from "react";
import MapView from "./frontend/MapView.jsx";
import "./App.css";
import team_logo from "./assets/peter_parkers_logo.jpg";

export default function App() {

  return (


    <div className="app-container">
        <header className="app-header">
            <div className="header-logo">
                <img src={team_logo} alt="Team Logo" className="logo-img" />
                <h1 className="header-title">Parking Lot Scanner</h1>
            </div>
        </header>

        {/*Map View */}
        <div className="map-section">
          <MapView/>
        </div>
    </div>
  );
}
