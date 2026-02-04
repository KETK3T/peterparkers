import { useNavigate } from "react-router-dom";
import React, { useState, useEffect } from "react";
import "./Detailed.css";
import MapView from "../backend/MapView.jsx";
import axios from "axios";

export default function Detailed() {



  const navigate = useNavigate();

  const [cameraDevices, setCameraDevices] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  // Automatically detect available cameras
  useEffect(() => {
    async function getCameraDevices() {
      try {
        // Ask permission to access the cameras first (important for labels)
        await navigator.mediaDevices.getUserMedia({ video: true });

        const devices = await navigator.mediaDevices.enumerateDevices();
        const foundVideoDevices = devices.filter((d) => d.kind === "videoinput");


        setCameraDevices(activeCams);
        setIsLoading(false);
      } catch (err) {
        console.error("Error accessing or enumerating cameras:", err);
        setIsLoading(false);
      }
    }

    // Check API support
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      console.error("Browser does not support enumerateDevices().");
      setIsLoading(false);
      return;
    }

    getCameraDevices();
  }, []);

   //Part of the Exit Button
  const handleExit = () => {
    const confirmed = window.confirm("Are you sure you want to exit the camera view?");
    if (confirmed) navigate("/");
  };

  // Loading state
  if (isLoading) {
    return <div className="detailed-container">LOADING....</div>;
  }



//THE INTERFACE SETUP
  return (
    <div className="detailed-container">


      {/* LEFT SIDE OF THE INTERFACE */}
      <div className="map-side">
        <div className="map-section">
          <MapView />
        </div>

        <div className="exit-section">
          <button onClick={handleExit} className="exit-button">
            Exit
          </button>
        </div>


      </div>

      {/* RIGHT SIDE OF THE INTERFACE */}
      <div className="camera-side">
         <div className="left-camera-box">
                <img
                src="http://localhost:8000/video/left"
                alt="Live Camera"
                className="camera-feed"
              />
        </div>

        <div className="main-camera-box">
            <img
                src="http://localhost:8000/video/front"
                alt="Live Camera"
                className="camera-feed"
              />
        </div>

        <div className="right-camera-box">
                <img
                src="http://localhost:8000/video/right"
                alt="Live Camera"
                className="camera-feed"
              />
        </div>
        </div>
      </div>
  );
}
