import { useNavigate } from "react-router-dom";
import React, { useState, useEffect } from "react";
import MapView from "../backend/MapView.jsx";
import "./Simple.css";
import axios from "axios"

export default function Simple() {
const navigate = useNavigate();

  // We only need to know if we've attempted to load the camera
  const [isCameraChecked, setIsCameraChecked] = useState(false);
  const [cameraError, setCameraError] = useState(null);

  {/* Python running processor */}
  const [camera, setCamera] = useState("Running Python test...");

  const fetchAPI = async () => {
    const response = await axios.get("http://localhost:8000/api/users");
    console.log(response.data.users);
    setCamera(response.data.users);
    setIsCameraChecked(true);
  }

useEffect(() => {
    fetchAPI()

   }, []);

  const handleExit = () => {
    const isConfirmed = window.confirm("Are you sure you want to exit the Simplified View?");
    if (isConfirmed) {
      navigate("/");
    }
  };



  // Show a loading state while waiting for the check to complete
  if (!isCameraChecked) {
      return <div className="simple-container">Loading camera device...</div>;
  }


  return (
    <div className="simple-container">
      {/* Left Side - Map View */}
      <div className="left-side">
        <div className="map-section">
          <MapView/>
        </div>
        <div className="exit-section">
          <button onClick={handleExit} className="exit-button">
            Exit
          </button>
        </div>
      </div>

      {/* Right Side - Camera View */}
      <div className="right-side">
        <div className="camera-box">
          <div className="camera-label">CAMERA VIEW</div>
              <img
                src="http://localhost:8000/video/front"
                alt="Live Camera"
                className="camera-feed"
              />
        </div>
      </div>
    </div>
  );
}