import { useNavigate } from "react-router-dom";
import React, { useState, useEffect } from "react";
import "./Detailed.css";
import MapView from "../backend/MapView.jsx";
import axios from "axios";

export default function Detailed() {

  const leftCameraSrc = "http://localhost:8000/videoleft";
  const rightCameraSrc = "http://localhost:8000/videoright";
  const mainCameraSrc = "http://localhost:8000/videofront";

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

        console.log("--- Detected Cameras ---");
        foundVideoDevices.forEach((d, i) => {
          console.log(`Camera ${i + 1}: ${d.label || "Unnamed Camera"} (${d.deviceId})`);
        });
        console.log("------------------------");

        // Sort or label cameras
        const frontCam =
          foundVideoDevices.find((d) => d.label.toLowerCase().includes("front")) ||
          foundVideoDevices[0];
        const leftCam =
          foundVideoDevices.find((d) => d.label.toLowerCase().includes("left")) ||
          foundVideoDevices[1];
        const rightCam =
          foundVideoDevices.find((d) => d.label.toLowerCase().includes("right")) ||
          foundVideoDevices[2];

        // Filter undefined/null values
        const activeCams = [frontCam, leftCam, rightCam].filter(Boolean);

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
    const confirmed = window.confirm("Are you sure you want to exit the Detailed View?");
    if (confirmed) navigate("/");
  };

  // Loading state
  if (isLoading) {
    return <div className="detailed-container">LOADING....</div>;
  }


  // Assign the first three cameras (if available)
  const frontCam = cameraDevices[0];
  const leftCam = cameraDevices[1];
  const rightCam = cameraDevices[2];



//THE INTERFACE SETUP
  return (
    <div className="detailed-container">
      {/* LEFT SIDE OF THE INTERFACE */}
      <div className="left-side">
        <div className="map-section">
          <MapView />
        </div>

        {/* EXIT BUTTON */}
        <div className="exit-section">
          <button onClick={handleExit} className="exit-button">
            Exit
          </button>
        </div>


      </div>

      {/* RIGHT SIDE OF THE INTERFACE */}
      <div className="right-side">
        {/*MAIN CAMERA */}
        <div className="main-camera-container">
            <img src={mainCameraSrc} className="camera-feed" />
        </div>

        {/* LOWER CAMERAS */}
        <div className="lower-cameras">
          {/* LEFT CAMERA */}
          <div className="small-camera-box">
                <img src={leftCameraSrc} className="camera-feed" />
            </div>

          {/* RIGHT CAMERA */}
          <div className="small-camera-box">
                <img src={rightCameraSrc} className="camera-feed" />
          </div>
        </div>
      </div>
    </div>
  );
}
