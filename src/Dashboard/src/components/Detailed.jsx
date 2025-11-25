import { useNavigate } from "react-router-dom";
import React, { useState, useEffect } from "react";
import "./Detailed.css";
// import MainCamera from "../backend/MainCamera.jsx";
// import LeftCamera from "../backend/LeftCamera.jsx";
// import RightCamera from "../backend/RightCamera.jsx";
import MapView from "../backend/MapView.jsx";
import axios from "axios";

export default function Detailed() {

  const leftCameraSrc = "http://localhost:8000/video_left";
  const rightCameraSrc = "http://localhost:8000/video_right";
  const mainCameraSrc = "http://localhost:8000/video_main";

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

  const handleExit = () => {
    const confirmed = window.confirm("Are you sure you want to exit the Detailed View?");
    if (confirmed) navigate("/");
  };

  // Loading state
  if (isLoading) {
    return <div className="detailed-container">Detecting available cameras...</div>;
  }


  // Assign the first three cameras (if available)
  const frontCam = cameraDevices[0];
  const leftCam = cameraDevices[1];
  const rightCam = cameraDevices[2];

  return (
    <div className="detailed-container">
      {/* LEFT SIDE — MAP */}
      <div className="left-side">
        <div className="map-section">
          <MapView />
        </div>
      </div>

      {/* RIGHT SIDE — CAMERAS */}
      <div className="right-side">
        {/* FRONT CAMERA */}
        <div className="main-camera-container">
          <div className="camera-label">
            FRONT CAMERA {frontCam?.label ? `(${frontCam.label})` : ""}
          </div>
            <img src={mainCameraSrc} className="camera-feed" />
        </div>

        {/* LOWER CAMERAS */}
        <div className="lower-cameras">
          {/* LEFT CAMERA */}
          <div className="small-camera-box">
            <div className="camera-label">
              LEFT CAMERA {leftCam?.label ? `(${leftCam.label})` : ""}
            </div>
                <img src={leftCameraSrc} className="camera-feed" />
          </div>

          {/* RIGHT CAMERA */}
          <div className="small-camera-box">
            <div className="camera-label">
              RIGHT CAMERA {rightCam?.label ? `(${rightCam.label})` : ""}
            </div>
                <img src={rightCameraSrc} className="camera-feed" />
          </div>
        </div>

        {/* EXIT BUTTON */}
        <div className="exit-section">
          <button onClick={handleExit} className="exit-button">
            Exit
          </button>
        </div>
      </div>
    </div>
  );
}
