import React, { useRef, useEffect, useState } from "react";
import "./Camera.css";

function RightCamera({ deviceId, label }) {
  const videoRef = useRef(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let stream;

    const startStream = async () => {
      try {
        // Always enumerate devices again to ensure we get the right IDs
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter((d) => d.kind === "videoinput");
        const target = videoDevices.find((d) => d.deviceId === deviceId);

        if (!target) {
          setError(`${label}: Device not found or disconnected.`);
          setIsLoading(false);
          return;
        }

        // Request specific camera stream by deviceId with unique groupId constraint
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: target.deviceId },
            // Optional: ensure the browser doesn't reuse the same camera
            advanced: [{ groupId: target.groupId }],
          },
          audio: false,
        });

        // Attach the stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        setIsLoading(false);
      } catch (err) {
        console.error(`Error starting ${label}:`, err);
        setError(`Failed to start ${label}: ${err.message}`);
        setIsLoading(false);
      }
    };

    startStream();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      }
    };
  }, [deviceId, label]);

  return (
    <div className="camera-feed-container">
      {isLoading ? (
        <p className="camera-loading-message">🎥 Loading {label}...</p>
      ) : error ? (
        <p className="camera-error-message">🛑 {error}</p>
      ) : (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="camera-video-element"
        />
      )}
    </div>
  );
}

export default RightCamera;
