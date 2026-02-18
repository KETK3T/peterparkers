import React, { useState, useEffect, } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./MapView.css"
import {MapContainer, TileLayer, Marker, Popup, useMapEvents} from "react-leaflet";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

function LocationMarker() {
  const [position, setPosition] = useState(null);
  const map = useMapEvents({});

  useEffect(() => {
    async function fetchState() {
      try {
        const response = await fetch("http://localhost:5000/state");
        const data = await response.json();

        const newPosition = [data.latitude, data.longitude];

        setPosition(newPosition);
        map.flyTo(newPosition, map.getZoom());

      } catch (error) {
        console.error("Error fetching state:", error);
      }
    }

    const interval = setInterval(fetchState, 500);

    return () => clearInterval(interval);
  }, [map]);

  return position ? (
    <Marker position={position}>
      <Popup>
        Lat: {position[0]} <br />
        Lon: {position[1]}
      </Popup>
    </Marker>
  ) : null;
}


/*
function LocationMarker() {
  const [position, setPosition] = useState(null);
  const [bbox, setBbox] = useState([]);

  const map = useMapEvents({
      // 2. Event that triggers when location is found
      locationfound(e) {

          // Set the marker position
          setPosition(e.latlng);

          // Calculate and set the bounding box for zooming
          setBbox(e.bounds.toBBoxString().split(",").map(Number));

          // Automatically pan the map to the new location
          map.flyTo(e.latlng, map.getZoom());
          console.log("Location found:", e.latlng);
      },
  
  // 3. Event that triggers if location finding fails
      locationerror(e) {
          console.error("Location error:", e.message);
          alert(`Location tracking failed: ${e.message}. Please ensure location services are enabled.`);
      }
    });

useEffect(() => {
    // Starts the browser's location tracking
    map.locate({
        watch: true,     // Key for REAL-TIME tracking
        setView: false,  // We'll use map.flyTo instead
        maxZoom: 16,     // Max zoom level for the map to set
    });

return () => {
        map.stopLocate();
    };
  }, [map]); // Dependency on the map instance

  // 5. Render the Marker at the current position
  return position === null ? null : (
    <Marker position={position}>
      <Popup>
        You are here. <br />
        Map bounds: <br />
        <pre>bbox: {bbox.join(", ")}</pre>
      </Popup>
    </Marker>
  );
}*/

export default function MapView() {
    return(
    <MapContainer center = {[32.7260083,-97.1125303]} zoom = {17} minZoom= {17} maxZoom= {22} scrollWheelZoom={true}
      style={{ height: '100%', width: '100%' }}>
        <TileLayer
            attribution= '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="/tiles/{z}/{x}/{y}.png"
            minZoom= {17}
            maxZoom= {22}
            maxNativeZoom= {22}
        />
        <LocationMarker/>
    </MapContainer>

    );
}
