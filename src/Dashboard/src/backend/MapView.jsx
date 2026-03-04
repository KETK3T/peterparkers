import React, { useState, useEffect, } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./MapView.css"
import {MapContainer, TileLayer, Marker, Popup, useMapEvents, Rectangle} from "react-leaflet";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

function DrawRectanglesLeft({ spotStates, markerPosition }) {

  const heightOfSpot = 0.0000207
  const heightOfLine = 0.0000033
  const spacing = heightOfSpot + heightOfLine

  const DistanceFromMarker2SpotTL = 0.00010035
  const DistanceFromMarker2SpotBR = 0.00005655

  // ✅ Split array in half
  const mid = Math.ceil(spotStates.length / 2)
  const upper = spotStates.slice(0, mid).reverse()
  const lower = spotStates.slice(mid)

  const rectangles = []

  // 🔽 LOWER (go downward)
  lower.forEach((state, index) => {
    const offset = index * spacing

    const bounds = [
      [markerPosition[0] - offset, markerPosition[1] - DistanceFromMarker2SpotTL],
      [markerPosition[0] - heightOfSpot - offset, markerPosition[1] - DistanceFromMarker2SpotBR]
    ]

    rectangles.push(
      <Rectangle
        key={`lower-${index}`}
        bounds={bounds}
        pathOptions={{
          color: state === "taken" ? "red" : "green",
          weight: 2
        }}
      />
    )
  })

  // 🔼 UPPER (go upward)
upper.forEach((state, index) => {

  const offset = (index + 1) * spacing   // 👈 start one spot above

  const bounds = [
    [markerPosition[0] + offset, markerPosition[1] - DistanceFromMarker2SpotTL],
    [markerPosition[0] - heightOfSpot + offset, markerPosition[1] - DistanceFromMarker2SpotBR]
  ]

  rectangles.push(
    <Rectangle
      key={`upper-${index}`}
      bounds={bounds}
      pathOptions={{
        color: state === "taken" ? "red" : "green",
        weight: 2
      }}
    />
  )
})


  return <>{rectangles}</>
}

function DrawRectanglesRight({ spotStates, markerPosition }) {
  const heightOfSpot = 0.0000207
  const heightOfLine = 0.0000033
  const spacing = heightOfSpot + heightOfLine

  const DistanceFromMarker2SpotTL = 0.00010035
  const DistanceFromMarker2SpotBR = 0.00005655

  // ✅ Split array in half
  const mid = Math.ceil(spotStates.length / 2)
  const upper = spotStates.slice(0, mid).reverse()
  const lower = spotStates.slice(mid)

  const rectangles = []

  // 🔽 LOWER (go downward)
  lower.forEach((state, index) => {
    const offset = index * spacing

    const bounds = [
      [markerPosition[0] - offset, markerPosition[1] + DistanceFromMarker2SpotTL],
      [markerPosition[0] - heightOfSpot - offset, markerPosition[1] + DistanceFromMarker2SpotBR]
    ]

    rectangles.push(
      <Rectangle
        key={`lower-${index}`}
        bounds={bounds}
        pathOptions={{
          color: state === "taken" ? "red" : "green",
          weight: 2
        }}
      />
    )
  })

  // 🔼 UPPER (go upward)
upper.forEach((state, index) => {

  const offset = (index + 1) * spacing   // 👈 start one spot above

  const bounds = [
    [markerPosition[0] + offset, markerPosition[1] + DistanceFromMarker2SpotTL],
    [markerPosition[0] - heightOfSpot + offset, markerPosition[1] + DistanceFromMarker2SpotBR]
  ]

  rectangles.push(
    <Rectangle
      key={`upper-${index}`}
      bounds={bounds}
      pathOptions={{
        color: state === "taken" ? "red" : "green",
        weight: 2
      }}
    />
  )
})


  return <>{rectangles}</>
}

// function LocationMarker() {
//   const [position, setPosition] = useState(null);
//   const map = useMapEvents({});
//
//   useEffect(() => {
//     async function fetchState() {
//       try {
//         const response = await fetch("http://localhost:5000/state");
//         const data = await response.json();
//
//         const newPosition = [data.latitude, data.longitude];
//
//         setPosition(newPosition);
//         map.flyTo(newPosition, map.getZoom());
//
//       } catch (error) {
//         console.error("Error fetching state:", error);
//       }
//     }
//
//     const interval = setInterval(fetchState, 500);
//
//     return () => clearInterval(interval);
//   }, [map]);

//   return position ? (
//     <Marker position={position}>
//       <Popup>
//         Lat: {position[0]} <br />
//         Lon: {position[1]}
//       </Popup>
//     </Marker>
//   ) : null;
// }


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
    const [livePoint, setLivePoint] = useState(null)

    useEffect(() => {
      const interval = setInterval(async () => {
        const res = await fetch("http://localhost:8765/state")
        const data = await res.json()
        const last = data[data.length -1]
        console.log("raw: ", data )
        setLivePoint([last.lat, last.long])
      }, 1000) // adjust poll interval as needed

      return () => clearInterval(interval)
    }, [])

    useEffect(() => {
      console.log("livePoint updated:", livePoint)
    }, [livePoint])

    const leftSide = ['empty', 'taken', 'empty']
    const rightSide = ['taken', 'empty', 'taken']
    return(
    <MapContainer center = {[32.72591983,-97.1126803]} zoom = {18} minZoom= {18} maxZoom= {22} scrollWheelZoom={true}
      style={{ height: '100%', width: '100%' }}>
        <TileLayer
            attribution= '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="/tiles2/{z}/{x}/{y}.png"
            minZoom= {18}
            maxZoom= {22}
            maxNativeZoom= {22}
        />
{/*         <LocationMarker/> */}

    {/* Only render when we have live data */}
          {livePoint && (
            <>
              <Marker position={livePoint}>
                <Popup>Live Car</Popup>
              </Marker>

              <DrawRectanglesLeft
                spotStates={leftSide}
                markerPosition={livePoint}
              />

              <DrawRectanglesRight
                spotStates={rightSide}
                markerPosition={livePoint}
              />
            </>
          )}

    </MapContainer>

    );
}
