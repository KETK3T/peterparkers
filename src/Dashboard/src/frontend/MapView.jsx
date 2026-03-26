import React, { useState, useEffect, } from "react";
import "leaflet/dist/leaflet.css";
import "./MapView.css"
import {MapContainer, TileLayer, Marker, Popup, Rectangle} from "react-leaflet";
import { snapLongitude } from "./regions.jsx"


const START_LAT = 32.72626821031165
const STEP = 0.00002487

function snapLatitude(lat) {
  const steps = Math.round((lat - START_LAT) / STEP)
  return START_LAT + steps * STEP
}

function DrawRectanglesLeft({ spotStates, markerPosition }) {
  const heightOfSpot = 0.0000207
  const heightOfLine = 0.0000037
  const spacing = heightOfSpot + heightOfLine
  const DistanceFromMarker2SpotTL = 0.00010085
  const DistanceFromMarker2SpotBR = 0.00005655

  const mid = Math.ceil(spotStates.length / 2)
  const upper = spotStates.slice(0, mid).reverse()
  const lower = spotStates.slice(mid)
  const rectangles = []

  lower.forEach((state, index) => {
    const offset = index * spacing
    const bounds = [
      [markerPosition[0] - offset, markerPosition[1] - DistanceFromMarker2SpotTL],
      [markerPosition[0] - heightOfSpot - offset, markerPosition[1] - DistanceFromMarker2SpotBR]
    ]
    rectangles.push(
      <Rectangle key={`lower-${index}`} bounds={bounds}
        pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
    )
  })

  upper.forEach((state, index) => {
    const offset = (index + 1) * spacing
    const bounds = [
      [markerPosition[0] + offset, markerPosition[1] - DistanceFromMarker2SpotTL],
      [markerPosition[0] - heightOfSpot + offset, markerPosition[1] - DistanceFromMarker2SpotBR]
    ]
    rectangles.push(
      <Rectangle key={`upper-${index}`} bounds={bounds}
        pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
    )
  })

  return <>{rectangles}</>
}

function DrawRectanglesRight({ spotStates, markerPosition }) {
  const heightOfSpot = 0.0000207
  const heightOfLine = 0.0000040
  const spacing = heightOfSpot + heightOfLine
  const DistanceFromMarker2SpotTL = 0.00009485
  const DistanceFromMarker2SpotBR = 0.00005655

  const mid = Math.ceil(spotStates.length / 2)
  const upper = spotStates.slice(0, mid).reverse()
  const lower = spotStates.slice(mid)
  const rectangles = []

  lower.forEach((state, index) => {
    const offset = index * spacing
    const bounds = [
      [markerPosition[0] - offset, markerPosition[1] + DistanceFromMarker2SpotTL],
      [markerPosition[0] - heightOfSpot - offset, markerPosition[1] + DistanceFromMarker2SpotBR]
    ]
    rectangles.push(
      <Rectangle key={`lower-${index}`} bounds={bounds}
        pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
    )
  })

  upper.forEach((state, index) => {
    const offset = (index + 1) * spacing
    const bounds = [
      [markerPosition[0] + offset, markerPosition[1] + DistanceFromMarker2SpotTL],
      [markerPosition[0] - heightOfSpot + offset, markerPosition[1] + DistanceFromMarker2SpotBR]
    ]
    rectangles.push(
      <Rectangle key={`upper-${index}`} bounds={bounds}
        pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
    )
  })

  return <>{rectangles}</>
}

export default function MapView() {
    const [livePoint, setLivePoint] = useState(null)
  const [leftSide, setLeftSide] = useState([])
  const [rightSide, setRightSide] = useState([])

  // WebSocket for GPS
  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8765")
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      const snappedLat = snapLatitude(data.lat)
      const snappedLong = snapLongitude(data.long)
      setLivePoint([snappedLat, snappedLong])
    }
    return () => socket.close()
  }, [])

  // Poll Flask endpoint every second for parking spot states
  useEffect(() => {
    const mapState = (val) => val === "Car" ? "taken" : "empty"

    const fetchSpots = async () => {
      try {
        const res = await fetch("http://localhost:5000/detections")
        const data = await res.json()
        setLeftSide(data.spots.Left.map(mapState))
        setRightSide(data.spots.Right.map(mapState))
      } catch (err) {
        console.error("Failed to fetch parking spots:", err)
      }
    }

    fetchSpots() // fetch immediately on mount
    const interval = setInterval(fetchSpots, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <MapContainer
      center={[32.72591944383129, -97.11268490981709]}
      zoom={18}
      minZoom={19}
      maxZoom={22}
      style={{ height: "100vh", width: "100%" }}
    >
      <TileLayer
        url="/tiles5/{z}/{x}/{y}.png"
        minZoom={18}
        maxZoom={22}
        maxNativeZoom={22}
      />

      {livePoint && (
        <>
          <Marker position={livePoint}>
            <Popup>Live Car</Popup>
          </Marker>

          <DrawRectanglesLeft spotStates={leftSide} markerPosition={livePoint} />
          <DrawRectanglesRight spotStates={rightSide} markerPosition={livePoint} />
        </>
      )}

    </MapContainer>
  )
}
