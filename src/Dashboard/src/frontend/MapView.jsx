import './MapView.css'
import { MapContainer, TileLayer, Marker, Popup, Rectangle, useMap } from "react-leaflet"
import 'leaflet/dist/leaflet.css'
import { useEffect, useState, useRef } from "react"
import { snapLongitude, snapLatitude, snapLongitudeBottom } from "./regions"
import 'leaflet-rotate'
import L from "leaflet"
import markerIcon from "leaflet/dist/images/marker-icon.png"

const START_LAT = 32.72626821031165
const STEP = 0.00002465


function snapLatitude_grid(lat) {
  const steps = Math.round((lat - START_LAT) / STEP)
  return START_LAT + steps * STEP
}

function DrawRectanglesLeft({ spotStates, markerPosition, rotation }) {
  const heightOfSpot = 0.0000201
  const heightOfLine = 0.0000043
  const heightOfSpotForUpRight = 0.0000202
  const heightOfLineForUpRight = 0.0000045
  const spacing = heightOfSpot + heightOfLine
  const spacingForUpRight = heightOfSpotForUpRight + heightOfLineForUpRight
  const DistanceFromMarker2SpotTL = 0.00010085
  const DistanceFromMarker2SpotBR = 0.00005655
  const DistanceFromMarker2SpotTLLng = 0.00008185
  const DistanceFromMarker2SpotBRLng = 0.00004455
  const heightOfSpotLng = heightOfSpot * (111000 / 93000)
  const spacingLng = spacing * (111000 / 93000)

  const mid = Math.ceil(spotStates.length / 2)
  const upper = spotStates.slice(0, mid).reverse()
  const lower = spotStates.slice(mid)
  const rectangles = []

  if (rotation === 90) {
    lower.forEach((state, index) => {
      const offset = index * spacingLng
      const bounds = [
        [markerPosition[0] - DistanceFromMarker2SpotTLLng, markerPosition[1] + offset],
        [markerPosition[0] - DistanceFromMarker2SpotBRLng, markerPosition[1] + heightOfSpotLng + offset]
      ]
      rectangles.push(
        <Rectangle key={`lower-${index}`} bounds={bounds}
          pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
      )
    })

    upper.forEach((state, index) => {
      const offset = (index + 1) * spacingLng
      const bounds = [
        [markerPosition[0] - DistanceFromMarker2SpotTLLng, markerPosition[1] - offset],
        [markerPosition[0] - DistanceFromMarker2SpotBRLng, markerPosition[1] + heightOfSpotLng - offset]
      ]
      rectangles.push(
        <Rectangle key={`upper-${index}`} bounds={bounds}
          pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
      )
    })
  } else {
    lower.forEach((state, index) => {
      const offset = index * spacingForUpRight
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
      const offset = (index + 1) * spacingForUpRight
      const bounds = [
        [markerPosition[0] + offset, markerPosition[1] - DistanceFromMarker2SpotTL],
        [markerPosition[0] - heightOfSpot + offset, markerPosition[1] - DistanceFromMarker2SpotBR]
      ]
      rectangles.push(
        <Rectangle key={`upper-${index}`} bounds={bounds}
          pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
      )
    })
  }

  return <>{rectangles}</>
}

function DrawRectanglesRight({ spotStates, markerPosition, rotation }) {
  const heightOfSpot = 0.0000201
  const heightOfLine = 0.0000043
  const heightOfSpotForUpRight = 0.0000202
  const heightOfLineForUpRight = 0.0000045
  const spacing = heightOfSpot + heightOfLine
  const spacingForUpRight = heightOfSpotForUpRight + heightOfLineForUpRight
  const DistanceFromMarker2SpotTL = 0.00009485
  const DistanceFromMarker2SpotBR = 0.00005655
  const DistanceFromMarker2SpotTLLng = 0.00008585
  const DistanceFromMarker2SpotBRLng = 0.00004455
  const heightOfSpotLng = heightOfSpot * (111000 / 93000)
  const spacingLng = spacing * (111000 / 93000)

  const mid = Math.ceil(spotStates.length / 2)
  const upper = spotStates.slice(0, mid).reverse()
  const lower = spotStates.slice(mid)
  const rectangles = []

  if (rotation === 90) {
    lower.forEach((state, index) => {
      const offset = index * spacingLng
      const bounds = [
        [markerPosition[0] + DistanceFromMarker2SpotBRLng, markerPosition[1] + offset],
        [markerPosition[0] + DistanceFromMarker2SpotTLLng, markerPosition[1] + heightOfSpotLng + offset]
      ]
      rectangles.push(
        <Rectangle key={`lower-${index}`} bounds={bounds}
          pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
      )
    })

    upper.forEach((state, index) => {
      const offset = (index + 1) * spacingLng
      const bounds = [
        [markerPosition[0] + DistanceFromMarker2SpotBRLng, markerPosition[1] - offset],
        [markerPosition[0] + DistanceFromMarker2SpotTLLng, markerPosition[1] + heightOfSpotLng - offset]
      ]
      rectangles.push(
        <Rectangle key={`upper-${index}`} bounds={bounds}
          pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
      )
    })
  } else {
    lower.forEach((state, index) => {
      const offset = index * spacingForUpRight
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
      const offset = (index + 1) * spacingForUpRight
      const bounds = [
        [markerPosition[0] + offset, markerPosition[1] + DistanceFromMarker2SpotTL],
        [markerPosition[0] - heightOfSpot + offset, markerPosition[1] + DistanceFromMarker2SpotBR]
      ]
      rectangles.push(
        <Rectangle key={`upper-${index}`} bounds={bounds}
          pathOptions={{ color: state === "taken" ? "red" : "green", weight: 2 }} />
      )
    })
  }

  return <>{rectangles}</>
}

function RotateMap({ bearing }) {
  const map = useMap()
  const currentBearingRef = useRef(0)

  useEffect(() => {
    const start = currentBearingRef.current
    const end = bearing
    const duration = 800  // ms — increase for slower, decrease for faster
    const startTime = performance.now()

    const animate = (now) => {
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)

      // Ease in-out
      const eased = progress < 0.5
        ? 2 * progress * progress
        : -1 + (4 - 2 * progress) * progress

      const current = start + (end - start) * eased
      map.setBearing(current)

      if (progress < 1) {
        requestAnimationFrame(animate)
      } else {
        currentBearingRef.current = end
      }
    }

    requestAnimationFrame(animate)
  }, [bearing, map])

  return null
}

function FollowMarker({ position, following, shouldRecenter, onRecentered, mapReady }) {
  const map = useMap()
  const hasCenteredRef = useRef(false)

  useEffect(() => {
    if (!mapReady || !position || !following) return

    const zoom = Math.max(map.getZoom(), 21)

    // 🔥 Force FIRST real recenter AFTER map is ready
    if (!hasCenteredRef.current) {
      map.setView(position, zoom, { animate: false })
      hasCenteredRef.current = true
      return
    }

    // Normal follow behavior
    map.setView(position, zoom, { animate: true })

  }, [position, following, mapReady, map])

  useEffect(() => {
    if (shouldRecenter && position && mapReady) {
      map.setView(position, 21, { animate: true })
      onRecentered()
    }
  }, [shouldRecenter, position, mapReady, map])

  return null
}

function StopFollowingOnInteract({ setFollowing, recenter }) {
  const map = useMap()
  const userInteractedRef = useRef(false)

  useEffect(() => {
    const markInteracted = () => {
      userInteractedRef.current = true
    }

    const stop = () => {
      // ❌ Ignore fake initial events
      if (!userInteractedRef.current) return

      if (!recenter) {
        setFollowing(false)
      }
    }

    // Detect REAL user interaction
    map.on('mousedown', markInteracted)
    map.on('touchstart', markInteracted)
    map.on('wheel', markInteracted)

    // Stop following only AFTER interaction
    map.on('dragstart', stop)
    map.on('zoomstart', stop)

    return () => {
      map.off('mousedown', markInteracted)
      map.off('touchstart', markInteracted)
      map.off('wheel', markInteracted)

      map.off('dragstart', stop)
      map.off('zoomstart', stop)
    }
  }, [map, recenter, setFollowing])

  return null
}

function MapView() {
  const [livePoint, setLivePoint] = useState(null)
  const [leftSide, setLeftSide] = useState([])
  const [rightSide, setRightSide] = useState([])
  const [rotation, setRotation] = useState(0)
  const [following, setFollowing] = useState(true)
  const [recenter, setRecenter] = useState(false)
  const [mapReady, setMapReady] = useState(false)

  // WebSocket for GPS
useEffect(() => {
  const eventSource = new EventSource("http://localhost:5000/stream")

  eventSource.onmessage = (event) => {

      console.log("RAW EVENT:", event.data) //log to see if update
    try {
      const data = JSON.parse(event.data)

      const lat = data.lat
      const lon = data.lon  // NOTE: your backend uses "lon", not "long"

      console.log("LAT/LON:", lat, lon)

      const isBelow = lat < START_LAT

      const snappedLat = isBelow
        ? snapLatitude(lat)
        : snapLatitude_grid(lat)

      const snappedLong = isBelow
        ? snapLongitudeBottom(lon)
        : snapLongitude(lon)

      setLivePoint([snappedLat, snappedLong])
      setRotation(isBelow ? 90 : 0)

    } catch (err) {
      console.error("Error parsing SSE data:", err)
    }
  }

  eventSource.onerror = (err) => {
    console.error("SSE connection error:", err)
  }

  return () => {
    eventSource.close()
  }
}, [])

  // Poll Flask endpoint every second for parking spot states
  useEffect(() => {
    const mapState = (val) => val === "Car" ? "taken" : "empty"

    const fetchSpots = async () => {
      try {
        const res = await fetch("http://localhost:5001/detections")
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
      zoom={20}
      minZoom={18}
      maxZoom={22}
      rotate={true} 
      bearing={rotation} 
      style={{ height: "100vh", width: "100%" }}
      whenReady={() => setMapReady(true)} 
    >
      <TileLayer
        url="/tiles10/{z}/{x}/{y}.png"
        minZoom={18}
        maxZoom={22}
        maxNativeZoom={22}
      />

    <RotateMap bearing={rotation} />
    <FollowMarker 
      position={livePoint} 
      following={following} 
      shouldRecenter={recenter}
      onRecentered={() => setRecenter(false)}
      mapReady={mapReady}
    />
    <StopFollowingOnInteract setFollowing={setFollowing} following={following} recenter={recenter} />

      {/* Re-center button */}
      <div className="leaflet-top leaflet-left" style={{ marginTop: '80px' }}>
        <div className="leaflet-bar leaflet-control">
          <a
            href="#"
            title="Follow marker"
            role="button"
            onClick={(e) => { 
              e.preventDefault()
              setFollowing(true)
              setRecenter(true)
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '30px',
              height: '30px',
              fontSize: '18px',
              textDecoration: 'none',
              color: following ? '#0078ff' : '#333',
              backgroundColor: 'gray',
            }}
          >
            <img
              src={markerIcon}
              alt="marker"
              style={{
                width: '20px',
                height: '32px',
                objectFit: 'contain'
              }}
            />
          </a>
        </div>
      </div>


      {livePoint && (
        <>
          <Marker position={livePoint}>
          </Marker>

          <DrawRectanglesLeft spotStates={leftSide} markerPosition={livePoint} rotation={rotation} />
          <DrawRectanglesRight spotStates={rightSide} markerPosition={livePoint} rotation={rotation}/>
        </>
      )}

    </MapContainer>
  )
}

export default MapView