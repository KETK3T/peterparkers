import { MapContainer, TileLayer, Marker, Popup, Rectangle } from "react-leaflet"
import 'leaflet/dist/leaflet.css'
import MarkerClusterGroup from "react-leaflet-cluster"
import { useEffect, useState } from "react"

const START_LONG = -97.11141225279618
const STEP_LONG = 0.00002949

const LONG_0 = -97.11215105459435
const LONG_1 = -97.11225975469435
const LONG_2 = -97.11280605469435
const LONG_3 = -97.11292905469435

export function snapLongitudeBottom(long) {
  // In region 1 (gap between Long2 and Long3) - free placement
  if (long <= LONG_2 && long >= LONG_3) {
    return long
  }

  // In region 2 (gap between Long0 and Long1) - free placement
  if (long <= LONG_0 && long >= LONG_1) {
    return long
  }

  // After Long3 (more negative than Long3) - start from Long3
  if (long < LONG_3) {
    const steps = Math.round((long - LONG_3) / STEP_LONG)
    return LONG_3 + steps * STEP_LONG
  }

  // Between Long1 and Long2 - start from Long1
  if (long < LONG_1 && long > LONG_2) {
    const steps = Math.round((long - LONG_1) / STEP_LONG)
    return LONG_1 + steps * STEP_LONG
  }

  // Default - before Long0, start from START_LONG
  const steps = Math.round((long - START_LONG) / STEP_LONG)
  return START_LONG + steps * STEP_LONG
}

const REGION_LONGITUDES = [
  -97.11408752,
  -97.11388841,
  -97.11368841,
  -97.11348541,
  -97.11328241,
  -97.11307901,
  -97.11287701,
  -97.11264701,
  -97.11242701,
  -97.11220501,
  -97.11200301,
  -97.11180001,
  -97.11159699
]

const REGION_LATITUDES = [
  32.72536936,
  32.72554031,
  32.72571126,
  32.72588221,
  32.72605316,
  32.72622411
]

export function snapLongitude(long) {
  let closest = REGION_LONGITUDES[0]
  let minDiff = Math.abs(long - closest)

  for (let i = 1; i < REGION_LONGITUDES.length; i++) {
    const diff = Math.abs(long - REGION_LONGITUDES[i])
    if (diff < minDiff) {
      minDiff = diff
      closest = REGION_LONGITUDES[i]
    }
  }

  return closest
}

export function snapLatitude(lat) {
  let closest = REGION_LATITUDES[0]
  let minDiff = Math.abs(lat - closest)

  for (let i = 1; i < REGION_LATITUDES.length; i++) {
    const diff = Math.abs(lat - REGION_LATITUDES[i])
    if (diff < minDiff) {
      minDiff = diff
      closest = REGION_LATITUDES[i]
    }
  }

  return closest
}