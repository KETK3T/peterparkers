import 'leaflet/dist/leaflet.css'



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

// 🔥 Find closest longitude
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
