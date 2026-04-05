import { createHandTracker } from './lib/handTracker'
import {
  applyReleaseImpulse,
  createClothState,
  defaultClothConfig,
  ensureMeshInitialized,
  releaseDrag,
  resetDisplacement,
  setDragTarget,
  stepClothSimulation,
  tryStartDrag,
  type ClothConfig,
  type ClothState,
} from './sim/cloth'
import './style.css'

const app = document.querySelector<HTMLDivElement>('#app')

if (!app) {
  throw new Error('App root not found')
}

app.innerHTML = `
  <div class="camera-shell">
    <video id="camera" autoplay playsinline muted></video>
    <canvas id="distortion"></canvas>
    <canvas id="overlay"></canvas>
  </div>

  <h2 id="camera-error" hidden></h2>

  <div class="controls">
    <h2>Drag nodes with mouse/fist. R: reset mesh, T: toggle overlay, S: toggle pinned edges</h2>
    <div class="control-actions">
      <button id="upload-bg" type="button">Upload BG</button>
      <input id="bg-file" type="file" accept="image/*" hidden>
    </div>
  </div>
`

const video = document.querySelector<HTMLVideoElement>('#camera')
const distortion = document.querySelector<HTMLCanvasElement>('#distortion')
const overlay = document.querySelector<HTMLCanvasElement>('#overlay')
const errorEl = document.querySelector<HTMLParagraphElement>('#camera-error')
const cameraShell = document.querySelector<HTMLDivElement>('.camera-shell')
const uploadBgButton = document.querySelector<HTMLButtonElement>('#upload-bg')
const bgFileInput = document.querySelector<HTMLInputElement>('#bg-file')

if (overlay) {
  // Disable default 300x150 backing size until we compute the real viewport size.
  overlay.width = 0
  overlay.height = 0
}

if (distortion) {
  distortion.width = 0
  distortion.height = 0
}

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
]

const HAND_CLOSED_THRESHOLD = 0.58
const TIP_IDS = [4, 8, 12, 16, 20]
const MCP_IDS = [2, 5, 9, 13, 17]
const MESH_ROWS = 14
const MESH_COLS = 18
const DETECTION_INTERVAL_MS = 66
const HAND_DETECTION_ERROR_COOLDOWN_MS = 1000
const HAND_DETECTION_ERROR_LOG_INTERVAL_MS = 2000
const CLOTH_LINE_COLOR = 'rgba(80, 255, 120, 0.9)'
const CLOTH_POINT_COLOR = 'rgba(30, 200, 255, 0.95)'
const CAMERA_ASPECT_RATIO = 16 / 9
const CAMERA_MAX_WIDTH_PX = 1100
const CAMERA_VIEWPORT_WIDTH_RATIO = 0.96
const CAMERA_MIN_WIDTH_PX = 320
const OVERLAY_DEBUG = false
const OVERLAY_DEBUG_INTERVAL_MS = 900
const POINT_GRAB_RADIUS = 18
const HAND_GRAB_RADIUS_SCALE = 1.4
const TRIANGLE_SEAM_OVERLAP_PX = 0.65
const WARP_SUBDIVISIONS = 2
const MOTION_SAMPLE_MIN_DT_MS = 8
const RELEASE_IMPULSE_FRAME_MS = 16.6667
const RELEASE_IMPULSE_MAX = 120
const MOUSE_RELEASE_IMPULSE_SCALE = 0.95
const HAND_RELEASE_IMPULSE_SCALE = 0.85

const frameBuffer = document.createElement('canvas')
const frameBufferCtx = frameBuffer.getContext('2d')
const clothConfig: ClothConfig = { ...defaultClothConfig }

type Landmark2D = { x: number; y: number }

type CoverMapping = {
  renderWidth: number
  renderHeight: number
  offsetX: number
  offsetY: number
}

type Size2D = {
  width: number
  height: number
}

type MotionSample = {
  x: number
  y: number
  tMs: number
}

function pointDistance(a: Landmark2D, b: Landmark2D): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function isHandClosed(landmarks: Landmark2D[]): boolean {
  const wrist = landmarks[0]
  const mcp = landmarks[9]
  if (!wrist || !mcp) {
    return false
  }

  const palmSize = pointDistance(mcp, wrist)
  if (palmSize < 1e-6) {
    return false
  }

  let sum = 0
  let count = 0
  for (let i = 0; i < TIP_IDS.length; i += 1) {
    const tip = landmarks[TIP_IDS[i]]
    const base = landmarks[MCP_IDS[i]]
    if (!tip || !base) {
      continue
    }

    sum += pointDistance(tip, base) / palmSize
    count += 1
  }

  if (count === 0) {
    return false
  }

  const meanFold = sum / count
  return meanFold < HAND_CLOSED_THRESHOLD
}

function handCursorPoint(landmarks: Landmark2D[]): Landmark2D | null {
  const p0 = landmarks[0]
  const p9 = landmarks[9]
  if (!p0 || !p9) {
    return null
  }

  return {
    x: (p0.x + p9.x) * 0.5,
    y: (p0.y + p9.y) * 0.5,
  }
}

function getCoverMapping(sourceWidth: number, sourceHeight: number, targetWidth: number, targetHeight: number): CoverMapping {
  const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight)
  const renderWidth = sourceWidth * scale
  const renderHeight = sourceHeight * scale
  const offsetX = (targetWidth - renderWidth) * 0.5
  const offsetY = (targetHeight - renderHeight) * 0.5

  return { renderWidth, renderHeight, offsetX, offsetY }
}

function mapNormalizedToCover(x: number, y: number, mapping: CoverMapping): Landmark2D {
  return {
    x: mapping.offsetX + x * mapping.renderWidth,
    y: mapping.offsetY + y * mapping.renderHeight,
  }
}

function readElementSize(el: HTMLElement | null): Size2D {
  if (!el) {
    return { width: 0, height: 0 }
  }

  const rect = el.getBoundingClientRect()
  return {
    width: Math.round(rect.width) || el.clientWidth || el.offsetWidth,
    height: Math.round(rect.height) || el.clientHeight || el.offsetHeight,
  }
}

function getViewportFallbackSize(): Size2D {
  const viewportWidth = Math.max(window.innerWidth || 0, CAMERA_MIN_WIDTH_PX)
  const width = Math.round(
    Math.min(
      CAMERA_MAX_WIDTH_PX,
      Math.max(CAMERA_MIN_WIDTH_PX, viewportWidth * CAMERA_VIEWPORT_WIDTH_RATIO),
    ),
  )

  return {
    width,
    height: Math.round(width / CAMERA_ASPECT_RATIO),
  }
}

let shellFallbackMinHeightApplied = false

function applyShellFallbackMinHeight(shellSize: Size2D): Size2D {
  if (!cameraShell) {
    return shellSize
  }

  if (shellSize.width > 10 && shellSize.height > 10) {
    if (shellFallbackMinHeightApplied) {
      cameraShell.style.minHeight = ''
      shellFallbackMinHeightApplied = false
    }
    return shellSize
  }

  const fallback = shellSize.width > 10
    ? {
        width: shellSize.width,
        height: Math.round(shellSize.width / CAMERA_ASPECT_RATIO),
      }
    : getViewportFallbackSize()

  if (fallback.height > 10) {
    cameraShell.style.minHeight = `${fallback.height}px`
    shellFallbackMinHeightApplied = true
  }

  return readElementSize(cameraShell)
}

function resolveOverlaySize(): Size2D {
  const initialShellSize = readElementSize(cameraShell)
  const shellSize = applyShellFallbackMinHeight(initialShellSize)

  if (shellSize.width > 0 && shellSize.height > 0) {
    return shellSize
  }

  if (initialShellSize.width > 0) {
    return {
      width: initialShellSize.width,
      height: Math.round(initialShellSize.width / CAMERA_ASPECT_RATIO),
    }
  }

  const overlaySize = readElementSize(overlay)
  if (overlaySize.width > 0 && overlaySize.height > 0) {
    return overlaySize
  }

  const videoSize = readElementSize(video)
  if (videoSize.width > 0 && videoSize.height > 0) {
    return videoSize
  }

  const sourceWidth = video?.videoWidth ?? 0
  const sourceHeight = video?.videoHeight ?? 0
  if (sourceWidth > 0 && sourceHeight > 0) {
    if (shellSize.width > 0) {
      return {
        width: shellSize.width,
        height: Math.round((shellSize.width * sourceHeight) / sourceWidth),
      }
    }

    if (shellSize.height > 0) {
      return {
        width: Math.round((shellSize.height * sourceWidth) / sourceHeight),
        height: shellSize.height,
      }
    }

    return {
      width: sourceWidth,
      height: sourceHeight,
    }
  }

  return getViewportFallbackSize()
}

function resizeOverlayToVideo(): void {
  if (!overlay && !distortion) {
    return
  }

  const { width, height } = resolveOverlaySize()
  if (width <= 0 || height <= 0) {
    return
  }

  if (overlay && (overlay.width !== width || overlay.height !== height)) {
    overlay.width = width
    overlay.height = height
  }

  if (distortion && (distortion.width !== width || distortion.height !== height)) {
    distortion.width = width
    distortion.height = height
  }
}

function mapClientToCanvas(clientX: number, clientY: number): Landmark2D | null {
  if (!overlay) {
    return null
  }

  const rect = overlay.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0 || overlay.width <= 0 || overlay.height <= 0) {
    return null
  }

  return {
    x: ((clientX - rect.left) / rect.width) * overlay.width,
    y: ((clientY - rect.top) / rect.height) * overlay.height,
  }
}

function getCoverSourceCrop(sourceWidth: number, sourceHeight: number, targetWidth: number, targetHeight: number): {
  sx: number
  sy: number
  sw: number
  sh: number
} {
  const sourceAspect = sourceWidth / Math.max(sourceHeight, 1)
  const targetAspect = targetWidth / Math.max(targetHeight, 1)

  if (sourceAspect > targetAspect) {
    const sw = sourceHeight * targetAspect
    return {
      sx: (sourceWidth - sw) * 0.5,
      sy: 0,
      sw,
      sh: sourceHeight,
    }
  }

  const sh = sourceWidth / targetAspect
  return {
    sx: 0,
    sy: (sourceHeight - sh) * 0.5,
    sw: sourceWidth,
    sh,
  }
}

function drawVideoFrameToBuffer(width: number, height: number): boolean {
  if (!video || !frameBufferCtx || video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) {
    return false
  }

  if (frameBuffer.width !== width || frameBuffer.height !== height) {
    frameBuffer.width = width
    frameBuffer.height = height
  }

  const { sx, sy, sw, sh } = getCoverSourceCrop(video.videoWidth, video.videoHeight, width, height)
  frameBufferCtx.save()
  frameBufferCtx.clearRect(0, 0, width, height)
  frameBufferCtx.translate(width, 0)
  frameBufferCtx.scale(-1, 1)
  frameBufferCtx.drawImage(video, sx, sy, sw, sh, 0, 0, width, height)
  frameBufferCtx.restore()
  return true
}

function drawTexturedTriangle(
  ctx: CanvasRenderingContext2D,
  texture: CanvasImageSource,
  sx0: number,
  sy0: number,
  sx1: number,
  sy1: number,
  sx2: number,
  sy2: number,
  dx0: number,
  dy0: number,
  dx1: number,
  dy1: number,
  dx2: number,
  dy2: number,
): void {
  const denom = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1)
  if (Math.abs(denom) < 1e-6) {
    return
  }

  const a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denom
  const b = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denom
  const c = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denom
  const d = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denom
  const e = (
    dx0 * (sx1 * sy2 - sx2 * sy1)
    + dx1 * (sx2 * sy0 - sx0 * sy2)
    + dx2 * (sx0 * sy1 - sx1 * sy0)
  ) / denom
  const f = (
    dy0 * (sx1 * sy2 - sx2 * sy1)
    + dy1 * (sx2 * sy0 - sx0 * sy2)
    + dy2 * (sx0 * sy1 - sx1 * sy0)
  ) / denom

  ctx.save()
  ctx.beginPath()
  ctx.moveTo(dx0, dy0)
  ctx.lineTo(dx1, dy1)
  ctx.lineTo(dx2, dy2)
  ctx.closePath()
  ctx.clip()
  ctx.transform(a, b, c, d, e, f)
  ctx.drawImage(texture, 0, 0)
  ctx.restore()
}

function expandTriangleDestination(
  dx0: number,
  dy0: number,
  dx1: number,
  dy1: number,
  dx2: number,
  dy2: number,
  overlap: number,
): [number, number, number, number, number, number] {
  if (overlap <= 0) {
    return [dx0, dy0, dx1, dy1, dx2, dy2]
  }

  const cx = (dx0 + dx1 + dx2) / 3
  const cy = (dy0 + dy1 + dy2) / 3

  const inflateVertex = (x: number, y: number): [number, number] => {
    const vx = x - cx
    const vy = y - cy
    const len = Math.hypot(vx, vy)
    if (len < 1e-6) {
      return [x, y]
    }
    return [x + (vx / len) * overlap, y + (vy / len) * overlap]
  }

  const [ex0, ey0] = inflateVertex(dx0, dy0)
  const [ex1, ey1] = inflateVertex(dx1, dy1)
  const [ex2, ey2] = inflateVertex(dx2, dy2)
  return [ex0, ey0, ex1, ey1, ex2, ey2]
}

function bilerpCoord(a00: number, a10: number, a01: number, a11: number, u: number, v: number): number {
  const top = a00 + (a10 - a00) * u
  const bottom = a01 + (a11 - a01) * u
  return top + (bottom - top) * v
}

function sampleMotionVelocity(prev: MotionSample | null, x: number, y: number, nowMs: number): Landmark2D | null {
  if (!prev) {
    return null
  }

  const dtMs = nowMs - prev.tMs
  if (dtMs < MOTION_SAMPLE_MIN_DT_MS) {
    return null
  }

  return {
    x: ((x - prev.x) / dtMs) * RELEASE_IMPULSE_FRAME_MS,
    y: ((y - prev.y) / dtMs) * RELEASE_IMPULSE_FRAME_MS,
  }
}

function drawDistortedCamera(cloth: ClothState | null): void {
  if (!distortion) {
    return
  }

  const ctx = distortion.getContext('2d')
  if (!ctx) {
    return
  }

  const width = distortion.width
  const height = distortion.height
  if (width <= 0 || height <= 0) {
    return
  }

  const hasFrame = drawVideoFrameToBuffer(width, height)
  ctx.clearRect(0, 0, width, height)
  if (!hasFrame || !cloth || frameBuffer.width !== width || frameBuffer.height !== height) {
    return
  }

  ctx.imageSmoothingEnabled = true

  const { rows, cols, basePos, pos } = cloth
  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const i00 = (r * cols + c) * 2
      const i10 = (r * cols + (c + 1)) * 2
      const i01 = ((r + 1) * cols + c) * 2
      const i11 = ((r + 1) * cols + (c + 1)) * 2

      const bx00 = basePos[i00]
      const by00 = basePos[i00 + 1]
      const bx10 = basePos[i10]
      const by10 = basePos[i10 + 1]
      const bx01 = basePos[i01]
      const by01 = basePos[i01 + 1]
      const bx11 = basePos[i11]
      const by11 = basePos[i11 + 1]

      const px00 = pos[i00]
      const py00 = pos[i00 + 1]
      const px10 = pos[i10]
      const py10 = pos[i10 + 1]
      const px01 = pos[i01]
      const py01 = pos[i01 + 1]
      const px11 = pos[i11]
      const py11 = pos[i11 + 1]

      for (let sv = 0; sv < WARP_SUBDIVISIONS; sv += 1) {
        const v0 = sv / WARP_SUBDIVISIONS
        const v1 = (sv + 1) / WARP_SUBDIVISIONS

        for (let su = 0; su < WARP_SUBDIVISIONS; su += 1) {
          const u0 = su / WARP_SUBDIVISIONS
          const u1 = (su + 1) / WARP_SUBDIVISIONS

          const sx00 = bilerpCoord(bx00, bx10, bx01, bx11, u0, v0)
          const sy00 = bilerpCoord(by00, by10, by01, by11, u0, v0)
          const sx10 = bilerpCoord(bx00, bx10, bx01, bx11, u1, v0)
          const sy10 = bilerpCoord(by00, by10, by01, by11, u1, v0)
          const sx01 = bilerpCoord(bx00, bx10, bx01, bx11, u0, v1)
          const sy01 = bilerpCoord(by00, by10, by01, by11, u0, v1)
          const sx11 = bilerpCoord(bx00, bx10, bx01, bx11, u1, v1)
          const sy11 = bilerpCoord(by00, by10, by01, by11, u1, v1)

          const dx00 = bilerpCoord(px00, px10, px01, px11, u0, v0)
          const dy00 = bilerpCoord(py00, py10, py01, py11, u0, v0)
          const dx10 = bilerpCoord(px00, px10, px01, px11, u1, v0)
          const dy10 = bilerpCoord(py00, py10, py01, py11, u1, v0)
          const dx01 = bilerpCoord(px00, px10, px01, px11, u0, v1)
          const dy01 = bilerpCoord(py00, py10, py01, py11, u0, v1)
          const dx11 = bilerpCoord(px00, px10, px01, px11, u1, v1)
          const dy11 = bilerpCoord(py00, py10, py01, py11, u1, v1)

          const [t1x0, t1y0, t1x1, t1y1, t1x2, t1y2] = expandTriangleDestination(
            dx00,
            dy00,
            dx10,
            dy10,
            dx01,
            dy01,
            TRIANGLE_SEAM_OVERLAP_PX,
          )

          drawTexturedTriangle(
            ctx,
            frameBuffer,
            sx00,
            sy00,
            sx10,
            sy10,
            sx01,
            sy01,
            t1x0,
            t1y0,
            t1x1,
            t1y1,
            t1x2,
            t1y2,
          )

          const [t2x0, t2y0, t2x1, t2y1, t2x2, t2y2] = expandTriangleDestination(
            dx10,
            dy10,
            dx11,
            dy11,
            dx01,
            dy01,
            TRIANGLE_SEAM_OVERLAP_PX,
          )

          drawTexturedTriangle(
            ctx,
            frameBuffer,
            sx10,
            sy10,
            sx11,
            sy11,
            sx01,
            sy01,
            t2x0,
            t2y0,
            t2x1,
            t2y1,
            t2x2,
            t2y2,
          )
        }
      }
    }
  }
}

function clothPositionsAreFinite(state: ClothState): boolean {
  for (let i = 0; i < state.pos.length; i += 1) {
    if (!Number.isFinite(state.pos[i])) {
      return false
    }
  }
  return true
}

function isVideoReadyForDetection(videoEl: HTMLVideoElement | null): videoEl is HTMLVideoElement {
  if (!videoEl) {
    return false
  }

  if (videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return false
  }

  if (videoEl.videoWidth < 2 || videoEl.videoHeight < 2) {
    return false
  }

  if (videoEl.paused || videoEl.ended) {
    return false
  }

  if (!Number.isFinite(videoEl.currentTime) || videoEl.currentTime < 0) {
    return false
  }

  return true
}

function logOverlayDebug(reason: string): void {
  if (!OVERLAY_DEBUG) {
    return
  }

  const nowMs = performance.now()
  if (nowMs - lastOverlayDebugMs < OVERLAY_DEBUG_INTERVAL_MS) {
    return
  }
  lastOverlayDebugMs = nowMs

  const snapshot = {
    reason,
    shellCss: readElementSize(cameraShell),
    overlayCss: readElementSize(overlay),
    overlayBacking: { width: overlay?.width ?? 0, height: overlay?.height ?? 0 },
    resolved: resolveOverlaySize(),
    cloth: clothState
      ? {
          rows: clothState.rows,
          cols: clothState.cols,
          finite: clothPositionsAreFinite(clothState),
        }
      : null,
    landmarks: latestLandmarks?.length ?? 0,
    video: {
      readyState: video?.readyState ?? -1,
      width: video?.videoWidth ?? 0,
      height: video?.videoHeight ?? 0,
      time: video?.currentTime ?? 0,
    },
  }

  ;(window as Window & { __uroOverlayDebug?: unknown }).__uroOverlayDebug = snapshot
  console.log('[overlay-debug]', snapshot)
}

function waitForFrame(): Promise<void> {
  return new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
}

let startupWidenPulseDone = false
let showInteractionOverlay = true
let handGrabbing = false
let mouseGrabbing = false
let lastMouseSample: MotionSample | null = null
let latestMouseVelocity: Landmark2D | null = null
let lastHandSample: MotionSample | null = null
let latestHandVelocity: Landmark2D | null = null
let uploadedBackgroundUrl: string | null = null

function applyUploadedBackground(file: File): void {
  const nextUrl = URL.createObjectURL(file)
  if (uploadedBackgroundUrl) {
    URL.revokeObjectURL(uploadedBackgroundUrl)
  }
  uploadedBackgroundUrl = nextUrl

  const bgValue = `url("${nextUrl}")`
  if (cameraShell) {
    cameraShell.style.backgroundImage = bgValue
    cameraShell.style.backgroundRepeat = 'no-repeat'
    cameraShell.style.backgroundPosition = 'center'
    cameraShell.style.backgroundSize = 'cover'
  }
}

function installBackgroundUpload(): void {
  if (!uploadBgButton || !bgFileInput) {
    return
  }

  uploadBgButton.addEventListener('click', () => {
    bgFileInput.click()
  })

  bgFileInput.addEventListener('change', () => {
    const file = bgFileInput.files?.[0]
    if (!file || !file.type.startsWith('image/')) {
      return
    }

    applyUploadedBackground(file)
    bgFileInput.value = ''
  })

  window.addEventListener('beforeunload', () => {
    if (uploadedBackgroundUrl) {
      URL.revokeObjectURL(uploadedBackgroundUrl)
      uploadedBackgroundUrl = null
    }
  })
}

async function runStartupWidenPulse(): Promise<void> {
  if (!cameraShell || startupWidenPulseDone) {
    return
  }

  startupWidenPulseDone = true

  cameraShell.classList.add('wide-mode')
  await waitForFrame()
  await waitForFrame()
  await refreshOverlayCanvas('startup-pulse:wide')

  cameraShell.classList.remove('wide-mode')
  await waitForFrame()
  await waitForFrame()
  await refreshOverlayCanvas('startup-pulse:normal')
}

async function refreshOverlayCanvas(reason = 'manual'): Promise<void> {
  logOverlayDebug(`refresh:start:${reason}`)

  resizeOverlayToVideo()
  await waitForFrame()
  resizeOverlayToVideo()
  await waitForFrame()
  resizeOverlayToVideo()

  const { width, height } = resolveOverlaySize()
  if (width > 10 && height > 10) {
    if (!clothState || !clothPositionsAreFinite(clothState)) {
      clothState = createClothState(MESH_ROWS, MESH_COLS, width, height)
    }
    ensureMeshInitialized(clothState, width, height)
  } else {
    console.warn('[overlay-debug] refresh resolved invalid canvas size', { width, height, reason })
  }

  window.dispatchEvent(new Event('resize'))
  try {
    window.visualViewport?.dispatchEvent(new Event('resize'))
  } catch {
    // Some browser engines may not allow synthetic visualViewport resize events.
  }

  resizeOverlayToVideo()
  logOverlayDebug(`refresh:end:${reason}`)
}

function drawHandOverlay(
  cloth: ClothState | null,
  landmarks: Landmark2D[] | undefined,
  handClosed: boolean,
  cursor: Landmark2D | null,
): void {
  if (!overlay) {
    return
  }

  const ctx = overlay.getContext('2d')
  if (!ctx) {
    return
  }

  const width = overlay.width
  const height = overlay.height
  if (!width || !height) {
    return
  }

  const sourceWidth = video?.videoWidth || width
  const sourceHeight = video?.videoHeight || height

  const mapping = getCoverMapping(sourceWidth, sourceHeight, width, height)

  ctx.clearRect(0, 0, width, height)

  if (!showInteractionOverlay) {
    return
  }

  if (cloth) {
    const { rows, cols, pos } = cloth
    ctx.lineWidth = 1.5
    ctx.strokeStyle = CLOTH_LINE_COLOR

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols - 1; c += 1) {
        const i1 = (r * cols + c) * 2
        const i2 = (r * cols + (c + 1)) * 2
        ctx.beginPath()
        ctx.moveTo(pos[i1], pos[i1 + 1])
        ctx.lineTo(pos[i2], pos[i2 + 1])
        ctx.stroke()
      }
    }

    for (let r = 0; r < rows - 1; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const i1 = (r * cols + c) * 2
        const i2 = ((r + 1) * cols + c) * 2
        ctx.beginPath()
        ctx.moveTo(pos[i1], pos[i1 + 1])
        ctx.lineTo(pos[i2], pos[i2 + 1])
        ctx.stroke()
      }
    }

    ctx.fillStyle = CLOTH_POINT_COLOR
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const i = (r * cols + c) * 2
        ctx.beginPath()
        ctx.arc(pos[i], pos[i + 1], 2.8, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }

  if (!landmarks || landmarks.length === 0) {
    return
  }

  ctx.lineWidth = 2
  ctx.strokeStyle = 'rgba(56, 255, 173, 0.95)'
  for (const [start, end] of HAND_CONNECTIONS) {
    const p1 = landmarks[start]
    const p2 = landmarks[end]
    if (!p1 || !p2) {
      continue
    }

    const p1c = mapNormalizedToCover(p1.x, p1.y, mapping)
    const p2c = mapNormalizedToCover(p2.x, p2.y, mapping)

    ctx.beginPath()
    ctx.moveTo(p1c.x, p1c.y)
    ctx.lineTo(p2c.x, p2c.y)
    ctx.stroke()
  }

  ctx.fillStyle = 'rgba(255, 152, 66, 0.98)'
  for (const point of landmarks) {
    const mapped = mapNormalizedToCover(point.x, point.y, mapping)
    ctx.beginPath()
    ctx.arc(mapped.x, mapped.y, 4.5, 0, Math.PI * 2)
    ctx.fill()
  }

  if (cursor) {
    const cursorMapped = mapNormalizedToCover(cursor.x, cursor.y, mapping)
    ctx.strokeStyle = handClosed ? 'rgba(255, 80, 0, 0.95)' : 'rgba(0, 220, 120, 0.95)'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.arc(cursorMapped.x, cursorMapped.y, 9, 0, Math.PI * 2)
    ctx.stroke()
  }
}
 
function showError(message: string): void {
  if (!errorEl) return
  errorEl.hidden = false
  errorEl.textContent = message
}

async function startCamera(): Promise<void> {
  if (!video) {
    throw new Error('Camera element not found')
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
    },
  })

  video.srcObject = stream

  await new Promise<void>((resolve) => {
    const onLoaded = () => {
      video.removeEventListener('loadedmetadata', onLoaded)
      resolve()
    }
    if (video.readyState >= 1) {
      resolve()
      return
    }
    video.addEventListener('loadedmetadata', onLoaded)
  })

  try {
    await video.play()
  } catch {
    // Some browsers may delay autoplay; rendering loop still runs.
  }

  await refreshOverlayCanvas('camera-ready')
  await runStartupWidenPulse()
  void startHandTracker()
}

function handleHandGrab(cloth: ClothState | null): void {
  if (!cloth || mouseGrabbing) {
    lastHandSample = null
    latestHandVelocity = null
    return
  }

  if (!latestCursor || !latestHandClosed) {
    if (handGrabbing) {
      if (latestHandVelocity) {
        applyReleaseImpulse(
          cloth,
          latestHandVelocity.x * HAND_RELEASE_IMPULSE_SCALE,
          latestHandVelocity.y * HAND_RELEASE_IMPULSE_SCALE,
          RELEASE_IMPULSE_MAX,
        )
      }
      handGrabbing = false
      releaseDrag(cloth)
    }
    lastHandSample = null
    latestHandVelocity = null
    return
  }

  const sourceWidth = video?.videoWidth || cloth.width
  const sourceHeight = video?.videoHeight || cloth.height
  const mapping = getCoverMapping(sourceWidth, sourceHeight, cloth.width, cloth.height)
  const cursorMapped = mapNormalizedToCover(latestCursor.x, latestCursor.y, mapping)

  if (!handGrabbing) {
    handGrabbing = tryStartDrag(cloth, cursorMapped.x, cursorMapped.y, POINT_GRAB_RADIUS * HAND_GRAB_RADIUS_SCALE)
    if (handGrabbing) {
      const nowMs = performance.now()
      lastHandSample = { x: cursorMapped.x, y: cursorMapped.y, tMs: nowMs }
      latestHandVelocity = null
    }
  }

  if (handGrabbing) {
    const nowMs = performance.now()
    const velocity = sampleMotionVelocity(lastHandSample, cursorMapped.x, cursorMapped.y, nowMs)
    if (velocity) {
      latestHandVelocity = velocity
    }
    lastHandSample = { x: cursorMapped.x, y: cursorMapped.y, tMs: nowMs }
    setDragTarget(cloth, cursorMapped.x, cursorMapped.y)
  }
}

function installPointerGrabbing(): void {
  if (!overlay) {
    return
  }

  overlay.addEventListener('pointerdown', (event) => {
    if (!clothState) {
      return
    }

    const point = mapClientToCanvas(event.clientX, event.clientY)
    if (!point) {
      return
    }

    const grabbed = tryStartDrag(clothState, point.x, point.y, POINT_GRAB_RADIUS)
    if (!grabbed) {
      return
    }

    mouseGrabbing = true
    handGrabbing = false
    lastHandSample = null
    latestHandVelocity = null
    const nowMs = performance.now()
    lastMouseSample = { x: point.x, y: point.y, tMs: nowMs }
    latestMouseVelocity = null
    overlay.setPointerCapture(event.pointerId)
    event.preventDefault()
  })

  overlay.addEventListener('pointermove', (event) => {
    if (!mouseGrabbing || !clothState) {
      return
    }

    const point = mapClientToCanvas(event.clientX, event.clientY)
    if (!point) {
      return
    }

    const nowMs = performance.now()
    const velocity = sampleMotionVelocity(lastMouseSample, point.x, point.y, nowMs)
    if (velocity) {
      latestMouseVelocity = velocity
    }
    lastMouseSample = { x: point.x, y: point.y, tMs: nowMs }

    setDragTarget(clothState, point.x, point.y)
    event.preventDefault()
  })

  const stopMouseGrab = (event: PointerEvent) => {
    if (!mouseGrabbing || !clothState) {
      return
    }

    mouseGrabbing = false
    if (latestMouseVelocity) {
      applyReleaseImpulse(
        clothState,
        latestMouseVelocity.x * MOUSE_RELEASE_IMPULSE_SCALE,
        latestMouseVelocity.y * MOUSE_RELEASE_IMPULSE_SCALE,
        RELEASE_IMPULSE_MAX,
      )
    }
    releaseDrag(clothState)
    lastMouseSample = null
    latestMouseVelocity = null
    if (overlay.hasPointerCapture(event.pointerId)) {
      overlay.releasePointerCapture(event.pointerId)
    }
  }

  overlay.addEventListener('pointerup', stopMouseGrab)
  overlay.addEventListener('pointercancel', stopMouseGrab)
}

function installKeybinds(): void {
  window.addEventListener('keydown', (event) => {
    const key = event.key.toLowerCase()
    if (key === 't') {
      showInteractionOverlay = !showInteractionOverlay
      console.log(`Overlay: ${showInteractionOverlay ? 'on' : 'off'}`)
      event.preventDefault()
      return
    }

    if (key === 's') {
      clothConfig.pinEdges = !clothConfig.pinEdges
      console.log(`Pinned edges: ${clothConfig.pinEdges ? 'on' : 'off'}`)
      event.preventDefault()
      return
    }

    if (key !== 'r') {
      return
    }

    if (!clothState) {
      return
    }

    resetDisplacement(clothState)
    mouseGrabbing = false
    handGrabbing = false
    lastMouseSample = null
    latestMouseVelocity = null
    lastHandSample = null
    latestHandVelocity = null
    releaseDrag(clothState)
    event.preventDefault()
  })
}

let tracker: Awaited<ReturnType<typeof createHandTracker>> | null = null
let lastVideoTime = -1
let lastDetectionMs = -Infinity
let clothState: ClothState | null = null
let latestLandmarks: Landmark2D[] | undefined
let latestHandClosed = false
let latestCursor: Landmark2D | null = null
let overlayLoopStarted = false
let lastOverlayDebugMs = -Infinity
let handTrackerInitStarted = false
let handDetectPausedUntilMs = -Infinity
let lastHandDetectErrorLogMs = -Infinity

async function startHandTracker(): Promise<void> {
  if (handTrackerInitStarted) {
    return
  }
  handTrackerInitStarted = true

  try {
    tracker = await createHandTracker()
  } catch (err) {
    console.warn('Hand tracker failed to initialize. Cloth overlay will continue without landmarks.', err)
  }
}

function startOverlayLoop(): void {
  if (overlayLoopStarted) {
    return
  }
  overlayLoopStarted = true

  if (cameraShell && typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(() => {
      resizeOverlayToVideo()
    })
    observer.observe(cameraShell)
  }

  const tick = () => {
    if (!overlay) {
      requestAnimationFrame(tick)
      return
    }

    resizeOverlayToVideo()

    const nowMs = performance.now()

    const { width: overlayWidth, height: overlayHeight } = resolveOverlaySize()
    if (overlayWidth > 10 && overlayHeight > 10) {
      if (!clothState || !clothPositionsAreFinite(clothState)) {
        clothState = createClothState(MESH_ROWS, MESH_COLS, overlayWidth, overlayHeight)
        logOverlayDebug('tick:cloth-created-or-reset')
      }
      handleHandGrab(clothState)
      ensureMeshInitialized(clothState, overlayWidth, overlayHeight)
      stepClothSimulation(clothState, clothConfig)
      if (!clothPositionsAreFinite(clothState)) {
        console.warn('[overlay-debug] cloth became non-finite after step; rebuilding state')
        clothState = createClothState(MESH_ROWS, MESH_COLS, overlayWidth, overlayHeight)
      }
    } else {
      logOverlayDebug('tick:invalid-overlay-size')
    }

    // update hand detection only when new frame and at a capped cadence
    const videoReadyForDetection = isVideoReadyForDetection(video)
    if (
      videoReadyForDetection
      && tracker
      && nowMs >= handDetectPausedUntilMs
      && video.currentTime !== lastVideoTime
      && nowMs - lastDetectionMs >= DETECTION_INTERVAL_MS
    ) {
      lastVideoTime = video.currentTime
      lastDetectionMs = nowMs

      try {
        const result = tracker.detectForVideo(video, nowMs)

        const first = result.landmarks[0] as Landmark2D[] | undefined
        latestLandmarks = first?.map((lm) => ({ x: 1 - lm.x, y: lm.y }))
        latestHandClosed = false
        latestCursor = null

        if (latestLandmarks && latestLandmarks.length > 0) {
          latestHandClosed = isHandClosed(latestLandmarks)
          latestCursor = handCursorPoint(latestLandmarks)
        }
      } catch (err) {
        handDetectPausedUntilMs = nowMs + HAND_DETECTION_ERROR_COOLDOWN_MS

        latestLandmarks = undefined
        latestHandClosed = false
        latestCursor = null
        if (handGrabbing && clothState && !mouseGrabbing) {
          handGrabbing = false
          releaseDrag(clothState)
        }
        lastHandSample = null
        latestHandVelocity = null

        if (nowMs - lastHandDetectErrorLogMs >= HAND_DETECTION_ERROR_LOG_INTERVAL_MS) {
          lastHandDetectErrorLogMs = nowMs
          console.warn('Hand detection frame skipped after error. Retrying shortly.', err)
        }
      }
    } else if (!videoReadyForDetection && handGrabbing && clothState && !mouseGrabbing) {
      handGrabbing = false
      releaseDrag(clothState)
      lastHandSample = null
      latestHandVelocity = null
    }

    drawDistortedCamera(clothState)

    drawHandOverlay(clothState, latestLandmarks, latestHandClosed, latestCursor)

    requestAnimationFrame(tick)
  }
  tick()
}

async function init(): Promise<void> {
  window.addEventListener('resize', resizeOverlayToVideo)
  window.visualViewport?.addEventListener('resize', resizeOverlayToVideo)
  installKeybinds()
  installPointerGrabbing()
  installBackgroundUpload()

  startOverlayLoop()

  void refreshOverlayCanvas('init')

  void startCamera()
    .catch((err) => {
      console.error('Camera startup failed:', err)
      showError('Unable to access camera. Please allow camera access and refresh the page.')
    })
}

init().catch((err) => {
  console.error('Initialization failed:', err)
  showError('Initialization failed. Please refresh the page.')
})
