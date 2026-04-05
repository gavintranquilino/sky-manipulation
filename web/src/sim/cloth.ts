export type Vec2 = { x: number; y: number }

export type ClothConfig = {
  physicsSubsteps: number
  physicsTimestep: number
  velocityDamping: number
  springStiffnessStruct: number
  springStiffnessDiag: number
  anchorStiffness: number
  dragStiffness: number
  restLengthAdaptation: number
  dragFollow: number
  minCompressionRatio: number
  maxNodeDisplacement: number
  pinEdges: boolean
}

export const defaultClothConfig: ClothConfig = {
  physicsSubsteps: 3,
  physicsTimestep: 1.0,
  velocityDamping: 0.985,
  springStiffnessStruct: 0.14,
  springStiffnessDiag: 0.11,
  anchorStiffness: 0.0,
  dragStiffness: 0.52,
  restLengthAdaptation: 0.08,
  dragFollow: 0.74,
  minCompressionRatio: 0.72,
  maxNodeDisplacement: 2200,
  pinEdges: false,
}

export type ClothState = {
  width: number
  height: number
  rows: number
  cols: number
  basePos: Float32Array
  pos: Float32Array
  vel: Float32Array
  restH: Float32Array
  restV: Float32Array
  restD1: Float32Array
  restD2: Float32Array
  dragIndex: number
  dragTarget: Vec2 | null
}

function nodeIndex(row: number, col: number, cols: number): number {
  return (row * cols + col) * 2
}

function edgeHIndex(row: number, col: number, cols: number): number {
  return row * (cols - 1) + col
}

function edgeVIndex(row: number, col: number, cols: number): number {
  return row * cols + col
}

function edgeDiagIndex(row: number, col: number, cols: number): number {
  return row * (cols - 1) + col
}

function pointDistance(x1: number, y1: number, x2: number, y2: number): number {
  return Math.hypot(x2 - x1, y2 - y1)
}

function buildBaseGrid(width: number, height: number, rows: number, cols: number): Float32Array {
  const data = new Float32Array(rows * cols * 2)
  const xDenom = Math.max(cols - 1, 1)
  const yDenom = Math.max(rows - 1, 1)

  for (let r = 0; r < rows; r += 1) {
    const y = yDenom === 0 ? 0 : (r / yDenom) * (height - 1)
    for (let c = 0; c < cols; c += 1) {
      const x = xDenom === 0 ? 0 : (c / xDenom) * (width - 1)
      const i = nodeIndex(r, c, cols)
      data[i] = x
      data[i + 1] = y
    }
  }

  return data
}

function computeRestLengths(state: ClothState): void {
  const { rows, cols, basePos, restH, restV, restD1, restD2 } = state

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const i1 = nodeIndex(r, c, cols)
      const i2 = nodeIndex(r, c + 1, cols)
      restH[edgeHIndex(r, c, cols)] = pointDistance(
        basePos[i1],
        basePos[i1 + 1],
        basePos[i2],
        basePos[i2 + 1],
      )
    }
  }

  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const i1 = nodeIndex(r, c, cols)
      const i2 = nodeIndex(r + 1, c, cols)
      restV[edgeVIndex(r, c, cols)] = pointDistance(
        basePos[i1],
        basePos[i1 + 1],
        basePos[i2],
        basePos[i2 + 1],
      )
    }
  }

  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const iTL = nodeIndex(r, c, cols)
      const iBR = nodeIndex(r + 1, c + 1, cols)
      restD1[edgeDiagIndex(r, c, cols)] = pointDistance(
        basePos[iTL],
        basePos[iTL + 1],
        basePos[iBR],
        basePos[iBR + 1],
      )

      const iTR = nodeIndex(r, c + 1, cols)
      const iBL = nodeIndex(r + 1, c, cols)
      restD2[edgeDiagIndex(r, c, cols)] = pointDistance(
        basePos[iTR],
        basePos[iTR + 1],
        basePos[iBL],
        basePos[iBL + 1],
      )
    }
  }
}

export function createClothState(rows: number, cols: number, width: number, height: number): ClothState {
  const nodeCount = rows * cols
  const state: ClothState = {
    width,
    height,
    rows,
    cols,
    basePos: new Float32Array(nodeCount * 2),
    pos: new Float32Array(nodeCount * 2),
    vel: new Float32Array(nodeCount * 2),
    restH: new Float32Array(rows * Math.max(cols - 1, 0)),
    restV: new Float32Array(Math.max(rows - 1, 0) * cols),
    restD1: new Float32Array(Math.max(rows - 1, 0) * Math.max(cols - 1, 0)),
    restD2: new Float32Array(Math.max(rows - 1, 0) * Math.max(cols - 1, 0)),
    dragIndex: -1,
    dragTarget: null,
  }

  ensureMeshInitialized(state, width, height)
  return state
}

export function ensureMeshInitialized(state: ClothState, width: number, height: number): void {
  if (state.width === width && state.height === height && state.basePos.length > 0) {
    return
  }

  state.width = width
  state.height = height
  state.basePos = buildBaseGrid(width, height, state.rows, state.cols)
  state.pos = state.basePos.slice()
  state.vel = new Float32Array(state.basePos.length)
  state.dragIndex = -1
  state.dragTarget = null
  computeRestLengths(state)
}

export function resetDisplacement(state: ClothState): void {
  state.pos.set(state.basePos)
  state.vel.fill(0)
  state.dragIndex = -1
  state.dragTarget = null
  computeRestLengths(state)
}

function clampPoint(x: number, y: number, width: number, height: number): Vec2 {
  return {
    x: Math.max(0, Math.min(width - 1, x)),
    y: Math.max(0, Math.min(height - 1, y)),
  }
}

export function setDragTarget(state: ClothState, x: number, y: number): void {
  state.dragTarget = clampPoint(x, y, state.width, state.height)
}

export function releaseDrag(state: ClothState): void {
  state.dragIndex = -1
  state.dragTarget = null
}

export function applyReleaseImpulse(state: ClothState, vx: number, vy: number, maxImpulse = 120): void {
  if (state.dragIndex < 0 || !Number.isFinite(vx) || !Number.isFinite(vy)) {
    return
  }

  let impulseX = vx
  let impulseY = vy
  const mag = Math.hypot(impulseX, impulseY)
  if (mag > maxImpulse && mag > 1e-6) {
    const scale = maxImpulse / mag
    impulseX *= scale
    impulseY *= scale
  }

  const row = Math.floor(state.dragIndex / state.cols)
  const col = state.dragIndex % state.cols
  const i = nodeIndex(row, col, state.cols)
  state.vel[i] += impulseX
  state.vel[i + 1] += impulseY
}

export function tryStartDrag(state: ClothState, x: number, y: number, radius: number): boolean {
  const { rows, cols, pos } = state
  let best = -1
  let bestDist = Number.POSITIVE_INFINITY

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const i = nodeIndex(r, c, cols)
      const d = pointDistance(x, y, pos[i], pos[i + 1])
      if (d < bestDist) {
        bestDist = d
        best = r * cols + c
      }
    }
  }

  if (best < 0 || bestDist > radius) {
    return false
  }

  state.dragIndex = best
  setDragTarget(state, x, y)
  return true
}

function enforcePinnedEdges(state: ClothState): void {
  const { rows, cols, pos, basePos, vel } = state

  for (let c = 0; c < cols; c += 1) {
    const top = nodeIndex(0, c, cols)
    const bottom = nodeIndex(rows - 1, c, cols)

    pos[top] = basePos[top]
    pos[top + 1] = basePos[top + 1]
    vel[top] = 0
    vel[top + 1] = 0

    pos[bottom] = basePos[bottom]
    pos[bottom + 1] = basePos[bottom + 1]
    vel[bottom] = 0
    vel[bottom + 1] = 0
  }

  for (let r = 0; r < rows; r += 1) {
    const left = nodeIndex(r, 0, cols)
    const right = nodeIndex(r, cols - 1, cols)

    pos[left] = basePos[left]
    pos[left + 1] = basePos[left + 1]
    vel[left] = 0
    vel[left + 1] = 0

    pos[right] = basePos[right]
    pos[right + 1] = basePos[right + 1]
    vel[right] = 0
    vel[right + 1] = 0
  }
}

function clampDisplacementsFromBase(state: ClothState, maxNodeDisplacement: number): void {
  if (maxNodeDisplacement <= 0) {
    return
  }

  const { pos, basePos } = state
  for (let i = 0; i < pos.length; i += 2) {
    const dx = pos[i] - basePos[i]
    const dy = pos[i + 1] - basePos[i + 1]
    const mag = Math.hypot(dx, dy)
    if (mag > maxNodeDisplacement && mag > 1e-6) {
      const scale = maxNodeDisplacement / mag
      pos[i] = basePos[i] + dx * scale
      pos[i + 1] = basePos[i + 1] + dy * scale
    }
  }
}

function enforceMinNeighborDistance(state: ClothState, minCompressionRatio: number): void {
  if (minCompressionRatio <= 0) {
    return
  }

  const { rows, cols, pos, restH, restV, restD1, restD2 } = state

  const solvePair = (i1: number, i2: number, minLen: number): void => {
    const dx = pos[i2] - pos[i1]
    const dy = pos[i2 + 1] - pos[i1 + 1]
    const dist = Math.hypot(dx, dy)
    if (dist >= minLen) {
      return
    }

    const safeDist = Math.max(dist, 1e-6)
    const correctionMag = 0.5 * (minLen - dist)
    const nx = dx / safeDist
    const ny = dy / safeDist

    pos[i1] -= correctionMag * nx
    pos[i1 + 1] -= correctionMag * ny
    pos[i2] += correctionMag * nx
    pos[i2 + 1] += correctionMag * ny
  }

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const i1 = nodeIndex(r, c, cols)
      const i2 = nodeIndex(r, c + 1, cols)
      solvePair(i1, i2, restH[edgeHIndex(r, c, cols)] * minCompressionRatio)
    }
  }

  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const i1 = nodeIndex(r, c, cols)
      const i2 = nodeIndex(r + 1, c, cols)
      solvePair(i1, i2, restV[edgeVIndex(r, c, cols)] * minCompressionRatio)
    }
  }

  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const iTL = nodeIndex(r, c, cols)
      const iBR = nodeIndex(r + 1, c + 1, cols)
      solvePair(iTL, iBR, restD1[edgeDiagIndex(r, c, cols)] * minCompressionRatio)

      const iTR = nodeIndex(r, c + 1, cols)
      const iBL = nodeIndex(r + 1, c, cols)
      solvePair(iTR, iBL, restD2[edgeDiagIndex(r, c, cols)] * minCompressionRatio)
    }
  }
}

function springForce(
  forces: Float32Array,
  pos: Float32Array,
  i1: number,
  i2: number,
  restLength: number,
  stiffness: number,
): void {
  const dx = pos[i2] - pos[i1]
  const dy = pos[i2 + 1] - pos[i1 + 1]
  const dist = Math.hypot(dx, dy)
  const safeDist = Math.max(dist, 1e-6)
  const nx = dx / safeDist
  const ny = dy / safeDist
  const ext = dist - restLength
  const fx = stiffness * ext * nx
  const fy = stiffness * ext * ny

  forces[i1] += fx
  forces[i1 + 1] += fy
  forces[i2] -= fx
  forces[i2 + 1] -= fy
}

function adaptRestLengths(state: ClothState, alpha: number): void {
  const { rows, cols, pos, restH, restV, restD1, restD2 } = state

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const i1 = nodeIndex(r, c, cols)
      const i2 = nodeIndex(r, c + 1, cols)
      const current = pointDistance(pos[i1], pos[i1 + 1], pos[i2], pos[i2 + 1])
      const i = edgeHIndex(r, c, cols)
      restH[i] = (1 - alpha) * restH[i] + alpha * current
    }
  }

  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const i1 = nodeIndex(r, c, cols)
      const i2 = nodeIndex(r + 1, c, cols)
      const current = pointDistance(pos[i1], pos[i1 + 1], pos[i2], pos[i2 + 1])
      const i = edgeVIndex(r, c, cols)
      restV[i] = (1 - alpha) * restV[i] + alpha * current
    }
  }

  for (let r = 0; r < rows - 1; r += 1) {
    for (let c = 0; c < cols - 1; c += 1) {
      const iTL = nodeIndex(r, c, cols)
      const iBR = nodeIndex(r + 1, c + 1, cols)
      const d1Current = pointDistance(pos[iTL], pos[iTL + 1], pos[iBR], pos[iBR + 1])
      const dIndex = edgeDiagIndex(r, c, cols)
      restD1[dIndex] = (1 - alpha) * restD1[dIndex] + alpha * d1Current

      const iTR = nodeIndex(r, c + 1, cols)
      const iBL = nodeIndex(r + 1, c, cols)
      const d2Current = pointDistance(pos[iTR], pos[iTR + 1], pos[iBL], pos[iBL + 1])
      restD2[dIndex] = (1 - alpha) * restD2[dIndex] + alpha * d2Current
    }
  }
}

export function stepClothSimulation(state: ClothState, config: ClothConfig = defaultClothConfig): void {
  const {
    physicsSubsteps,
    physicsTimestep,
    velocityDamping,
    springStiffnessStruct,
    springStiffnessDiag,
    anchorStiffness,
    dragStiffness,
    restLengthAdaptation,
    dragFollow,
    minCompressionRatio,
    maxNodeDisplacement,
    pinEdges,
  } = config

  const dt = physicsTimestep / Math.max(physicsSubsteps, 1)
  const forces = new Float32Array(state.pos.length)

  for (let sub = 0; sub < physicsSubsteps; sub += 1) {
    forces.fill(0)

    const { rows, cols, pos, basePos, vel, restH, restV, restD1, restD2 } = state

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols - 1; c += 1) {
        const i1 = nodeIndex(r, c, cols)
        const i2 = nodeIndex(r, c + 1, cols)
        springForce(forces, pos, i1, i2, restH[edgeHIndex(r, c, cols)], springStiffnessStruct)
      }
    }

    for (let r = 0; r < rows - 1; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const i1 = nodeIndex(r, c, cols)
        const i2 = nodeIndex(r + 1, c, cols)
        springForce(forces, pos, i1, i2, restV[edgeVIndex(r, c, cols)], springStiffnessStruct)
      }
    }

    for (let r = 0; r < rows - 1; r += 1) {
      for (let c = 0; c < cols - 1; c += 1) {
        const iTL = nodeIndex(r, c, cols)
        const iBR = nodeIndex(r + 1, c + 1, cols)
        springForce(forces, pos, iTL, iBR, restD1[edgeDiagIndex(r, c, cols)], springStiffnessDiag)

        const iTR = nodeIndex(r, c + 1, cols)
        const iBL = nodeIndex(r + 1, c, cols)
        springForce(forces, pos, iTR, iBL, restD2[edgeDiagIndex(r, c, cols)], springStiffnessDiag)
      }
    }

    if (anchorStiffness !== 0) {
      for (let i = 0; i < pos.length; i += 2) {
        forces[i] += anchorStiffness * (basePos[i] - pos[i])
        forces[i + 1] += anchorStiffness * (basePos[i + 1] - pos[i + 1])
      }
    }

    if (state.dragIndex !== -1 && state.dragTarget) {
      const row = Math.floor(state.dragIndex / state.cols)
      const col = state.dragIndex % state.cols
      const i = nodeIndex(row, col, state.cols)
      forces[i] += dragStiffness * (state.dragTarget.x - pos[i])
      forces[i + 1] += dragStiffness * (state.dragTarget.y - pos[i + 1])
    }

    for (let i = 0; i < pos.length; i += 2) {
      vel[i] += forces[i] * dt
      vel[i + 1] += forces[i + 1] * dt
      vel[i] *= velocityDamping
      vel[i + 1] *= velocityDamping
      pos[i] += vel[i] * dt
      pos[i + 1] += vel[i + 1] * dt
    }

    if (state.dragIndex !== -1 && state.dragTarget) {
      const row = Math.floor(state.dragIndex / state.cols)
      const col = state.dragIndex % state.cols
      const i = nodeIndex(row, col, state.cols)
      pos[i] = (1 - dragFollow) * pos[i] + dragFollow * state.dragTarget.x
      pos[i + 1] = (1 - dragFollow) * pos[i + 1] + dragFollow * state.dragTarget.y
      vel[i] *= 0.35
      vel[i + 1] *= 0.35
    }

    if (restLengthAdaptation > 0) {
      const alpha = Math.min(Math.max(restLengthAdaptation * dt, 0), 1)
      adaptRestLengths(state, alpha)
    }

    enforceMinNeighborDistance(state, minCompressionRatio)
    clampDisplacementsFromBase(state, maxNodeDisplacement)

    if (pinEdges) {
      enforcePinnedEdges(state)
    }
  }
}

export function getNodePosition(state: ClothState, row: number, col: number): Vec2 {
  const i = nodeIndex(row, col, state.cols)
  return { x: state.pos[i], y: state.pos[i + 1] }
}

export function forEachNode(state: ClothState, callback: (row: number, col: number, x: number, y: number) => void): void {
  for (let r = 0; r < state.rows; r += 1) {
    for (let c = 0; c < state.cols; c += 1) {
      const i = nodeIndex(r, c, state.cols)
      callback(r, c, state.pos[i], state.pos[i + 1])
    }
  }
}
