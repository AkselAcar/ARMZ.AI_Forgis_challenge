import type { Flow, FlowNode, FlowEdge } from "@/types";

// ── Layout constants ─────────────────────────────────────────

const W = 220;       // node width
const HB = 180;      // base node height (header + padding)
const HS = 80;       // additional height per step row
const GAP_X = 40;    // horizontal gap between parallel columns
const GAP_Y = 50;    // vertical gap between consecutive nodes
const START_H = 60;  // start/end node height

// ── Public API ───────────────────────────────────────────────

/**
 * Computes positions and sizes for every node in the flow.
 * Returns a new Flow with updated `position` and `style` fields.
 * Works with any flow shape: linear, branching, looping.
 */
export function layoutFlow(flow: Flow): Flow {
  const { nodes, edges } = flow;

  // Build lookup maps
  const nodeMap = new Map<string, FlowNode>(nodes.map((n) => [n.id, n]));

  // Non-loop edges are the only ones relevant for layout
  const layoutEdges = edges.filter((e) => e.data?.isLoop !== true);

  // Position map: nodeId → { x, y }
  const positions = new Map<string, { x: number; y: number }>();
  // Size map: nodeId → { width, height }
  const sizes = new Map<string, { width: number; height: number }>();

  // Pre-compute sizes for all nodes
  for (const node of nodes) {
    sizes.set(node.id, computeNodeSize(node));
  }

  // Find the start node
  const startNode = nodes.find((n) => n.type === "start");
  if (!startNode) return flow;

  // Layout from start node using recursive column placement
  layoutColumn(startNode.id, 0, 0, nodeMap, layoutEdges, positions, sizes);

  // Center the graph horizontally so the widest point starts at x=0
  centerHorizontally(positions);

  // Build the new nodes with computed positions and sizes
  const layoutNodes = nodes.map((n) => ({
    ...n,
    position: positions.get(n.id) ?? n.position,
    style: sizes.get(n.id),
  }));

  return { ...flow, nodes: layoutNodes };
}

// ── Core layout ──────────────────────────────────────────────

/**
 * Recursively lays out a column of nodes starting from `nodeId`.
 * Returns the total height consumed by this column.
 */
function layoutColumn(
  nodeId: string,
  x: number,
  y: number,
  nodeMap: Map<string, FlowNode>,
  edges: FlowEdge[],
  positions: Map<string, { x: number; y: number }>,
  sizes: Map<string, { width: number; height: number }>,
): number {
  // Skip if already positioned (handles merge points)
  if (positions.has(nodeId)) return 0;

  const node = nodeMap.get(nodeId);
  if (!node) return 0;

  const size = sizes.get(nodeId)!;

  // Center this node horizontally in its column
  positions.set(nodeId, { x: x + (W - size.width) / 2, y });

  let currentY = y + size.height + GAP_Y;

  // Find non-loop outgoing edges
  const outgoing = edges.filter((e) => e.source === nodeId);

  if (outgoing.length === 0) {
    // Terminal node — no children
    return currentY - y;
  }

  if (outgoing.length === 1) {
    // Linear chain — continue downward
    const childHeight = layoutColumn(
      outgoing[0].target, x, currentY, nodeMap, edges, positions, sizes,
    );
    return (currentY - y) + childHeight;
  }

  // Multiple outgoing edges → branching (conditional)
  // Find the merge point where branches converge
  const branchTargets = outgoing.map((e) => e.target);
  const mergeId = findCommonMerge(edges, branchTargets);

  // Count downstream weight for each branch (for proportional column widths)
  const branchWeights = branchTargets.map((target) =>
    countBranchNodes(target, mergeId, nodeMap, edges),
  );
  const totalWeight = branchWeights.reduce((a, b) => a + b, 0) || branchTargets.length;

  // Available width for all branches
  const totalBranchWidth = totalWeight * (W + GAP_X) - GAP_X;

  // Start branches from the left
  let branchX = x + (W - totalBranchWidth) / 2;
  let maxBranchHeight = 0;

  for (let i = 0; i < branchTargets.length; i++) {
    const weight = branchWeights[i] || 1;
    const columnWidth = weight * (W + GAP_X) - GAP_X;
    const columnCenter = branchX + (columnWidth - W) / 2;

    const branchHeight = layoutColumn(
      branchTargets[i], columnCenter, currentY, nodeMap, edges, positions, sizes,
    );

    maxBranchHeight = Math.max(maxBranchHeight, branchHeight);
    branchX += columnWidth + GAP_X;
  }

  currentY += maxBranchHeight;

  // Continue layout from merge point (if found and not already positioned)
  if (mergeId && !positions.has(mergeId)) {
    const mergeHeight = layoutColumn(
      mergeId, x, currentY, nodeMap, edges, positions, sizes,
    );
    return (currentY - y) + mergeHeight;
  }

  return currentY - y;
}

// ── Helper functions ─────────────────────────────────────────

/** Computes the pixel size of a node based on its type and step count. */
function computeNodeSize(node: FlowNode): { width: number; height: number } {
  if (node.type === "start" || node.type === "end") {
    return { width: START_H, height: START_H };
  }
  const stepCount = node.steps?.length ?? 0;
  return { width: W, height: HB + stepCount * HS };
}

/**
 * Counts nodes reachable from `start` up to (but not including) `stopBefore`.
 * Used to weight branch column widths proportionally.
 */
function countBranchNodes(
  start: string,
  stopBefore: string | null,
  nodeMap: Map<string, FlowNode>,
  edges: FlowEdge[],
): number {
  const visited = new Set<string>();
  const queue = [start];

  while (queue.length > 0) {
    const current = queue.shift()!;
    if (visited.has(current)) continue;
    if (current === stopBefore) continue;
    if (!nodeMap.has(current)) continue;
    visited.add(current);

    for (const e of edges) {
      if (e.source === current && !visited.has(e.target)) {
        queue.push(e.target);
      }
    }
  }
  return visited.size;
}

/**
 * Given multiple branch start nodes, finds the first node reachable from
 * ALL of them — the merge point where branches converge.
 * Returns null if no common merge exists (branches never rejoin).
 */
function findCommonMerge(
  edges: FlowEdge[],
  branchTargets: string[],
): string | null {
  if (branchTargets.length < 2) return null;

  // Collect all reachable nodes for each branch
  const reachableSets = branchTargets.map((target) => {
    const reachable = new Set<string>();
    const queue = [target];
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (reachable.has(current)) continue;
      reachable.add(current);
      for (const e of edges) {
        if (e.source === current && !reachable.has(e.target)) {
          queue.push(e.target);
        }
      }
    }
    return reachable;
  });

  // Find nodes reachable from ALL branches
  const common = [...reachableSets[0]].filter((id) =>
    reachableSets.every((set) => set.has(id)),
  );

  if (common.length === 0) return null;

  // Return the closest common node — the one with shortest max distance from any branch
  // BFS distance from each branch target
  const distances = new Map<string, number[]>();
  for (const id of common) {
    distances.set(id, []);
  }

  for (let i = 0; i < branchTargets.length; i++) {
    const dist = new Map<string, number>();
    const queue: [string, number][] = [[branchTargets[i], 0]];
    while (queue.length > 0) {
      const [current, d] = queue.shift()!;
      if (dist.has(current)) continue;
      dist.set(current, d);
      for (const e of edges) {
        if (e.source === current && !dist.has(e.target)) {
          queue.push([e.target, d + 1]);
        }
      }
    }
    for (const id of common) {
      distances.get(id)!.push(dist.get(id) ?? Infinity);
    }
  }

  // Pick the node with the smallest maximum distance (closest merge point)
  let bestId: string | null = null;
  let bestMaxDist = Infinity;
  for (const [id, dists] of distances) {
    const maxDist = Math.max(...dists);
    if (maxDist < bestMaxDist) {
      bestMaxDist = maxDist;
      bestId = id;
    }
  }

  return bestId;
}

/**
 * Centers the entire graph horizontally so that the minimum x position is 0.
 */
function centerHorizontally(positions: Map<string, { x: number; y: number }>) {
  let minX = Infinity;
  for (const pos of positions.values()) {
    if (pos.x < minX) minX = pos.x;
  }
  if (minX !== 0 && minX !== Infinity) {
    for (const pos of positions.values()) {
      pos.x -= minX;
    }
  }
}
