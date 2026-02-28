import type { BoundingBoxOverlay } from "@/types";

interface BboxOverlayProps {
  bboxStyle: { left: number; top: number; width: number; height: number };
  overlay: BoundingBoxOverlay;
}

export function BboxOverlay({ bboxStyle, overlay }: BboxOverlayProps) {
  return (
    <div
      className="absolute border-2 border-[var(--status-healthy)] pointer-events-none animate-pulse"
      style={{
        left: bboxStyle.left,
        top: bboxStyle.top,
        width: bboxStyle.width,
        height: bboxStyle.height,
      }}
    >
      <span className="absolute -top-5 left-0 forgis-text-label bg-[var(--status-healthy)] text-white px-1 rounded">
        {overlay.bbox.class_name} ({(overlay.bbox.confidence * 100).toFixed(0)}%)
      </span>
    </div>
  );
}
