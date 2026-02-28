import { Box, Tag, ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";

type DetectionType = "bbox" | "label" | "qc";

export interface LatestDetection {
  type: DetectionType;
  label: string;
  confidence?: number;
  shot: number;
}

interface DetectionStatusProps {
  latestDetection: LatestDetection | null;
}

export function DetectionStatus({ latestDetection }: DetectionStatusProps) {
  return (
    <div className="flex items-center justify-between mt-2 mb-3">
      {latestDetection ? (
        <>
          <div className="flex items-center gap-1.5">
            {latestDetection.type === "bbox" ? (
              <Box size={11} className="text-[var(--status-healthy)]" />
            ) : latestDetection.type === "qc" ? (
              <ShieldCheck size={12} className={latestDetection.label === "Readable" ? "text-[var(--status-healthy)]" : "text-[var(--status-critical)]"} />
            ) : (
              <Tag size={13} className="text-[var(--gunmetal-50)]" />
            )}
            <span className={cn(
              "forgis-text-caption leading-none font-forgis-body truncate",
              latestDetection.type === "bbox" ? "text-[var(--status-healthy)]" : latestDetection.type === "qc" ? (latestDetection.label === "Readable" ? "text-[var(--status-healthy)]" : "text-[var(--status-critical)]") : "text-[var(--gunmetal-50)]"
            )}>
              {latestDetection.label}
              {latestDetection.confidence !== undefined && ` (${(latestDetection.confidence * 100).toFixed(0)}%)`}
            </span>
          </div>
          <span className="forgis-text-micro text-[var(--gunmetal-50)] font-forgis-digit whitespace-nowrap">
            Shot {latestDetection.shot}
          </span>
        </>
      ) : (
        <span className="forgis-text-caption text-[var(--gunmetal-50)] font-forgis-body">
          No detections yet
        </span>
      )}
    </div>
  );
}
