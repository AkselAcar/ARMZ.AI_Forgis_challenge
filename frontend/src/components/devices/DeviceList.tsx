import type { Device } from "@/types";
import { DEVICE_ICONS, STATUS_COLOR, STATUS_LABEL } from "@/constants/deviceConfig";
import { cn } from "@/lib/utils";

interface DeviceListProps {
  devices: Device[];
  compact?: boolean;
}

export function DeviceList({ devices, compact = false }: DeviceListProps) {
  if (devices.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-1.5 py-5 text-muted-foreground">
        <span className="forgis-text-label font-forgis-body">No devices connected.</span>
      </div>
    );
  }

  if (compact) {
    return (
      <div className="divide-y divide-border">
        {devices.map((device) => {
          const Icon = DEVICE_ICONS[device.type];
          const color = STATUS_COLOR[device.status];
          const label = STATUS_LABEL[device.status];
          return (
            <div key={device.id} className="flex items-center gap-2.5 px-3 py-2">
              <div className="shrink-0 w-6 h-6 rounded-md bg-muted/50 flex items-center justify-center">
                <Icon className="w-3.5 h-3.5 text-muted-foreground" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="forgis-text-detail font-normal text-foreground leading-tight truncate font-forgis-digit">
                  {device.name}
                </div>
                <div className="forgis-text-detail text-[var(--gunmetal-50)] truncate font-forgis-body">
                  {device.vendor} · {device.ip}
                </div>
              </div>
              <div className="flex items-center gap-1 shrink-0">
                <span
                  className={cn(
                    "w-1.5 h-1.5 rounded-full shrink-0",
                    device.status === "warning" && "animate-pulse",
                  )}
                  style={{ background: color }}
                />
                <span className="forgis-text-detail font-forgis-digit" style={{ color }}>{label}</span>
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="divide-y divide-border">
      {devices.map((device) => {
        const Icon = DEVICE_ICONS[device.type];
        const color = STATUS_COLOR[device.status];
        const label = STATUS_LABEL[device.status];
        return (
          <div key={device.id} className="flex items-center gap-3 px-6 py-2.5">
            <div className="shrink-0 w-7 h-7 rounded-md bg-muted/50 flex items-center justify-center">
              <Icon className="w-4 h-4 text-muted-foreground" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="forgis-text-label font-medium text-foreground leading-tight truncate">
                {device.name}
              </div>
              <div className="forgis-text-caption text-muted-foreground truncate">{device.vendor}</div>
            </div>
            <span className="forgis-text-caption text-muted-foreground font-mono shrink-0">
              {device.ip}
            </span>
            <div className="flex items-center gap-1.5 shrink-0 w-[76px] justify-end">
              <span
                className={cn(
                  "w-1.5 h-1.5 rounded-full shrink-0",
                  device.status === "warning" && "animate-pulse",
                )}
                style={{ background: color }}
              />
              <span className="forgis-text-caption" style={{ color }}>{label}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
