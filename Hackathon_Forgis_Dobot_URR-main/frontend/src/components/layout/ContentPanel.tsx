import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import type { ContentPanelProps } from '@/types/index';

/**
 * Generic content panel wrapper with consistent header styling.
 * Matches Figma design: white bg, subtle #f1f5f7 border, 10px rounded.
 */
export function ContentPanel({
  title,
  subtitle,
  actions,
  centerActions,
  children,
  className,
  contentClassName,
  scrollable = true,
}: ContentPanelProps) {
  return (
    <Card
      className={cn(
        'flex flex-col overflow-hidden py-0 gap-0',
        'bg-card border border-border rounded-[10px]',
        className
      )}
    >
      <CardHeader className="relative flex-shrink-0 flex flex-row items-center justify-between gap-4 pt-4 pb-4 px-6">
        <div className="flex flex-col gap-1 shrink-0">
          <CardTitle className="forgis-text-title font-normal uppercase text-[var(--gunmetal-50)] leading-none font-forgis-digit">
            {title}
          </CardTitle>
          {subtitle && (
            <span className="forgis-text-detail text-[var(--gunmetal-50)] font-forgis-body leading-tight">
              {subtitle}
            </span>
          )}
        </div>
        {centerActions && (
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center gap-2">
            {centerActions}
          </div>
        )}
        {actions ? <div className="flex items-center gap-2 shrink-0">{actions}</div> : <div />}
      </CardHeader>
      <CardContent className={cn(
        'flex-1 px-6 pb-4 pt-0',
        scrollable ? 'overflow-auto' : 'overflow-hidden',
        contentClassName
      )}>
        {children}
      </CardContent>
    </Card>
  );
}
