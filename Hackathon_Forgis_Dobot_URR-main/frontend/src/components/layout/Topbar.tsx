import { Link } from 'react-router-dom';
import { Bell, Settings, User } from 'lucide-react';
import { cn } from '@/lib/utils';

interface TopBarProps {
  className?: string;
}

/**
 * Global app header with Forgis branding and utility icons.
 * Matches Figma: 32px height, 12px icons, dark gunmetal background.
 * Logo click navigates back to Factory View (homepage).
 */
export function TopBar({ className }: TopBarProps) {
  return (
    <header
      className={cn(
        'flex items-center justify-between h-8 px-4',
        'bg-[var(--gunmetal)]',
        className
      )}
    >
      {/* Left side - Logo (icon + wordmark) - links to Factory View */}
      <Link to="/" className="flex items-center hover:opacity-80 transition-opacity">
        <img
          src="/logo-wordmark.png"
          alt="Forgis"
          className="h-[17px] w-auto"
        />
      </Link>

      {/* Right side - Utility icons (placeholders) */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          className="p-1 hover:opacity-80 transition-opacity"
          aria-label="Notifications"
        >
          <Bell size={12} className="text-white" />
        </button>
        <button
          type="button"
          className="p-1 hover:opacity-80 transition-opacity"
          aria-label="Settings"
        >
          <Settings size={12} className="text-white" />
        </button>
        <button
          type="button"
          className="p-1 hover:opacity-80 transition-opacity"
          aria-label="User profile"
        >
          <User size={12} className="text-white" />
        </button>
      </div>
    </header>
  );
}
