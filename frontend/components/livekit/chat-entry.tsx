import * as React from 'react';
import { cn } from '@/lib/utils';

export interface ChatEntryProps extends React.HTMLAttributes<HTMLLIElement> {
  /** The locale to use for the timestamp. */
  locale: string;
  /** The timestamp of the message. */
  timestamp: number;
  /** The message to display. */
  message: string;
  /** The origin of the message. */
  messageOrigin: 'local' | 'remote';
  /** The sender's name. */
  name?: string;
  /** Whether the message has been edited. */
  hasBeenEdited?: boolean;
}

export const ChatEntry = ({
  name,
  locale,
  timestamp,
  message,
  messageOrigin,
  hasBeenEdited = false,
  className,
  ...props
}: ChatEntryProps) => {
  const time = new Date(timestamp);
  const title = time.toLocaleTimeString(locale, { timeStyle: 'full' });

  return (
    <li
      title={title}
      data-lk-message-origin={messageOrigin}
      className={cn('group flex w-full flex-col gap-2', className)}
      {...props}
    >
      <header
        className={cn(
          'text-muted-foreground flex items-center gap-2 text-base font-semibold',
          messageOrigin === 'local' ? 'flex-row-reverse' : 'text-left'
        )}
      >
        {name && <strong className="text-lg">{name}</strong>}
        <span className="font-mono text-sm opacity-60 transition-opacity ease-linear group-hover:opacity-100">
          {hasBeenEdited && '*'}
          {time.toLocaleTimeString(locale, { timeStyle: 'short' })}
        </span>
      </header>
      <div
        className={cn(
          'text-[18px] leading-relaxed px-5 py-3.5 max-w-[85%] shadow-sm w-fit',
          messageOrigin === 'local' 
            ? 'ml-auto bg-primary text-on-primary rounded-3xl rounded-tr-md' 
            : 'mr-auto bg-surface-variant text-on-surface-variant border border-outline-variant/30 rounded-3xl rounded-tl-md'
        )}
      >
        {message}
      </div>
    </li>
  );
};
