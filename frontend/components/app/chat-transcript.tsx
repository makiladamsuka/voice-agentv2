'use client';

import { type ReceivedMessage } from '@livekit/components-react';
import { ChatEntry } from '@/components/livekit/chat-entry';

interface ChatTranscriptProps {
  hidden?: boolean;
  messages?: ReceivedMessage[];
  transcriptions?: any[];
  className?: string;
}

export function ChatTranscript({
  hidden = false,
  messages = [],
  transcriptions = [],
  className,
  ...props
}: ChatTranscriptProps & React.HTMLAttributes<HTMLDivElement>) {
  // Combine messages and transcriptions
  const rawItems = [
    ...messages.map((m: any) => ({
      id: m.id || String(m.timestamp),
      timestamp: m.timestamp,
      message: m.message || m.text,
      isLocal: m.from?.isLocal || false,
      isFinal: true
    })),
    ...transcriptions.map((t: any) => ({
        id: t.id,
        timestamp: t.firstReceivedTime || Date.now(),
        message: t.text,
        isLocal: t.participant?.isLocal || false,
        isFinal: t.isFinal
    }))
  ].sort((a, b) => a.timestamp - b.timestamp);

  // Deduplicate progressive transcriptions and instant messages
  const combinedItems = rawItems.reduce((acc: any[], current: any) => {
    if (!current.message || current.message.trim() === '') return acc;

    const existingIndex = acc.findIndex(item => 
      item.isLocal === current.isLocal && 
      (item.message.includes(current.message) || current.message.includes(item.message)) &&
      Math.abs(item.timestamp - current.timestamp) < 10000 // within 10 seconds
    );

    if (existingIndex >= 0) {
      // Keep the longer (more complete) message
      if (current.message.length > acc[existingIndex].message.length) {
        acc[existingIndex] = { ...current, id: acc[existingIndex].id }; // preserve original ID to avoid React re-mounting
      }
    } else {
      acc.push(current);
    }
    return acc;
  }, []);

  if (hidden) return null;

  return (
    <div className={`flex flex-col gap-4 pb-4 ${className || ''}`} {...props}>
      {combinedItems.map((item) => {
        if (!item.message) return null;

        const locale = navigator?.language ?? 'en-US';
        const messageOrigin = item.isLocal ? 'local' : 'remote';

        return (
          <ChatEntry
            key={item.id}
            locale={locale}
            timestamp={item.timestamp}
            message={item.message}
            messageOrigin={messageOrigin}
            hasBeenEdited={false}
          />
        );
      })}
    </div>
  );
}
