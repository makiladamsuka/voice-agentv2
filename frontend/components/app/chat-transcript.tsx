"use client";

import { type ReceivedMessage } from "@livekit/components-react";
import { ChatEntry } from "@/components/livekit/chat-entry";

interface ChatTranscriptProps {
  hidden?: boolean;
  messages?: ReceivedMessage[];
  transcriptions?: any[];
  className?: string;
  stagingText?: string;
  isLoading?: boolean;
}

export function ChatTranscript({
  hidden = false,
  messages = [],
  transcriptions = [],
  className,
  stagingText = "",
  isLoading = false,
  ...props
}: ChatTranscriptProps & React.HTMLAttributes<HTMLDivElement>) {
  
  // Use ONLY LiveKit finalized messages to prevent ID conflicts and layout jumps
  const rawItems = messages
    .map((m: any) => {
      const text = m.message || m.text;
      const isLocal = m.from?.isLocal || false;
      return {
        id: m.id || String(m.timestamp),
        timestamp: m.timestamp || Date.now(),
        message: text,
        isLocal: isLocal,
        isFinal: true,
      };
    })
    .sort((a: any, b: any) => a.timestamp - b.timestamp);

  if (hidden) return null;

  return (
    <div className={`flex flex-col gap-4 pb-4 ${className || ""}`} {...props}>
      {rawItems.map((item) => {
        if (!item.message) return null;

        // Hide the item from chat if it's currently being spoken in the staging area!
        if (
          stagingText &&
          (item.message.includes(stagingText) ||
            stagingText.includes(item.message))
        ) {
          return null;
        }

        const locale = navigator?.language ?? "en-US";
        const messageOrigin = item.isLocal ? "local" : "remote";

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
