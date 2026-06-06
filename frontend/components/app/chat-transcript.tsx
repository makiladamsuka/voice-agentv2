'use client';

import { AnimatePresence, type HTMLMotionProps, motion } from 'motion/react';
import { type ReceivedMessage } from '@livekit/components-react';
import { ChatEntry } from '@/components/livekit/chat-entry';

const MotionContainer = motion.create('div');
const MotionChatEntry = motion.create(ChatEntry);

const CONTAINER_MOTION_PROPS = {
  variants: {
    hidden: {
      opacity: 0,
      transition: {
        ease: 'easeOut' as const,
        duration: 0.3,
        staggerChildren: 0.1,
        staggerDirection: -1,
      },
    },
    visible: {
      opacity: 1,
      transition: {
        delay: 0.2,
        ease: 'easeOut' as const,
        duration: 0.3,
        delayChildren: 0.2,
        staggerChildren: 0.1,
        staggerDirection: 1,
      },
    },
  },
  initial: 'hidden' as const,
  animate: 'visible' as const,
  exit: 'hidden' as const,
} satisfies HTMLMotionProps<'div'>;

const MESSAGE_MOTION_PROPS = {
  variants: {
    hidden: {
      opacity: 0,
      translateY: 10,
    },
    visible: {
      opacity: 1,
      translateY: 0,
    },
  },
};

interface ChatTranscriptProps {
  hidden?: boolean;
  messages?: ReceivedMessage[];
  transcriptions?: any[];
}

export function ChatTranscript({
  hidden = false,
  messages = [],
  transcriptions = [],
  ...props
}: ChatTranscriptProps & Omit<HTMLMotionProps<'div'>, 'ref'>) {
  // Combine messages and transcriptions
  const combinedItems = [
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

  return (
    <AnimatePresence>
      {!hidden && (
        <MotionContainer {...CONTAINER_MOTION_PROPS} {...props} className="flex flex-col gap-4 pb-4">
          {combinedItems.map((item) => {
            if (!item.message) return null;

            const locale = navigator?.language ?? 'en-US';
            const messageOrigin = item.isLocal ? 'local' : 'remote';

            return (
              <MotionChatEntry
                key={item.id}
                locale={locale}
                timestamp={item.timestamp}
                message={item.message}
                messageOrigin={messageOrigin}
                hasBeenEdited={false}
                {...MESSAGE_MOTION_PROPS}
              />
            );
          })}
        </MotionContainer>
      )}
    </AnimatePresence>
  );
}
