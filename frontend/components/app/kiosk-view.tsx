'use client';

import { useSessionContext, useSessionMessages, useTranscriptions } from '@livekit/components-react';
import React, { useState, useEffect, useRef } from 'react';
import { ChatTranscript } from '@/components/app/chat-transcript';
import { ScrollArea } from '@/components/livekit/scroll-area/scroll-area';

export function KioskView() {
  const session = useSessionContext();
  const { isConnected, start, end } = session;
  const { messages } = useSessionMessages(session);
  const transcriptions = useTranscriptions();
  
  const [time, setTime] = useState('');
  const [dateStr, setDateStr] = useState('');
  
  // Facebook Posts State
  const [fbPosts, setFbPosts] = useState<any[]>([]);
  const [currentSlide, setCurrentSlide] = useState(0);

  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const lastMessage = messages.at(-1);
    const lastMessageIsLocal = lastMessage?.from?.isLocal === true;

    if (scrollAreaRef.current && lastMessageIsLocal) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
      setDateStr(now.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' }));
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch Facebook Posts
  useEffect(() => {
    const fetchPosts = async () => {
      try {
        const response = await fetch('/api/facebook');
        const data = await response.json();
        if (Array.isArray(data) && data.length > 0) {
          setFbPosts(data);
        }
      } catch (error) {
        console.error('Failed to fetch FB posts:', error);
      }
    };
    fetchPosts();
    // Refresh every 30 minutes
    const interval = setInterval(fetchPosts, 30 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Slideshow Logic
  useEffect(() => {
    if (fbPosts.length <= 1) return;
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % fbPosts.length);
    }, 8000); // 8 seconds per slide
    return () => clearInterval(interval);
  }, [fbPosts.length]);

  return (
    <div className="bg-background text-on-background w-full h-screen overflow-hidden flex flex-col select-none" style={{ fontFamily: 'Inter, sans-serif' }}>
      {/* Top App Bar */}
      <header className="bg-surface flex-shrink-0 w-full flex justify-between items-center px-8 h-[56px]">
        <div className="text-3xl font-bold text-primary">NEma</div>
        <div className="flex items-center gap-4">
          {isConnected && (
            <span className="text-green-500 font-semibold animate-pulse">Connected</span>
          )}
        </div>
      </header>
      
      {/* Main Content Area - Bento Grid */}
      <main className="flex-1 px-8 py-6 overflow-hidden min-h-0 flex flex-col">
        <div className="grid grid-cols-12 gap-6 flex-1 min-h-0 pb-4">
          {/* Left Column: Clock & Navigation */}
          <div className="col-span-4 flex flex-col gap-6 h-full min-h-0">
            {/* Clock Card */}
            <div className="bg-primary-container text-on-primary-container rounded-3xl p-6 flex flex-col items-center justify-center shadow-sm relative overflow-hidden flex-shrink-0">
              <span className="material-symbols-outlined absolute top-4 right-4 text-4xl opacity-20 fill-current">light_mode</span>
              <div className="text-[64px] leading-[64px] tracking-[-0.04em] font-bold text-primary">{time || '10:42'}</div>
              <div className="text-[18px] leading-[24px] mt-1 font-bold">{dateStr || 'Thursday, June 4'}</div>
            </div>
            
            {/* Where to? Card */}
            <div className="bg-surface-container rounded-3xl p-6 shadow-sm flex-1 flex flex-col relative overflow-hidden min-h-0">
              <h2 className="text-[32px] leading-[40px] tracking-[-0.02em] text-primary mb-3 font-bold flex-shrink-0">Where to?</h2>
              <div className="flex-1 flex flex-col justify-end gap-4 mt-auto">
                <button className="bg-primary text-on-primary rounded-full flex-1 max-h-[64px] text-[20px] flex items-center justify-center gap-4 hover:bg-surface-tint transition-colors active:scale-95 shadow-md font-bold flex-shrink-0">
                  <span className="material-symbols-outlined text-3xl">school</span>
                  Dean's Office
                </button>
                <button className="bg-surface-variant text-on-surface-variant rounded-full flex-1 max-h-[64px] text-[20px] flex items-center justify-center gap-4 hover:bg-surface-container-highest transition-colors active:scale-95 shadow-sm border border-outline-variant font-bold flex-shrink-0">
                  <span className="material-symbols-outlined text-3xl">computer</span>
                  Computer Lab 03
                </button>
                <button className="bg-surface-variant text-on-surface-variant rounded-full flex-1 max-h-[64px] text-[20px] flex items-center justify-center gap-4 hover:bg-surface-container-highest transition-colors active:scale-95 shadow-sm border border-outline-variant font-bold flex-shrink-0">
                  <span className="material-symbols-outlined text-3xl">apartment</span>
                  Lecture Hall
                </button>
              </div>
            </div>
          </div>
          
          {/* Middle Column: Events Carousel & Microphone */}
          <div className="col-span-4 h-full min-h-0 flex flex-col gap-6">
            <div className="bg-secondary-container rounded-3xl shadow-sm flex-1 overflow-hidden relative flex flex-col min-h-0">
              {isConnected ? (
                <div className="flex-1 flex flex-col relative h-full bg-surface-container">
                  <div className="px-6 py-4 bg-surface-container-high border-b border-outline-variant/30 shrink-0">
                    <h3 className="text-xl font-bold text-primary">Conversation</h3>
                  </div>
                  <ScrollArea ref={scrollAreaRef} className="flex-1 p-4">
                    <ChatTranscript messages={messages} className="space-y-4 pb-4" />
                  </ScrollArea>
                </div>
              ) : (
                <>
                  <div className="absolute inset-0 z-0 bg-secondary-container">
                {fbPosts.length > 0 ? (
                  <img alt="Facebook Post" className="w-full h-full object-cover transition-opacity duration-1000" src={fbPosts[currentSlide].full_picture} key={fbPosts[currentSlide].id} />
                ) : (
                  <img alt="Placeholder" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/aida-public/AB6AXuASe7OPmposO-19UAIeU4spfafXd_IIkyengbRnIoJXP5vzcgsqBX4KhpYGHDv1RVod-dKhSD4LadBgQAlGEoyLGT5i8i3olLcgb8xypR5mcuEL1Q78xoqtkxWnKF9jhItfILnYltqiwrrLAeE3ZFxZ7nCEHNlwi6t2MOxghHruNkBxUQQYFFp_Rkb-PqnZNEPZKbK-jp7fxgCeZsKJJkieYur0T9mHyCpYbIlQ5BJ_1U1E1ZsWoHM1etOrM2fPLnCL8NLiGnhxxs4" />
                )}
              </div>
              <div className="relative z-10 p-6 flex flex-col h-full bg-gradient-to-t from-black/80 via-black/30 to-transparent text-white">
                <div className="mt-auto">

                  {fbPosts.length > 0 ? (
                    <>
                      <h3 className="text-[20px] font-bold leading-tight mb-2 line-clamp-3">{fbPosts[currentSlide].message}</h3>
                      <p className="text-[14px] opacity-90">{new Date(fbPosts[currentSlide].created_time).toLocaleDateString()}</p>
                    </>
                  ) : (
                    <>
                      <h3 className="text-[24px] font-bold leading-tight mb-2">Connecting to Facebook...</h3>
                      <p className="text-[16px] opacity-90">Fetching latest posts.</p>
                    </>
                  )}
                </div>
              </div>
              {/* Carousel Indicators */}
              {fbPosts.length > 1 && (
                <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-2 z-20">
                  {fbPosts.map((_, idx) => (
                    <div key={idx} className={`w-2 h-2 rounded-full transition-colors ${idx === currentSlide ? 'bg-on-secondary' : 'bg-on-secondary/50'}`}></div>
                  ))}
                </div>
              )}
              {/* Facebook Logo Watermark */}
              <div className="absolute bottom-4 right-4 z-20 text-[#1877F2] bg-white rounded-full p-[2px] shadow-lg flex items-center justify-center">
                <svg viewBox="0 0 24 24" fill="currentColor" className="w-7 h-7"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.469h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.469h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
              </div>
              </>
              )}
            </div>

            {/* Microphone Action Area */}
            <div className="flex-shrink-0 h-[140px] flex flex-col items-center justify-center bg-surface-container-low rounded-3xl shadow-sm relative px-4">
              <div className="relative w-full h-12 mb-2 flex justify-center items-end text-center text-[24px] font-bold text-primary">
                {!isConnected ? (
                  <div className="relative w-full h-full overflow-hidden">
                    <div className="greeting-text greeting-1 leading-normal">How can I help you?</div>
                    <div className="greeting-text greeting-2 leading-normal">Tap the mic to ask a question!</div>
                  </div>
                ) : (
                  <div className="w-full flex justify-center break-words pb-1">
                    {transcriptions.slice(-1)[0]?.text || 'Listening...'}
                  </div>
                )}
              </div>
              <div className="flex justify-center w-full">
                <button 
                  onClick={() => isConnected ? end() : start()}
                  className={`w-[64px] h-[64px] text-on-primary rounded-full flex items-center justify-center shadow-xl hover:scale-105 transition-transform active:scale-95 border-none ${isConnected ? 'bg-error animate-pulse' : 'bg-primary animate-neon-pulse'}`}
                  style={{ backgroundColor: isConnected ? '#ba1a1a' : 'rgb(116, 86, 96)' }}
                >
                  <span className="material-symbols-outlined text-4xl fill-current">{isConnected ? 'mic_off' : 'mic'}</span>
                </button>
              </div>
            </div>
          </div>
          
          {/* Right Column: Faculty News */}
          <div className="col-span-4 h-full min-h-0">
            <div className="bg-primary text-on-primary rounded-3xl p-6 shadow-md h-full flex flex-col border-4 border-primary-container/30 min-h-0">
              <h2 className="text-[40px] font-bold text-on-primary mb-4 flex items-center gap-3">
                <span className="material-symbols-outlined text-5xl">campaign</span>
                Faculty News
              </h2>
              <div className="flex-1 flex flex-col justify-between overflow-hidden pr-2">
                {/* News Item 1 */}
                <div className="group cursor-pointer">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-primary-container text-on-primary-container px-3 py-1 rounded-full text-[14px] font-semibold">Jun 2</span>
                    <span className="text-[14px] font-bold uppercase tracking-widest text-on-primary">Competition</span>
                  </div>
                  <h4 className="text-[24px] font-bold text-on-primary group-hover:text-primary-container transition-colors leading-tight">Robotics Team Wins Nationals</h4>
                </div>
                {/* News Item 2 */}
                <div className="group cursor-pointer">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-primary-container text-on-primary-container px-3 py-1 rounded-full text-[14px] font-semibold">May 28</span>
                    <span className="text-[14px] font-bold uppercase tracking-widest text-on-primary">Announcement</span>
                  </div>
                  <h4 className="text-[24px] font-bold text-on-primary group-hover:text-primary-container transition-colors leading-tight">New Grant Awarded to CS Dept</h4>
                </div>
                {/* News Item 3 */}
                <div className="group cursor-pointer">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-primary-container text-on-primary-container px-3 py-1 rounded-full text-[14px] font-semibold">May 25</span>
                    <span className="text-[14px] font-bold uppercase tracking-widest text-on-primary">Seminar</span>
                  </div>
                  <h4 className="text-[24px] font-bold text-on-primary group-hover:text-primary-container transition-colors leading-tight">Guest Lecture: Ethics in ML</h4>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
