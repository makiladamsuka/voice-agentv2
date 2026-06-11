'use client';

import { useSessionContext, useSessionMessages, useTranscriptions, useTracks, useTrackVolume, useVoiceAssistant, useRoomContext } from '@livekit/components-react';
import { Track } from 'livekit-client';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { ChatTranscript } from '@/components/app/chat-transcript';
import { ScrollArea } from '@/components/livekit/scroll-area/scroll-area';
import { ThemeToggle } from '@/components/app/theme-toggle';
import { QRCodeSVG } from 'qrcode.react';
import { UploadCloud, X } from 'lucide-react';

export function KioskView() {
  const session = useSessionContext();
  const { isConnected, start, end } = session;
  const { messages } = useSessionMessages(session);
  const room = useRoomContext();

  // Focused event state — set when a news card is tapped
  const [focusedEvent, setFocusedEvent] = useState<any | null>(null);
  const pendingEventRef = useRef<any | null>(null);
  const transcriptions = useTranscriptions();

  const { audioTrack: agentTrack, state: agentState } = useVoiceAssistant();
  const agentVolume = useTrackVolume(agentTrack);
  const micTracks = useTracks([Track.Source.Microphone]);
  const localMicTrack = micTracks.find(t => t.participant.isLocal);
  const micVolume = useTrackVolume(localMicTrack);
  const maxVolume = isConnected ? Math.max(agentVolume || 0, micVolume || 0) : 0;
  
  const isThinking = agentState === 'thinking';
  // Dramatically amplify the scaling and opacity for the visual pulse effect
  const pulseScale = isThinking ? 1.05 : (!isConnected ? undefined : 1 + (maxVolume * 2.0));
  const pulseOpacity = !isConnected ? undefined : (isThinking ? 0.5 : 0.2 + (maxVolume * 0.8));

  // Send event context to backend via LiveKit data channel
  const sendEventFocus = useCallback((event: any) => {
    if (!room) return;
    try {
      const payload = JSON.stringify({ type: 'event_focus', event });
      room.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true });
      console.log('📲 Sent event_focus to agent:', event.message);
    } catch (e) {
      console.error('Failed to publish event data:', e);
    }
  }, [room]);

  // When connection established AND there's a pending event, send it
  useEffect(() => {
    if (isConnected && pendingEventRef.current) {
      const ev = pendingEventRef.current;
      pendingEventRef.current = null;
      // Small delay so agent finishes its "I'm ready" greeting first
      setTimeout(() => sendEventFocus(ev), 2500);
    }
  }, [isConnected, sendEventFocus]);

  // Handle clicking a news card
  const handleNewsClick = useCallback(async (post: any) => {
    setFocusedEvent(post);
    if (!isConnected) {
      pendingEventRef.current = post;
      await start();
    } else {
      sendEventFocus(post);
    }
  }, [isConnected, start, sendEventFocus]);
  
  const latestTranscription = transcriptions[transcriptions.length - 1];
  const [stagingText, setStagingText] = useState('');

  useEffect(() => {
    if (latestTranscription && latestTranscription.text) {
      setStagingText(latestTranscription.text);
      const timer = setTimeout(() => {
        setStagingText('');
      }, 2000); // Clear after 2 seconds of silence
      return () => clearTimeout(timer);
    }
  }, [latestTranscription?.text]);

  // Keep other state variables below
  const [time, setTime] = useState('');
  const [dateStr, setDateStr] = useState('');

  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [qrUrl, setQrUrl] = useState('');

  useEffect(() => {
    async function fetchIp() {
      try {
        const res = await fetch('/api/network-ip');
        const data = await res.json();
        if (data.ip) {
          setQrUrl(`http://${data.ip}:3000/upload-portal`);
        } else {
          setQrUrl(`http://${window.location.hostname}:3000/upload-portal`);
        }
      } catch (err) {
        if (typeof window !== 'undefined') {
          setQrUrl(`http://${window.location.hostname}:3000/upload-portal`);
        }
      }
    }
    fetchIp();
  }, []);

  const lastKnownUploadRef = useRef(0);

  // Poll for successful uploads to auto-close the modal and show the new poster
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/upload-status');
        const data = await res.json();
        
        if (lastKnownUploadRef.current === 0) {
           // Initial load, just sync the current state
           lastKnownUploadRef.current = data.lastUpload;
        } else if (data.lastUpload > lastKnownUploadRef.current) {
           // A new upload happened globally!
           lastKnownUploadRef.current = data.lastUpload;
           
           // We can still trigger UI changes like resetting current slide or closing modal
           setIsUploadModalOpen(false);
           setCurrentSlide(0);
        }
        
        // Always sync the local files list dynamically to reflect additions and deletions
        if (data.allFiles) {
          const newLocalPosts = data.allFiles.map((file: any) => {
             const categoryMap: Record<string, string> = {
                events: "Featured Campus Event",
                competitions: "Upcoming Competition",
                posts: "Campus Announcement"
             };
             const defaultTitle = categoryMap[file.category] || "Campus Highlight";
             
             // Prioritize the AI-extracted title
             const title = file.extracted?.title || defaultTitle;
             
             return {
                 id: 'local_' + file.mtimeMs + '_' + file.name,
                 full_picture: file.url,
                 message: title,
                 description: file.extracted?.description || '',
                 extracted_date: file.extracted?.date || '',
                 extracted_time: file.extracted?.time || '',
                 extracted_location: file.extracted?.location || '',
                 created_time: new Date(file.mtimeMs).toISOString(),
                 isLocal: true,
                 category: file.category
             };
          });
          setLocalPosts(newLocalPosts);
        }
      } catch (e) {
        // Ignore errors
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);

  const isAgentInitializing = isConnected && 
    messages.filter(m => !m.from?.isLocal).length === 0 && 
    transcriptions.length === 0;
  
  // Merged Posts State
  const [facebookPosts, setFacebookPosts] = useState<any[]>([]);
  const [localPosts, setLocalPosts] = useState<any[]>([]);
  const fbPosts = [...localPosts, ...facebookPosts];
  const [currentSlide, setCurrentSlide] = useState(0);

  // Standby Rotating Prompts
  const STANDBY_PROMPTS = [
    "Welcome to the Faculty of IT!",
    "Ask for directions to Lab 03",
    "Find the Dean's Office here",
    "Ask about events & news",
    "Need help navigating campus?",
    "Tap the mic to start chatting!"
  ];
  const [currentPromptIndex, setCurrentPromptIndex] = useState(0);

  useEffect(() => {
    if (isConnected) return;
    const interval = setInterval(() => {
      setCurrentPromptIndex((prev) => (prev + 1) % STANDBY_PROMPTS.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [isConnected]);

  // Weather State
  const [weather, setWeather] = useState<{ temp: number; icon: string } | null>(null);

  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Scroll to bottom whenever messages or transcriptions change
    if (scrollAreaRef.current) {
      // Use requestAnimationFrame to let React paint the new bubbles first
      requestAnimationFrame(() => {
        if (scrollAreaRef.current) {
          scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
        }
      });
    }
  }, [messages, transcriptions, stagingText]);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true }));
      setDateStr(now.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' }));
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  // Track connection state transitions
  const wasConnectedRef = useRef(isConnected);

  // Auto-disconnect after 5 minutes of inactivity
  useEffect(() => {
    if (!isConnected) {
      if (wasConnectedRef.current) {
        // Transitioned from connected to disconnected
        setFocusedEvent(null);
        pendingEventRef.current = null;
      }
      wasConnectedRef.current = false;
      return;
    }
    wasConnectedRef.current = true;
    
    const timeoutId = setTimeout(() => {
      console.log('Disconnecting due to inactivity');
      end();
    }, 5 * 60 * 1000); // 5 minutes
    
    return () => clearTimeout(timeoutId);
  }, [isConnected, end, messages, transcriptions]);

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        const res = await fetch('https://api.open-meteo.com/v1/forecast?latitude=6.7951&longitude=79.9003&current_weather=true');
        const data = await res.json();
        const code = data.current_weather.weathercode;
        let icon = 'light_mode';
        if (code === 0) icon = 'light_mode';
        else if (code === 1 || code === 2) icon = 'partly_cloudy_day';
        else if (code === 3) icon = 'cloud';
        else if (code === 45 || code === 48) icon = 'foggy';
        else if (code >= 51 && code <= 65) icon = 'rainy';
        else if (code >= 71 && code <= 77) icon = 'weather_snow';
        else if (code >= 80 && code <= 82) icon = 'rainy';
        else if (code >= 85 && code <= 86) icon = 'weather_snow';
        else if (code >= 95) icon = 'thunderstorm';
        
        setWeather({
          temp: Math.round(data.current_weather.temperature),
          icon
        });
      } catch (err) {
        console.error('Failed to fetch weather', err);
      }
    };
    fetchWeather();
    const interval = setInterval(fetchWeather, 30 * 60 * 1000); // 30 mins
    return () => clearInterval(interval);
  }, []);

  // Fetch Facebook Posts
  useEffect(() => {
    const fetchPosts = async () => {
      try {
        const response = await fetch('/api/facebook');
        const data = await response.json();
        if (Array.isArray(data) && data.length > 0) {
          setFacebookPosts(data);
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

  // Slideshow Logic - only rotates when standby (disconnected)
  useEffect(() => {
    if (isConnected) return;
    if (fbPosts.length <= 1) return;
    const interval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % fbPosts.length);
    }, 8000); // 8 seconds per slide
    return () => clearInterval(interval);
  }, [fbPosts.length, isConnected]);

  // Swipe to change slides
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);

  const minSwipeDistance = 50;

  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
  };

  const onTouchMove = (e: React.TouchEvent) => setTouchEnd(e.targetTouches[0].clientX);

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;
    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    if (isLeftSwipe && fbPosts.length > 0) {
      setCurrentSlide((prev) => (prev + 1) % fbPosts.length);
    }
    if (isRightSwipe && fbPosts.length > 0) {
      setCurrentSlide((prev) => (prev - 1 + fbPosts.length) % fbPosts.length);
    }
  };

  return (
    <div className="relative text-on-background w-full h-screen overflow-hidden flex flex-col select-none bg-background/40" style={{ fontFamily: 'Inter, sans-serif' }}>
      {/* Dynamic Ambient Background Glow */}
      <div className="absolute inset-0 -z-20 overflow-hidden pointer-events-none bg-background transition-colors duration-1000">
        <div 
          className="absolute -top-[15%] -left-[15%] w-[50%] h-[50%] rounded-full blur-[140px] transition-all duration-[2000ms] ease-in-out"
          style={{
            background: fbPosts[currentSlide]?.category === 'competitions' ? 'radial-gradient(circle, rgba(249,115,22,0.18) 0%, rgba(244,63,94,0.06) 70%, rgba(255,255,255,0) 100%)' :
                        fbPosts[currentSlide]?.category === 'events' ? 'radial-gradient(circle, rgba(139,92,246,0.15) 0%, rgba(99,102,241,0.05) 70%, rgba(255,255,255,0) 100%)' :
                        fbPosts[currentSlide]?.category === 'posts' ? 'radial-gradient(circle, rgba(20,184,166,0.15) 0%, rgba(6,182,212,0.05) 70%, rgba(255,255,255,0) 100%)' :
                        'radial-gradient(circle, rgba(78,96,118,0.12) 0%, rgba(78,96,118,0.03) 70%, rgba(255,255,255,0) 100%)',
            animation: 'ambientBlob1 12s ease-in-out infinite'
          }}
        />
        <div 
          className="absolute -bottom-[15%] -right-[15%] w-[50%] h-[50%] rounded-full blur-[140px] transition-all duration-[2000ms] ease-in-out"
          style={{
            background: fbPosts[currentSlide]?.category === 'competitions' ? 'radial-gradient(circle, rgba(244,63,94,0.15) 0%, rgba(249,115,22,0.05) 70%, rgba(255,255,255,0) 100%)' :
                        fbPosts[currentSlide]?.category === 'events' ? 'radial-gradient(circle, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.05) 70%, rgba(255,255,255,0) 100%)' :
                        fbPosts[currentSlide]?.category === 'posts' ? 'radial-gradient(circle, rgba(6,182,212,0.15) 0%, rgba(20,184,166,0.05) 70%, rgba(255,255,255,0) 100%)' :
                        'radial-gradient(circle, rgba(78,96,118,0.12) 0%, rgba(78,96,118,0.03) 70%, rgba(255,255,255,0) 100%)',
            animation: 'ambientBlob2 12s ease-in-out infinite'
          }}
        />
      </div>

      {/* Main Content Wrapper (must be above background) */}
      <div className="relative z-10 w-full h-full flex flex-col">

      <style>{`
        @keyframes ambientBlob1 {
          0%, 100% { transform: translate(0px, 0px) scale(1); }
          50% { transform: translate(30px, 15px) scale(1.1); }
        }
        @keyframes ambientBlob2 {
          0%, 100% { transform: translate(0px, 0px) scale(1); }
          50% { transform: translate(-30px, -15px) scale(1.1); }
        }
      `}</style>

      {/* Top App Bar */}
      <header 
        className="bg-surface dark:bg-[#141316] flex-shrink-0 w-full flex justify-between items-center px-8 h-[54px] pb-1 border-b border-outline-variant/30 dark:border-white/10 shadow-md z-20"
        style={{ clipPath: 'ellipse(80% 100% at 50% 0%)' }}
      >
        <div className="text-2xl font-bold text-primary tracking-tight">NEma</div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsUploadModalOpen(true)}
            className="bg-primary/10 hover:bg-primary/20 text-primary px-4 py-1.5 rounded-full text-sm font-bold flex items-center gap-2 transition-colors"
          >
            <UploadCloud className="w-4 h-4" />
            Upload Poster
          </button>
          {isConnected && (
            <div className="bg-green-500/10 border border-green-500/30 text-green-600 dark:text-green-400 px-3 py-1.5 rounded-full text-xs font-bold flex items-center gap-2 shadow-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              Connected
            </div>
          )}
          <ThemeToggle />
        </div>
      </header>
      
      {/* Main Content Area - Bento Grid */}
      <main className="flex-1 pl-8 pr-4 pt-3 pb-4 overflow-hidden min-h-0 flex flex-col">
        <div className="flex gap-6 flex-1 min-h-0 pb-2">
          {/* Left Column: Clock & Navigation — collapses when poster is focused */}
          <div
            className="flex flex-col gap-6 h-full min-h-0 flex-shrink-0 overflow-hidden transition-all duration-500 ease-in-out"
            style={{ width: focusedEvent ? '0px' : '25%', opacity: focusedEvent ? 0 : 1 }}
          >
            {/* Clock & Weather Card */}
            <div className="bg-primary-container text-on-primary-container rounded-3xl p-5 pt-8 flex flex-col items-center justify-center shadow-sm relative overflow-hidden flex-shrink-0">
              {weather ? (
                <div className="absolute top-3 right-4 flex items-center opacity-80 text-primary">
                  <span className="material-symbols-outlined text-[24px] fill-current">{weather.icon}</span>
                </div>
              ) : (
                <span className="material-symbols-outlined absolute top-3 right-4 text-[24px] opacity-20 fill-current">light_mode</span>
              )}
              <div className="text-[44px] leading-[44px] tracking-[-0.04em] font-bold text-primary">{time || '10:42'}</div>
              <div className="text-[14px] leading-[20px] mt-1 font-bold opacity-90">{dateStr || 'Thursday, June 4'}</div>
            </div>
            
            {/* Where to? Card */}
            <div className="bg-surface-container rounded-3xl p-5 shadow-sm flex-1 flex flex-col relative overflow-hidden min-h-0">
              <h2 className="text-[24px] leading-[32px] tracking-[-0.02em] text-primary mb-2 font-bold flex-shrink-0">Where to?</h2>
              <div className="flex flex-col gap-3 mt-auto w-full">
                <button className="bg-primary text-on-primary rounded-full h-[50px] w-full text-[17px] flex items-center justify-center gap-3 hover:bg-surface-tint transition-colors active:scale-95 shadow-md font-bold flex-shrink-0">
                  <span className="material-symbols-outlined text-2xl">school</span>
                  Dean's Office
                </button>
                <button className="bg-surface-variant text-on-surface-variant rounded-full h-[50px] w-full text-[17px] flex items-center justify-center gap-3 hover:bg-surface-container-highest transition-colors active:scale-95 shadow-sm border border-outline-variant font-bold flex-shrink-0">
                  <span className="material-symbols-outlined text-2xl">computer</span>
                  Computer Lab 03
                </button>
                <button className="bg-surface-variant text-on-surface-variant rounded-full h-[50px] w-full text-[17px] flex items-center justify-center gap-3 hover:bg-surface-container-highest transition-colors active:scale-95 shadow-sm border border-outline-variant font-bold flex-shrink-0">
                  <span className="material-symbols-outlined text-2xl">apartment</span>
                  Lecture Hall
                </button>
              </div>
            </div>
          </div>
          
          {/* Middle Column: Events Carousel & Microphone — flex-1 fills freed space */}
          <div className="flex-1 h-full min-h-0 flex flex-col gap-6 min-w-0">
            
            <div className="bg-secondary-container rounded-3xl shadow-sm flex-1 overflow-hidden relative flex flex-col min-h-0">
              {isConnected ? (
                <div className="flex-1 flex flex-col relative h-full bg-surface-container pt-4">
                  {isAgentInitializing && (
                    <div className="absolute inset-0 overflow-hidden pointer-events-none transition-all duration-1000 ease-in-out animate-pulse z-0">
                      <div className="absolute bottom-0 left-1/4 w-[400px] h-[400px] bg-indigo-500 opacity-60 mix-blend-screen rounded-full filter blur-[80px] animate-blob translate-y-1/2"></div>
                      <div className="absolute bottom-0 left-1/2 w-[450px] h-[400px] bg-purple-500 opacity-60 mix-blend-screen rounded-full filter blur-[80px] animate-blob animation-delay-2000 -translate-x-1/2 translate-y-1/2"></div>
                      <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] bg-pink-500 opacity-60 mix-blend-screen rounded-full filter blur-[80px] animate-blob animation-delay-4000 translate-y-1/2"></div>
                    </div>
                  )}

                  <ScrollArea ref={scrollAreaRef} className="flex-1 px-4 relative z-10">
                    <ChatTranscript messages={messages} transcriptions={transcriptions} stagingText={stagingText} isLoading={false} className="space-y-4 pb-4" />
                  </ScrollArea>
                </div>
              ) : (
                <div 
                  className="absolute inset-0 z-0 flex flex-col"
                  onTouchStart={onTouchStart}
                  onTouchMove={onTouchMove}
                  onTouchEnd={onTouchEnd}
                >
                  <div className="absolute inset-0 z-0 bg-secondary-container bg-black">
                {fbPosts.length > 0 ? (
                  fbPosts.map((post, index) => (
                    <img 
                      key={post.id}
                      alt="Facebook Post" 
                      className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-1000 ease-in-out ${index === currentSlide ? 'opacity-100' : 'opacity-0'}`} 
                      src={post.full_picture} 
                    />
                  ))
                ) : (
                  <div className="absolute inset-0 w-full h-full bg-gradient-to-tr from-surface-variant/80 via-surface/40 to-surface-container/80 animate-breathe"></div>
                )}
              </div>
              <div className="relative z-10 p-6 flex flex-col h-full bg-gradient-to-t from-black/80 via-black/30 to-transparent text-white">
                <div className="mt-auto">

                  {fbPosts.length > 0 ? (
                    <>
                      <h3 className="text-[20px] font-normal leading-tight mb-2 line-clamp-3 opacity-90">{fbPosts[currentSlide].message}</h3>
                      
                      {fbPosts[currentSlide].description && (
                        <p className="text-[14px] opacity-80 mb-2 line-clamp-2">{fbPosts[currentSlide].description}</p>
                      )}
                      
                      {fbPosts[currentSlide].extracted_date && (
                        <p className="text-[13px] font-semibold text-indigo-300 mb-1 drop-shadow-md">
                          📅 {fbPosts[currentSlide].extracted_date} {fbPosts[currentSlide].extracted_time ? `• ${fbPosts[currentSlide].extracted_time}` : ''}
                        </p>
                      )}
                      
                      {fbPosts[currentSlide].extracted_location && (
                        <p className="text-[13px] font-semibold text-purple-300 mb-3 drop-shadow-md">
                          📍 {fbPosts[currentSlide].extracted_location}
                        </p>
                      )}

                      <p className="text-[11px] opacity-60">Posted on: {new Date(fbPosts[currentSlide].created_time).toLocaleDateString()}</p>
                    </>
                  ) : (
                    <div className="space-y-3 animate-breathe opacity-60">
                      <div className="h-7 bg-white/30 rounded-md w-3/4"></div>
                      <div className="h-5 bg-white/30 rounded-md w-1/2"></div>
                      <div className="h-4 bg-white/30 rounded-md w-1/4 mt-4"></div>
                    </div>
                  )}
                </div>
              </div>
              {/* Carousel Indicators */}
              {fbPosts.length > 1 && (
                <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-2 z-20">
                  {fbPosts.map((_, idx) => (
                    <div 
                      key={idx} 
                      onClick={() => setCurrentSlide(idx)}
                      className={`w-2.5 h-2.5 rounded-full cursor-pointer transition-all ${idx === currentSlide ? 'bg-on-secondary scale-110' : 'bg-on-secondary/50 hover:bg-on-secondary/80 scale-100'}`}
                    ></div>
                  ))}
                </div>
              )}
              {/* Facebook Logo Watermark */}
              {fbPosts.length > 0 && !fbPosts[currentSlide]?.isLocal && (
                <div className="absolute bottom-4 right-4 z-20 text-[#1877F2] bg-white rounded-full p-[2px] shadow-lg flex items-center justify-center pointer-events-none">
                  <svg viewBox="0 0 24 24" fill="currentColor" className="w-7 h-7"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.469h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.469h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
                </div>
              )}
                </div>
              )}
            </div>
            {/* Microphone Action Area */}
            <div className={`flex-shrink-0 min-h-[96px] h-auto py-4 flex items-center justify-center rounded-3xl shadow-sm relative px-4 overflow-hidden transition-all duration-700 ${isConnected ? 'bg-surface-container border border-primary/20' : 'bg-surface-container-low'}`}>
              
              {/* Gemini-style Wavy Gradient Background */}
              <div className={`absolute inset-0 overflow-hidden pointer-events-none transition-all duration-300 ease-in-out ${isThinking ? 'animate-pulse' : ''} ${!isConnected ? 'animate-breathe' : ''}`} style={isConnected ? { transform: `scale(${pulseScale})`, opacity: pulseOpacity } : undefined}>
                <div className="absolute top-1/2 left-1/4 w-[400px] h-[250px] bg-indigo-500 opacity-80 mix-blend-screen rounded-full filter blur-[60px] animate-blob -translate-y-1/2"></div>
                <div className="absolute top-1/2 left-1/2 w-[450px] h-[250px] bg-purple-500 opacity-80 mix-blend-screen rounded-full filter blur-[60px] animate-blob animation-delay-2000 -translate-x-1/2 -translate-y-1/2"></div>
                <div className="absolute top-1/2 right-1/4 w-[400px] h-[250px] bg-pink-500 opacity-80 mix-blend-screen rounded-full filter blur-[60px] animate-blob animation-delay-4000 -translate-y-1/2"></div>
              </div>

              <div className="w-full flex justify-center items-center text-center text-[21px] font-extrabold text-on-surface dark:text-gray-100 tracking-tight leading-[1.2] min-h-[64px] relative z-10 pl-4 pr-20">
                {!isConnected ? (
                  <div className="relative w-full overflow-hidden flex items-center justify-center h-full min-h-[64px]">
                    {STANDBY_PROMPTS.map((prompt, index) => (
                      <div 
                        key={index}
                        className={`absolute inset-0 flex items-center justify-center text-center transition-all duration-1000 ease-in-out ${
                          currentPromptIndex === index 
                            ? 'opacity-100 translate-y-0 scale-100' 
                            : 'opacity-0 translate-y-4 scale-95 pointer-events-none'
                        }`}
                      >
                        {prompt}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="w-full flex justify-center break-words leading-[1.2] max-w-xl text-center">
                    {stagingText || ''}
                  </div>
                )}
              </div>
              <div className="absolute right-4 z-10">
                <button 
                  onClick={() => isConnected ? end() : start()}
                  className={`w-[56px] h-[56px] text-white rounded-full flex items-center justify-center shadow-xl hover:scale-105 transition-transform active:scale-95 border-none ${isConnected ? 'bg-error animate-pulse shadow-error/30' : 'bg-primary shadow-primary/30'}`}
                >
                  <span className="material-symbols-outlined text-3xl fill-current">{isConnected ? 'mic_off' : 'mic'}</span>
                </button>
              </div>
            </div>
          </div>
          
          {/* Right Column: Faculty News / Focused Poster — expands when poster is focused */}
          <div
            className="h-full min-h-0 flex-shrink-0 overflow-hidden transition-all duration-500 ease-in-out"
            style={{ width: focusedEvent ? '42%' : '25%' }}
          >
            <div className="bg-primary text-on-primary rounded-3xl shadow-md h-full flex flex-col border-4 border-primary-container/30 min-h-0 overflow-hidden relative">
              {focusedEvent ? (
                /* Full poster view */
                <>
                  {/* Poster image fills top */}
                  <div className="relative flex-1 min-h-0">
                    <img
                      src={focusedEvent.full_picture}
                      alt={focusedEvent.message}
                      className="w-full h-full object-cover"
                    />
                    {/* Back button */}
                    <button
                      onClick={() => setFocusedEvent(null)}
                      className="absolute top-3 left-3 z-10 bg-black/50 hover:bg-black/70 text-white rounded-full px-3 py-1.5 text-[12px] font-bold flex items-center gap-1.5 transition-colors backdrop-blur-sm"
                    >
                      <span className="material-symbols-outlined text-[16px]">arrow_back</span>
                      Back
                    </button>

                  </div>
                  {/* Event details below image */}
                  <div className="flex-shrink-0 p-4 bg-primary">
                    <p className="text-on-primary font-bold text-[16px] leading-snug mb-1">{focusedEvent.message}</p>
                    {focusedEvent.description && (
                      <p className="text-on-primary/75 text-[12px] leading-relaxed line-clamp-3 mb-2">{focusedEvent.description}</p>
                    )}
                    <div className="flex flex-wrap gap-2">
                      {focusedEvent.extracted_date && (
                        <span className="bg-primary-container text-on-primary-container px-2 py-1 rounded-full text-[11px] font-semibold">
                          📅 {focusedEvent.extracted_date}
                        </span>
                      )}
                      {focusedEvent.extracted_location && (
                        <span className="bg-primary-container text-on-primary-container px-2 py-1 rounded-full text-[11px] font-semibold">
                          📍 {focusedEvent.extracted_location}
                        </span>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                /* Normal news list */
                <>
                  {/* Header */}
                  <div className="flex-shrink-0 px-5 pt-5 pb-3">
                    <h2 className="text-[28px] font-black text-on-primary tracking-tight flex items-center gap-2">
                      <span className="material-symbols-outlined text-4xl">campaign</span>
                      Faculty News
                    </h2>
                  </div>

                  {/* Category color map */}
                  <div className="flex-1 flex flex-col gap-3 overflow-hidden px-4 pb-5">
                    {localPosts.slice(0, 3).map((post, i) => {
                      const categoryColors: Record<string, string> = {
                        events: 'from-violet-500 to-indigo-500',
                        competitions: 'from-orange-500 to-rose-500',
                        posts: 'from-teal-500 to-cyan-500',
                      };
                      const accent = categoryColors[post.category] || 'from-primary to-primary-container';
                      return (
                        <button
                          key={post.id}
                          className="w-full text-left rounded-2xl overflow-hidden cursor-pointer active:scale-[0.97] transition-transform focus:outline-none"
                          onClick={() => handleNewsClick(post)}
                        >
                          {/* Card: image thumbnail + text side by side */}
                          <div className="flex bg-white/10 hover:bg-white/20 transition-colors duration-150">
                            {/* Thumbnail */}
                            <div className="relative w-[80px] flex-shrink-0 overflow-hidden">
                              <img
                                src={post.full_picture}
                                alt={post.message}
                                className="w-full h-full object-cover min-h-[80px]"
                              />
                              {/* Gradient accent bar on left edge */}
                              <div className={`absolute inset-y-0 left-0 w-1 bg-gradient-to-b ${accent}`} />
                            </div>
                            {/* Text content */}
                            <div className="flex-1 p-3 min-w-0">
                              <div className="flex items-center gap-1.5 mb-1.5">
                                <span className={`inline-block w-2 h-2 rounded-full bg-gradient-to-br ${accent} flex-shrink-0`} />
                                <span className="text-[10px] font-black uppercase tracking-[0.12em] text-on-primary/70">
                                  {post.category.replace(/s$/, '')}
                                </span>
                                {post.extracted_date && (
                                  <span className="ml-auto text-[10px] font-semibold text-on-primary/50 flex-shrink-0">
                                    {post.extracted_date.substring(0, 6)}
                                  </span>
                                )}
                              </div>
                              <p className="text-[14px] font-bold text-on-primary leading-tight line-clamp-2">
                                {post.message}
                              </p>
                              {post.description && (
                                <p className="text-[11px] text-on-primary/60 mt-1 line-clamp-1">
                                  {post.description}
                                </p>
                              )}
                            </div>
                          </div>
                        </button>
                      );
                    })}
                    {localPosts.length === 0 && (
                      <div className="flex-1 flex flex-col items-center justify-center gap-3 opacity-50">
                        <span className="material-symbols-outlined text-5xl">newspaper</span>
                        <p className="text-[14px] italic text-center">No recent news.<br/>Upload a poster to get started.</p>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Upload Poster QR Modal */}
      {isUploadModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-surface text-on-surface p-8 rounded-3xl shadow-2xl max-w-md w-full relative animate-in zoom-in-95 duration-200">
            <button 
              onClick={() => setIsUploadModalOpen(false)}
              className="absolute top-4 right-4 text-on-surface-variant hover:text-on-surface bg-surface-variant/50 hover:bg-surface-variant p-2 rounded-full transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
            <div className="flex flex-col items-center text-center space-y-6">
              <div className="bg-primary/10 p-4 rounded-full">
                <UploadCloud className="w-8 h-8 text-primary" />
              </div>
              <div>
                <h2 className="text-2xl font-bold mb-2">Upload a Poster</h2>
                <p className="text-on-surface-variant">Scan this QR code with your phone to quickly upload an event poster to the Kiosk.</p>
              </div>
              <div className="bg-white p-4 rounded-2xl shadow-sm">
                <QRCodeSVG value={qrUrl} size={200} />
              </div>
              <p className="text-sm font-medium opacity-60">or visit<br/><span className="text-primary">{qrUrl}</span></p>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
