'use client';

import { useSessionContext, useSessionMessages } from '@livekit/components-react';
import React, { useState, useEffect } from 'react';

export function KioskView() {
  const session = useSessionContext();
  const { isConnected, start, end } = session;
  const { messages } = useSessionMessages(session);
  const [time, setTime] = useState('');
  const [dateStr, setDateStr] = useState('');

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

  return (
    <div className="bg-background text-on-background w-full h-full overflow-hidden flex flex-col select-none" style={{ fontFamily: 'Inter, sans-serif' }}>
      {/* Top App Bar */}
      <header className="bg-surface/80 backdrop-blur-md fixed top-0 w-full z-50 flex justify-between items-center px-8 h-[56px]">
        <div className="text-3xl font-bold text-primary">NEma</div>
        <div className="flex items-center gap-4">
          {isConnected && (
            <span className="text-green-500 font-semibold animate-pulse">Connected</span>
          )}
        </div>
      </header>
      
      {/* Main Content Area - Bento Grid */}
      <main className="flex-1 mt-[56px] mb-[160px] px-8 py-4 h-[calc(100vh-216px)]">
        <div className="grid grid-cols-12 gap-6 h-full">
          {/* Left Column: Clock & Navigation */}
          <div className="col-span-4 flex flex-col gap-6 h-full">
            {/* Clock Card */}
            <div className="bg-primary-container text-on-primary-container rounded-3xl p-6 flex flex-col items-center justify-center shadow-sm relative overflow-hidden flex-shrink-0">
              <span className="material-symbols-outlined absolute top-4 right-4 text-6xl opacity-20 fill-current">light_mode</span>
              <div className="text-[80px] leading-[80px] tracking-[-0.04em] font-bold text-primary">{time || '10:42'}</div>
              <div className="text-[24px] leading-[30px] mt-2 font-bold">{dateStr || 'Thursday, June 4'}</div>
            </div>
            
            {/* Where to? Card */}
            <div className="bg-surface-container rounded-3xl p-6 shadow-sm flex-1 flex flex-col relative overflow-y-auto kiosk-scrollbar">
              <h2 className="text-[36px] leading-[44px] tracking-[-0.02em] text-primary mb-4 font-bold">Where to?</h2>
              <div className="flex flex-col gap-4 mt-auto">
                <button className="bg-primary text-on-primary rounded-full h-[56px] text-[24px] flex items-center justify-center gap-4 hover:bg-surface-tint transition-colors active:scale-95 shadow-md font-bold">
                  <span className="material-symbols-outlined text-3xl">school</span>
                  Dean's Office
                </button>
                <button className="bg-surface-variant text-on-surface-variant rounded-full h-[56px] text-[24px] flex items-center justify-center gap-4 hover:bg-surface-container-highest transition-colors active:scale-95 shadow-sm border border-outline-variant font-bold">
                  <span className="material-symbols-outlined text-3xl">computer</span>
                  Computer Lab 03
                </button>
                <button className="bg-surface-variant text-on-surface-variant rounded-full h-[56px] text-[24px] flex items-center justify-center gap-4 hover:bg-surface-container-highest transition-colors active:scale-95 shadow-sm border border-outline-variant font-bold">
                  <span className="material-symbols-outlined text-3xl">apartment</span>
                  Lecture Hall
                </button>
              </div>
            </div>
          </div>
          
          {/* Middle Column: Events Carousel */}
          <div className="col-span-4 h-full">
            <div className="bg-secondary-container rounded-3xl shadow-sm h-full overflow-hidden relative flex flex-col">
              <div className="absolute inset-0 z-0 bg-secondary-container">
                <img alt="College Event" className="w-full h-full object-cover opacity-80 mix-blend-multiply" src="https://lh3.googleusercontent.com/aida-public/AB6AXuASe7OPmposO-19UAIeU4spfafXd_IIkyengbRnIoJXP5vzcgsqBX4KhpYGHDv1RVod-dKhSD4LadBgQAlGEoyLGT5i8i3olLcgb8xypR5mcuEL1Q78xoqtkxWnKF9jhItfILnYltqiwrrLAeE3ZFxZ7nCEHNlwi6t2MOxghHruNkBxUQQYFFp_Rkb-PqnZNEPZKbK-jp7fxgCeZsKJJkieYur0T9mHyCpYbIlQ5BJ_1U1E1ZsWoHM1etOrM2fPLnCL8NLiGnhxxs4" />
              </div>
              <div className="relative z-10 p-8 flex flex-col h-full bg-gradient-to-t from-on-secondary-container/90 to-transparent text-on-secondary">
                <div className="mt-auto">
                  <span className="bg-secondary text-on-secondary px-3 py-1 rounded-full text-[16px] font-semibold inline-block mb-3">Campus Life</span>
                  <h3 className="text-[32px] font-bold leading-tight mb-2">Spring Festival Begins Next Week</h3>
                  <p className="text-[20px] opacity-90">Join us on the main quad for food, music, and activities. Open to all students and faculty.</p>
                </div>
              </div>
              {/* Carousel Indicators */}
              <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-2 z-20">
                <div className="w-2 h-2 rounded-full bg-on-secondary"></div>
                <div className="w-2 h-2 rounded-full bg-on-secondary/50"></div>
                <div className="w-2 h-2 rounded-full bg-on-secondary/50"></div>
              </div>
            </div>
          </div>
          
          {/* Right Column: Faculty News */}
          <div className="col-span-4 h-full">
            <div className="bg-primary text-on-primary rounded-3xl p-8 shadow-md h-full flex flex-col border-4 border-primary-container/30">
              <h2 className="text-[48px] font-bold text-on-primary mb-8 flex items-center gap-3">
                <span className="material-symbols-outlined text-5xl">campaign</span>
                Faculty News
              </h2>
              <div className="flex-1 overflow-y-auto pr-2 flex flex-col gap-8 kiosk-scrollbar">
                {/* News Item 1 */}
                <div className="group cursor-pointer">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-primary-container text-on-primary-container px-3 py-1 rounded-full text-[16px] font-semibold">Jun 2</span>
                    <span className="text-[16px] font-bold uppercase tracking-widest text-on-primary">Competition</span>
                  </div>
                  <h4 className="text-[32px] font-bold text-on-primary group-hover:text-primary-container transition-colors leading-tight">Robotics Team Wins Nationals</h4>
                </div>
                {/* News Item 2 */}
                <div className="group cursor-pointer">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-primary-container text-on-primary-container px-3 py-1 rounded-full text-[16px] font-semibold">May 28</span>
                    <span className="text-[16px] font-bold uppercase tracking-widest text-on-primary">Announcement</span>
                  </div>
                  <h4 className="text-[32px] font-bold text-on-primary group-hover:text-primary-container transition-colors leading-tight">New Grant Awarded to CS Dept</h4>
                </div>
                {/* News Item 3 */}
                <div className="group cursor-pointer">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-primary-container text-on-primary-container px-3 py-1 rounded-full text-[16px] font-semibold">May 25</span>
                    <span className="text-[16px] font-bold uppercase tracking-widest text-on-primary">Seminar</span>
                  </div>
                  <h4 className="text-[32px] font-bold text-on-primary group-hover:text-primary-container transition-colors leading-tight">Guest Lecture: Ethics in ML</h4>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
      
      {/* Footer Action Bar */}
      <footer className="fixed bottom-0 left-0 w-full bg-transparent h-[200px] flex flex-col items-center justify-end pb-12 z-50 pointer-events-none">
        <div className="relative w-full min-h-16 h-auto mb-4 pointer-events-auto flex justify-center items-end px-8 text-center text-[32px] font-bold text-on-background drop-shadow-md">
          {!isConnected ? (
            <div className="relative w-full h-16 overflow-hidden">
              <div className="greeting-text greeting-1 leading-normal">How can I help you?</div>
              <div className="greeting-text greeting-2 leading-normal">Tap the mic to ask a question!</div>
            </div>
          ) : (
            <div className="w-full flex justify-center break-words pb-2">
              {messages.filter(m => m.text).slice(-1)[0]?.text || 'Listening...'}
            </div>
          )}
        </div>
        <div className="pointer-events-auto flex justify-center w-full">
          <button 
            onClick={() => isConnected ? end() : start()}
            className={`w-[90px] h-[90px] text-on-primary rounded-full flex items-center justify-center shadow-xl hover:scale-105 transition-transform active:scale-95 z-50 border-none ${isConnected ? 'bg-error animate-pulse' : 'bg-primary animate-neon-pulse'}`}
            style={{ backgroundColor: isConnected ? '#ba1a1a' : 'rgb(116, 86, 96)' }}
          >
            <span className="material-symbols-outlined text-5xl fill-current">{isConnected ? 'mic_off' : 'mic'}</span>
          </button>
        </div>
      </footer>
    </div>
  );
}
