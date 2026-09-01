'use client';

import React, { useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChevronDown, ChevronLeft, ChevronRight, Plus, Minus, Home, Maximize, Crosshair, Layers } from 'lucide-react';
import { QueryType, ViewMode } from '../app/types';

interface ImageViewerProps {
  queryType: QueryType;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  sliderPosition: number;
  setSliderPosition: (pos: number) => void;
  isAnalyzing: boolean;
  hasResult: boolean;
}

export default function ImageViewer({ 
  queryType, 
  viewMode, 
  setViewMode, 
  sliderPosition, 
  setSliderPosition, 
  isAnalyzing, 
  hasResult 
}: ImageViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const startSliding = (e: React.MouseEvent) => {
    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      let newPos = ((moveEvent.clientX - rect.left) / rect.width) * 100;
      newPos = Math.max(0, Math.min(100, newPos));
      setSliderPosition(newPos);
    };
    const handleMouseUp = () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  return (
    <div className="flex-1 bg-[#111217] rounded-xl border border-[#1E1F27] flex flex-col overflow-hidden relative shadow-lg">
      
      {/* Viewer Header */}
      <div className="h-14 border-b border-[#1E1F27] flex items-center justify-between px-5 shrink-0 bg-[#111217] z-20">
        <h2 className="text-xs font-bold text-white tracking-widest">IMAGE VIEWER</h2>
        
        <div className="flex items-center gap-6">
          {queryType === 'change_detection' && (
            <>
              {/* Date Compare Selectors */}
              <div className="flex items-center gap-2 text-sm">
                <button suppressHydrationWarning className="flex items-center gap-2 bg-[#1A1C24] px-3 py-1.5 rounded border border-[#222432] text-slate-200 hover:bg-[#222432] transition-colors">
                  15 Jun 2023 <ChevronDown size={14} className="text-slate-500"/>
                </button>
                <span className="text-slate-500 text-xs font-medium">vs</span>
                <button suppressHydrationWarning className="flex items-center gap-2 bg-[#1A1C24] px-3 py-1.5 rounded border border-[#222432] text-slate-200 hover:bg-[#222432] transition-colors">
                  15 Jun 2026 <ChevronDown size={14} className="text-slate-500"/>
                </button>
              </div>
              
              {/* Segmented Control */}
              <div className="flex items-center bg-[#0B0C10] p-1 rounded-lg border border-[#1E1F27]">
                <button suppressHydrationWarning onClick={() => setViewMode('before')} className={`px-3 py-1 text-xs font-medium rounded transition-colors ${viewMode === 'before' ? 'bg-[#2D2B55] text-indigo-200 border border-indigo-500/30 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}>Before</button>
                <button suppressHydrationWarning onClick={() => setViewMode('after')} className={`px-3 py-1 text-xs font-medium rounded transition-colors ${viewMode === 'after' ? 'bg-[#2D2B55] text-indigo-200 border border-indigo-500/30 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}>After</button>
                <button suppressHydrationWarning onClick={() => setViewMode('split')} className={`px-3 py-1 text-xs font-medium rounded transition-colors ${viewMode === 'split' ? 'bg-[#2D2B55] text-indigo-200 border border-indigo-500/30 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}>Split</button>
                <button suppressHydrationWarning onClick={() => setViewMode('slider')} className={`px-3 py-1 text-xs font-medium rounded transition-colors ${viewMode === 'slider' ? 'bg-[#2D2B55] text-indigo-200 border border-indigo-500/30 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}>Slider</button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Viewer Canvas */}
      <div className="flex-1 relative bg-[#1A1C23] overflow-hidden" ref={containerRef}>
        
        {/* Floating Left Controls */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 flex flex-col bg-[#0B0C10]/80 backdrop-blur-md rounded-lg border border-[#222432] overflow-hidden z-30 shadow-lg">
          <button suppressHydrationWarning className="p-2.5 text-slate-400 hover:text-white hover:bg-[#1E1F27] transition-colors border-b border-[#222432]"><Plus size={18} /></button>
          <button suppressHydrationWarning className="p-2.5 text-slate-400 hover:text-white hover:bg-[#1E1F27] transition-colors border-b border-[#222432]"><Minus size={18} /></button>
          <button suppressHydrationWarning className="p-2.5 text-slate-400 hover:text-white hover:bg-[#1E1F27] transition-colors border-b border-[#222432]"><Home size={18} /></button>
          <button suppressHydrationWarning className="p-2.5 text-slate-400 hover:text-white hover:bg-[#1E1F27] transition-colors border-b border-[#222432]"><Maximize size={18} /></button>
          <button suppressHydrationWarning className="p-2.5 text-slate-400 hover:text-white hover:bg-[#1E1F27] transition-colors"><Crosshair size={18} /></button>
        </div>

        {/* Floating Right Control */}
        <button suppressHydrationWarning className="absolute right-4 bottom-4 z-30 flex items-center gap-2 bg-[#0B0C10]/90 backdrop-blur-md px-4 py-2 rounded-lg border border-[#222432] text-sm text-slate-300 hover:text-white transition-colors shadow-lg">
          <Layers size={16} /> Layers
        </button>

        {/* Base Layer */}
        <div className={`absolute inset-0 bg-[#252830] bg-[url('https://images.unsplash.com/photo-1524661135-423995f22d0b?q=80&w=2000&auto=format&fit=crop')] bg-cover bg-center ${queryType === 'cross_modal' ? 'brightness-50 contrast-150 grayscale-[100%] hue-rotate-15' : 'brightness-75 contrast-125 grayscale-[30%]'} ${viewMode === 'before' && queryType === 'change_detection' ? 'hidden' : ''}`}>
          <div className={`absolute top-4 ${queryType === 'change_detection' ? 'right-4' : queryType === 'cross_modal' ? 'right-4' : 'left-4'} bg-[#0B0C10]/80 backdrop-blur-md px-3 py-1.5 rounded-md text-xs font-medium text-white border border-[#222432] z-10 shadow-lg`}>
            {queryType === 'change_detection' ? '15 Jun 2026' : queryType === 'cross_modal' ? 'Sentinel-1 SAR' : 'Current Scene'} <span className="text-slate-400 ml-1">| {queryType === 'cross_modal' ? 'C-Band' : 'Sentinel-2'}</span>
          </div>

          {/* Scanning Animation */}
          <AnimatePresence>
            {isAnalyzing && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 pointer-events-none z-10"
              >
                <motion.div 
                  initial={{ top: '0%' }}
                  animate={{ top: '100%' }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                  className="absolute left-0 right-0 h-1 bg-cyan-400/50 shadow-[0_0_20px_rgba(34,211,238,1)]"
                />
                <div className="absolute inset-0 bg-cyan-900/10 mix-blend-overlay" />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Detected Bounding Boxes */}
          <AnimatePresence>
            {hasResult && (queryType === 'grounding' || queryType === 'change_detection') && (
              <motion.div 
                initial="hidden"
                animate="visible"
                variants={{
                  visible: { transition: { staggerChildren: 0.1 } }
                }}
                className="absolute inset-0 z-10 pointer-events-none"
              >
                <motion.div variants={{ hidden: { opacity: 0, scale: 0.8 }, visible: { opacity: 1, scale: 1, transition: { type: 'spring' } } }} className={`absolute top-[22%] right-[25%] w-[12%] h-[15%] border-2 ${queryType === 'change_detection' ? 'border-[#FF5A65] bg-[#FF5A65]/10 shadow-[0_0_15px_rgba(255,90,101,0.5)]' : 'border-[#4F46E5] bg-[#4F46E5]/10 shadow-[0_0_15px_rgba(79,70,229,0.5)]'} rounded-sm flex items-center justify-center`}>
                   <div className={`absolute -top-6 ${queryType === 'change_detection' ? 'bg-[#1A151C] border-[#3A2228] text-[#FF5A65]' : 'bg-[#151522] border-[#2A2B44] text-[#818CF8]'} border px-2 py-0.5 rounded text-[10px] font-bold whitespace-nowrap shadow-md backdrop-blur-sm`}>
                     {queryType === 'change_detection' ? 'New Building 92%' : 'Building 92%'}
                   </div>
                </motion.div>
                <motion.div variants={{ hidden: { opacity: 0, scale: 0.8 }, visible: { opacity: 1, scale: 1 } }} className={`absolute top-[45%] right-[38%] w-[10%] h-[12%] border-2 ${queryType === 'change_detection' ? 'border-[#FF5A65] bg-[#FF5A65]/10 shadow-[0_0_15px_rgba(255,90,101,0.5)]' : 'border-[#4F46E5] bg-[#4F46E5]/10 shadow-[0_0_15px_rgba(79,70,229,0.5)]'} rounded-sm`} />
                <motion.div variants={{ hidden: { opacity: 0, scale: 0.8 }, visible: { opacity: 1, scale: 1 } }} className={`absolute top-[48%] right-[22%] w-[9%] h-[11%] border-2 ${queryType === 'change_detection' ? 'border-[#FF5A65] bg-[#FF5A65]/10 shadow-[0_0_15px_rgba(255,90,101,0.5)]' : 'border-[#4F46E5] bg-[#4F46E5]/10 shadow-[0_0_15px_rgba(79,70,229,0.5)]'} rounded-sm`} />
                <motion.div variants={{ hidden: { opacity: 0, scale: 0.8 }, visible: { opacity: 1, scale: 1 } }} className={`absolute bottom-[28%] right-[28%] w-[8%] h-[9%] border-2 ${queryType === 'change_detection' ? 'border-[#FF5A65] bg-[#FF5A65]/10 shadow-[0_0_15px_rgba(255,90,101,0.5)]' : 'border-[#4F46E5] bg-[#4F46E5]/10 shadow-[0_0_15px_rgba(79,70,229,0.5)]'} rounded-sm`} />
                <motion.div variants={{ hidden: { opacity: 0, scale: 0.8 }, visible: { opacity: 1, scale: 1 } }} className={`absolute bottom-[40%] right-[45%] w-[5%] h-[6%] border-2 ${queryType === 'change_detection' ? 'border-[#FF5A65] bg-[#FF5A65]/10 shadow-[0_0_15px_rgba(255,90,101,0.5)]' : 'border-[#4F46E5] bg-[#4F46E5]/10 shadow-[0_0_15px_rgba(79,70,229,0.5)]'} rounded-sm`} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* 2023 Image overlay */}
        {(queryType === 'change_detection' || queryType === 'cross_modal') && (
          <div 
            className={`absolute inset-0 bg-[#1A1C23] bg-[url('https://images.unsplash.com/photo-1524661135-423995f22d0b?q=80&w=2000&auto=format&fit=crop')] bg-cover bg-center ${queryType === 'cross_modal' ? 'brightness-75 contrast-125 grayscale-[10%]' : 'brightness-50 contrast-100 grayscale-[40%] hue-rotate-15 sepia-[20%]'} z-20 ${viewMode === 'after' && queryType === 'change_detection' ? 'hidden' : ''}`}
            style={{ 
              clipPath: queryType === 'cross_modal' ? 'inset(0 50% 0 0)' : (viewMode === 'slider' ? `inset(0 ${100 - sliderPosition}% 0 0)` : viewMode === 'split' ? 'inset(0 50% 0 0)' : 'inset(0 0 0 0)')
            }}
          >
            <div className={`absolute inset-y-0 right-0 w-px bg-white/20 shadow-[2px_0_15px_rgba(0,0,0,0.8)] ${(queryType === 'cross_modal' || viewMode === 'slider' || viewMode === 'split') ? '' : 'hidden'}`} />
            <div className="absolute top-4 left-4 bg-[#0B0C10]/80 backdrop-blur-md px-3 py-1.5 rounded-md text-xs font-medium text-white border border-[#222432] shadow-lg">
              {queryType === 'cross_modal' ? 'Sentinel-2 OPTICAL' : '15 Jun 2023'} <span className="text-slate-400 ml-1">| {queryType === 'cross_modal' ? 'Multispectral' : 'Sentinel-2'}</span>
            </div>
          </div>
        )}

        {/* Slider Handle */}
        {queryType === 'change_detection' && viewMode === 'slider' && (
          <div 
            className="absolute top-0 bottom-0 w-px bg-white/50 cursor-ew-resize z-30 group"
            style={{ left: `${sliderPosition}%` }}
            onMouseDown={startSliding}
          >
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 bg-[#111217] rounded-full border border-[#2A2B35] flex items-center justify-center text-slate-300 shadow-[0_0_20px_rgba(0,0,0,0.5)] group-hover:scale-110 group-hover:border-indigo-500/50 transition-all">
              <ChevronLeft size={14} className="-mr-0.5" />
              <ChevronRight size={14} className="-ml-0.5" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
