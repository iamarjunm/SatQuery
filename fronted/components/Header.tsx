'use client';

import React from 'react';
import { ChevronDown, HelpCircle, Settings } from 'lucide-react';

interface HeaderProps {
  showToast: (msg: string) => void;
}

export default function Header({ showToast }: HeaderProps) {
  return (
    <header className="h-16 flex items-center justify-between px-6 border-b border-[#1E1F27] shrink-0 bg-[#0B0C10]">
      <div className="flex items-center gap-3">
        <button suppressHydrationWarning className="flex items-center gap-2 bg-[#14151C] hover:bg-[#1E1F27] px-4 py-2 rounded-lg border border-[#222432] text-sm text-slate-200 transition-colors">
          Project Area 01 <ChevronDown size={14} className="text-slate-500 ml-2" />
        </button>
        <button suppressHydrationWarning className="flex items-center gap-2 bg-[#14151C] hover:bg-[#1E1F27] px-4 py-2 rounded-lg border border-[#222432] text-sm text-slate-200 transition-colors">
          <span className="text-slate-500">Dataset:</span> BigEarthNet (SAR/MSI) <ChevronDown size={14} className="text-slate-500 ml-1" />
        </button>
      </div>
      
      <div className="flex items-center gap-4">
        <button suppressHydrationWarning onClick={() => showToast('Help documentation not included in prototype')} className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-400 hover:text-white transition-colors rounded-lg border border-[#222432] bg-[#14151C]">
          <HelpCircle size={14} /> Help
        </button>
        <button suppressHydrationWarning onClick={() => showToast('Settings panel locked')} className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-400 hover:text-white transition-colors rounded-lg border border-[#222432] bg-[#14151C]">
          <Settings size={14} /> Settings
        </button>
        <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white text-xs font-bold border-2 border-[#1E1F27] cursor-pointer shadow-[0_0_10px_rgba(99,102,241,0.3)]">
          AD
        </div>
      </div>
    </header>
  );
}
