'use client';

import React from 'react';
import { 
  Satellite, Sparkles, LayoutDashboard, Image as ImageIcon, 
  SplitSquareHorizontal, Clock, History, Bookmark 
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  showToast: (msg: string) => void;
  onNewQuery: () => void;
}

export default function Sidebar({ activeTab, setActiveTab, showToast, onNewQuery }: SidebarProps) {
  return (
    <aside className="w-[260px] bg-[#111217] border-r border-[#1E1F27] flex flex-col shrink-0 z-20">
      {/* Logo */}
      <div className="h-16 flex items-center px-6 gap-3 border-b border-[#1E1F27] shrink-0">
        <Satellite className="text-indigo-400" size={24} />
        <div>
          <h1 className="text-white font-bold text-lg tracking-wide leading-none">SatQuery</h1>
          <p className="text-[10px] text-slate-500 mt-1">Ask. Analyze. Understand.</p>
        </div>
      </div>

      {/* New Query Button */}
      <button 
        suppressHydrationWarning
        onClick={onNewQuery}
        className="mx-4 mt-6 mb-4 flex items-center gap-3 bg-gradient-to-r from-[#201C3B] to-[#17152A] border border-[#3A2F63] text-[#A694F5] p-3 rounded-xl hover:brightness-110 transition-all shadow-[0_0_15px_rgba(99,102,241,0.1)] group"
      >
        <Sparkles size={18} className="group-hover:animate-pulse" />
        <span className="font-semibold text-sm">New Query</span>
      </button>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-4 py-2 space-y-6">
        <div>
          <div onClick={() => { setActiveTab('dashboard'); showToast('Dashboard module is locked in demo mode'); }} className={`flex items-center gap-3 px-2 py-2 rounded-lg cursor-pointer transition-colors mb-2 ${activeTab === 'dashboard' ? 'bg-[#201C3B]/50 text-indigo-300 border border-indigo-500/20' : 'text-slate-300 hover:bg-[#1E1F27]'}`}>
            <LayoutDashboard size={18} className={activeTab === 'dashboard' ? 'text-indigo-400' : 'text-slate-500'} />
            <span className="text-sm font-medium">Dashboard</span>
          </div>
        </div>
        
        <div>
          <h3 className="text-[10px] font-bold text-slate-600 tracking-wider mb-2 px-2 uppercase">Data</h3>
          <div className="space-y-1">
            <div onClick={() => { setActiveTab('scenes'); showToast('Imagery catalog is locked in demo mode'); }} className={`flex items-center gap-3 px-2 py-2 rounded-lg cursor-pointer transition-colors ${activeTab === 'scenes' ? 'bg-[#201C3B]/50 text-indigo-300 border border-indigo-500/20' : 'text-slate-300 hover:bg-[#1E1F27]'}`}>
              <ImageIcon size={18} className={activeTab === 'scenes' ? 'text-indigo-400' : 'text-slate-500'} />
              <span className="text-sm font-medium">Scenes / Imagery</span>
            </div>
            <div onClick={() => setActiveTab('compare')} className={`flex items-center gap-3 px-2 py-2 rounded-lg cursor-pointer transition-colors ${activeTab === 'compare' ? 'bg-[#201C3B]/50 text-indigo-300 border border-indigo-500/20' : 'text-slate-300 hover:bg-[#1E1F27]'}`}>
              <SplitSquareHorizontal size={18} className={activeTab === 'compare' ? 'text-indigo-400' : 'text-slate-500'} />
              <span className="text-sm font-medium">Compare</span>
            </div>
            <div onClick={() => { setActiveTab('timeline'); showToast('Timeline slider is locked in demo mode'); }} className={`flex items-center gap-3 px-2 py-2 rounded-lg cursor-pointer transition-colors ${activeTab === 'timeline' ? 'bg-[#201C3B]/50 text-indigo-300 border border-indigo-500/20' : 'text-slate-300 hover:bg-[#1E1F27]'}`}>
              <Clock size={18} className={activeTab === 'timeline' ? 'text-indigo-400' : 'text-slate-500'} />
              <span className="text-sm font-medium">Timeline</span>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-[10px] font-bold text-slate-600 tracking-wider mb-2 px-2 uppercase">History</h3>
          <div className="space-y-1">
            <div onClick={() => { setActiveTab('recent'); showToast('Recent queries will be available in V1'); }} className={`flex items-center gap-3 px-2 py-2 rounded-lg cursor-pointer transition-colors ${activeTab === 'recent' ? 'bg-[#201C3B]/50 text-indigo-300 border border-indigo-500/20' : 'text-slate-300 hover:bg-[#1E1F27]'}`}>
              <History size={18} className={activeTab === 'recent' ? 'text-indigo-400' : 'text-slate-500'} />
              <span className="text-sm font-medium">Recent Queries</span>
            </div>
            <div onClick={() => { setActiveTab('saved'); showToast('Saved results will be available in V1'); }} className={`flex items-center gap-3 px-2 py-2 rounded-lg cursor-pointer transition-colors ${activeTab === 'saved' ? 'bg-[#201C3B]/50 text-indigo-300 border border-indigo-500/20' : 'text-slate-300 hover:bg-[#1E1F27]'}`}>
              <Bookmark size={18} className={activeTab === 'saved' ? 'text-indigo-400' : 'text-slate-500'} />
              <span className="text-sm font-medium">Saved Results</span>
            </div>
          </div>
        </div>
      </nav>

      {/* Bottom Promo Card */}
      <div className="mx-4 mb-6 p-4 rounded-xl bg-gradient-to-br from-[#1A1C29] to-[#111217] border border-[#1E1F27] relative overflow-hidden shrink-0">
        <div className="absolute -right-4 -bottom-4 w-24 h-24 bg-indigo-500/10 rounded-full blur-xl pointer-events-none" />
        <h4 className="text-indigo-300 font-semibold text-sm mb-2 relative z-10">SatQuery AI</h4>
        <p className="text-[11px] text-slate-400 leading-relaxed relative z-10">
          Natural language interface for satellite intelligence powered by advanced AI models.
        </p>
        <div className="mt-3 relative z-10 text-right">
           <Satellite size={24} className="text-indigo-500/30 inline-block rotate-12" />
        </div>
      </div>
    </aside>
  );
}
