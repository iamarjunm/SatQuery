'use client';

import React from 'react';
import { Mic, Send } from 'lucide-react';

interface QueryInputProps {
  query: string;
  setQuery: (val: string) => void;
  isAnalyzing: boolean;
  handleAnalyze: (text: string) => void;
}

export default function QueryInput({ query, setQuery, isAnalyzing, handleAnalyze }: QueryInputProps) {
  return (
    <div className="shrink-0 flex flex-col gap-4">
      {/* Suggested Queries */}
      <div>
        <h3 className="text-[10px] font-bold text-slate-500 tracking-wider mb-2 uppercase">Suggested Queries</h3>
        <div className="flex flex-wrap items-center gap-2">
          <button 
            suppressHydrationWarning
            onClick={() => handleAnalyze('Describe the major land-cover types in this image.')}
            className="px-3 py-1.5 rounded-full border border-[#222432] bg-[#14151C] text-slate-300 hover:text-white hover:bg-[#1E1F27] transition-colors text-xs font-medium"
          >
            Describe land-cover
          </button>
          <button 
            suppressHydrationWarning
            onClick={() => handleAnalyze('Highlight the water body.')}
            className="px-3 py-1.5 rounded-full border border-[#222432] bg-[#14151C] text-slate-300 hover:text-white hover:bg-[#1E1F27] transition-colors text-xs font-medium"
          >
            Highlight water body
          </button>
          <button 
            suppressHydrationWarning
            onClick={() => handleAnalyze('What changed between these two dates?')}
            className="px-3 py-1.5 rounded-full border border-[#222432] bg-[#14151C] text-slate-300 hover:text-white hover:bg-[#1E1F27] transition-colors text-xs font-medium"
          >
            What changed?
          </button>
          <button 
            suppressHydrationWarning
            onClick={() => handleAnalyze('Use the optical and SAR images together to identify built-up and water-covered regions.')}
            className="px-3 py-1.5 rounded-full border border-[#222432] bg-[#14151C] text-slate-300 hover:text-white hover:bg-[#1E1F27] transition-colors text-xs font-medium"
          >
            Optical + SAR fusion
          </button>
        </div>
      </div>

      {/* Search Bar */}
      <div className="bg-[#111217] border border-[#1E1F27] rounded-xl p-3 flex flex-col gap-3 shadow-lg">
        <div className="flex items-center gap-3">
          <input 
            suppressHydrationWarning
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !isAnalyzing && handleAnalyze(query)}
            className="flex-1 bg-transparent text-lg text-white outline-none placeholder:text-slate-600 px-2" 
            placeholder="Ask anything about this satellite imagery..." 
            disabled={isAnalyzing}
          />
          <button suppressHydrationWarning className="p-3 rounded-lg bg-[#1A1C24] text-slate-400 hover:text-white hover:bg-[#222432] transition-colors border border-[#222432]">
            <Mic size={20} />
          </button>
          <button 
            suppressHydrationWarning
            onClick={() => handleAnalyze(query)}
            disabled={!query.trim() || isAnalyzing}
            className="p-3 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 transition-colors shadow-[0_0_15px_rgba(79,70,229,0.4)]"
          >
            {isAnalyzing ? (
               <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
               <Send size={20} />
            )}
          </button>
        </div>
        <div className="text-[11px] text-slate-500 px-2">
          Examples: <span className="text-slate-400 cursor-pointer hover:text-white transition-colors" onClick={() => handleAnalyze('Find buildings')}>Find buildings</span>, <span className="text-slate-400 cursor-pointer hover:text-white transition-colors" onClick={() => handleAnalyze('What changed?')}>What changed?</span> , <span className="text-slate-400 cursor-pointer hover:text-white transition-colors" onClick={() => handleAnalyze('Show roads')}>Show roads</span>, <span className="text-slate-400 cursor-pointer hover:text-white transition-colors" onClick={() => handleAnalyze('Identify water bodies')}>Identify water bodies</span>
        </div>
      </div>
    </div>
  );
}
