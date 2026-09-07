'use client';

import React from 'react';
import { ChevronDown, MapPin, Satellite } from 'lucide-react';
import { siteLabel } from '../lib/api';

interface HeaderProps {
  sites: string[];
  site: string;
  onSiteChange: (site: string) => void;
  sceneCount: number;
  loadError: string | null;
}

export default function Header({ sites, site, onSiteChange, sceneCount, loadError }: HeaderProps) {
  return (
    <header className="h-16 flex items-center justify-between px-6 border-b border-[#1E1F27] shrink-0 bg-[#0B0C10]">
      <div className="flex items-center gap-3">
        {/* Site picker: one entry per area the controller has chips for */}
        <label className="relative flex items-center gap-2 bg-[#14151C] hover:bg-[#1E1F27] px-4 py-2 rounded-lg border border-[#222432] text-sm text-slate-200 transition-colors cursor-pointer">
          <MapPin size={14} className="text-indigo-400" />
          <select
            suppressHydrationWarning
            value={site}
            onChange={e => onSiteChange(e.target.value)}
            className="bg-transparent outline-none appearance-none pr-6 cursor-pointer text-slate-200"
          >
            {sites.length === 0 && <option value={site}>Loading scenes...</option>}
            {sites.map(s => (
              <option key={s} value={s} className="bg-[#14151C]">{siteLabel(s)}</option>
            ))}
          </select>
          <ChevronDown size={14} className="text-slate-500 absolute right-3 pointer-events-none" />
        </label>
        <div className="flex items-center gap-2 bg-[#14151C] px-4 py-2 rounded-lg border border-[#222432] text-sm text-slate-200">
          <Satellite size={14} className="text-slate-500" />
          <span className="text-slate-500">Dataset:</span> Sentinel-2 L2A, 10 m
          <span className="text-slate-500 ml-1">| {sceneCount} chips</span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {loadError ? (
          <span className="text-xs text-[#FF5A65] max-w-[420px] truncate" title={loadError}>
            Backend unreachable: {loadError}
          </span>
        ) : (
          <span className="text-xs text-slate-500">Planner: LLM, validated plan · Agents over HTTP</span>
        )}
        <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white text-xs font-bold border-2 border-[#1E1F27] cursor-pointer shadow-[0_0_10px_rgba(99,102,241,0.3)]">
          SQ
        </div>
      </div>
    </header>
  );
}
