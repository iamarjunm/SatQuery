'use client';

import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { CheckCircle2, Circle, ChevronDown, Building2, Navigation, Trees, Copy, Sparkles } from 'lucide-react';
import { QueryType } from '../app/types';

interface ResultsPanelProps {
  query: string;
  isAnalyzing: boolean;
  hasResult: boolean;
  analysisStep: number;
  queryType: QueryType;
}

export default function ResultsPanel({ query, isAnalyzing, hasResult, analysisStep, queryType }: ResultsPanelProps) {
  return (
    <aside className="w-[340px] bg-[#111217] border-l border-[#1E1F27] flex flex-col shrink-0 overflow-y-auto z-20 shadow-[-10px_0_30px_rgba(0,0,0,0.5)]">
      <div className="p-5 border-b border-[#1E1F27] sticky top-0 bg-[#111217] z-10">
        <h2 className="text-xs font-bold text-white tracking-widest">AI ANALYSIS</h2>
      </div>

      {query || hasResult || isAnalyzing ? (
        <div className="pb-10">
          {/* Query Display */}
          <div className="mt-6 px-5">
            <h3 className="text-[10px] font-bold text-indigo-400 tracking-wider mb-3">QUERY</h3>
            <div className="bg-[#1A1C24] border border-[#222432] rounded-xl p-4 flex gap-3 relative group">
              <p className="text-sm text-slate-200 leading-relaxed flex-1 pr-6">
                "{query}"
              </p>
              <button suppressHydrationWarning className="absolute right-3 top-3 text-slate-500 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity">
                <Copy size={16} />
              </button>
            </div>
          </div>

          {/* Processing Steps */}
          <div className="mt-8 px-5">
            <h3 className="text-[10px] font-bold text-indigo-400 tracking-wider mb-4 uppercase">Processing Steps</h3>
            <div className="space-y-4 relative before:absolute before:inset-y-2 before:left-[7px] before:w-[2px] before:bg-[#222432]">
              {/* Step 1 */}
              <div className="flex items-center gap-4 relative">
                <div className="bg-[#111217] py-1 z-10">
                  {analysisStep > 0 || hasResult ? (
                    <CheckCircle2 size={16} className="text-emerald-500" />
                  ) : isAnalyzing && analysisStep === 0 ? (
                    <div className="w-4 h-4 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin" />
                  ) : (
                    <Circle size={16} className="text-[#3A3F58]" />
                  )}
                </div>
                <span className={`text-sm ${analysisStep > 0 || hasResult ? 'text-slate-300' : isAnalyzing && analysisStep === 0 ? 'text-white font-medium' : 'text-slate-600'}`}>
                  Understanding query
                </span>
              </div>
              
              {/* Step 2 */}
              <div className="flex items-center gap-4 relative">
                <div className="bg-[#111217] py-1 z-10">
                  {analysisStep > 1 || hasResult ? (
                    <CheckCircle2 size={16} className="text-emerald-500" />
                  ) : isAnalyzing && analysisStep === 1 ? (
                    <div className="w-4 h-4 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin" />
                  ) : (
                    <Circle size={16} className="text-[#3A3F58]" />
                  )}
                </div>
                <span className={`text-sm ${analysisStep > 1 || hasResult ? 'text-slate-300' : isAnalyzing && analysisStep === 1 ? 'text-white font-medium' : 'text-slate-600'}`}>
                  Validating inputs & selecting workflow
                </span>
              </div>

              {/* Step 3 */}
              <div className="flex items-center justify-between w-full relative">
                <div className="flex items-center gap-4">
                  <div className="bg-[#111217] py-1 z-10">
                    {analysisStep > 2 || hasResult ? (
                      <CheckCircle2 size={16} className="text-emerald-500" />
                    ) : isAnalyzing && analysisStep === 2 ? (
                      <div className="w-4 h-4 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin shadow-[0_0_10px_rgba(99,102,241,0.5)]" />
                    ) : (
                      <Circle size={16} className="text-[#3A3F58]" />
                    )}
                  </div>
                  <span className={`text-sm ${analysisStep > 2 || hasResult ? 'text-slate-300' : isAnalyzing && analysisStep === 2 ? 'text-white font-medium drop-shadow-[0_0_8px_rgba(255,255,255,0.3)]' : 'text-slate-600'}`}>
                    Running {queryType === 'cross_modal' ? 'cross-modal fusion' : queryType === 'change_detection' ? 'change analysis' : queryType === 'vqa' ? 'visual reasoning' : 'grounding analysis'}
                  </span>
                </div>
                {(analysisStep > 2 || hasResult || (isAnalyzing && analysisStep === 2)) && (
                  <span className="text-xs text-slate-500">1.3s</span>
                )}
              </div>

              {/* Step 4 */}
              <div className="flex items-center gap-4 relative">
                <div className="bg-[#111217] py-1 z-10">
                  {analysisStep > 3 || hasResult ? (
                    <CheckCircle2 size={16} className="text-emerald-500" />
                  ) : isAnalyzing && analysisStep === 3 ? (
                    <div className="w-4 h-4 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin" />
                  ) : (
                    <Circle size={16} className="text-[#3A3F58]" />
                  )}
                </div>
                <span className={`text-sm ${analysisStep > 3 || hasResult ? 'text-slate-300' : isAnalyzing && analysisStep === 3 ? 'text-white font-medium' : 'text-slate-600'}`}>
                  Generating evidence & preparing answer
                </span>
              </div>
            </div>
            
            {hasResult && (
              <div className="mt-6 border border-[#222432] bg-[#14151C] rounded-lg p-4">
                <div className="flex items-center justify-between mb-3 cursor-pointer">
                  <span className="text-xs font-semibold text-slate-400 tracking-wider">TECHNICAL DETAILS</span>
                  <ChevronDown size={14} className="text-slate-500" />
                </div>
                <div className="space-y-3 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-500">TASK</span>
                    <span className="text-slate-300 text-right">{queryType === 'cross_modal' ? 'Optical + SAR Fusion' : queryType === 'change_detection' ? 'Bi-temporal Change Detection' : queryType === 'vqa' ? 'VQA' : 'Grounding'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">MODELS</span>
                    <span className="text-slate-300 text-right">
                      {queryType === 'cross_modal' ? 'Fusion Model (BigEarthNet Adapted)' : queryType === 'change_detection' ? 'Change Model, Grounding Model' : 'VLM (BigEarthNet Adapted)'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">INPUT</span>
                    <span className="text-slate-300 text-right">
                      {queryType === 'cross_modal' ? 'Sentinel-2 (Optical) + Sentinel-1 (SAR)' : queryType === 'change_detection' ? 'Bi-temporal Optical Imagery' : 'Single Optical Imagery'}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <AnimatePresence>
            {hasResult && (
              <motion.div 
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
              >
                {/* RESULT SUMMARY CARD */}
                <div className="mt-10 px-5">
                  <h3 className="text-[10px] font-bold text-indigo-400 tracking-wider mb-3 uppercase">Result</h3>
                  
                  {queryType === 'vqa' ? (
                    <div className="bg-[#14151C] border border-[#222432] rounded-xl p-5 shadow-lg">
                      <p className="text-sm text-slate-200 leading-relaxed">
                        The scene contains primarily agricultural land, with patches of mixed forest and scattered built-up residential areas along the main access roads.
                      </p>
                      <div className="mt-4 flex items-center justify-between text-xs border-t border-[#222432] pt-4">
                         <span className="text-slate-400">Confidence: <span className="text-emerald-400">91%</span></span>
                         <span className="text-slate-500">Processing Time: 1.84 sec</span>
                      </div>
                    </div>
                  ) : queryType === 'cross_modal' ? (
                    <div className="bg-[#14151C] border border-[#222432] rounded-xl p-5 shadow-lg">
                      <p className="text-sm text-slate-200 leading-relaxed">
                        Successfully fused Sentinel-2 and Sentinel-1 data. SAR backscatter confirms dense built-up areas, while optical signatures clearly delineate surface water boundaries.
                      </p>
                      <div className="mt-4 flex items-center justify-between text-xs border-t border-[#222432] pt-4">
                         <span className="text-slate-400">Confidence: <span className="text-emerald-400">94%</span></span>
                         <span className="text-slate-500">Processing Time: 2.12 sec</span>
                      </div>
                    </div>
                  ) : (
                    <div className={`border rounded-xl p-5 shadow-lg ${queryType === 'change_detection' ? 'bg-[#171216] border-[#3A1E24]' : 'bg-[#12131C] border-[#2A2B44]'}`}>
                       <div className="flex items-center gap-5 mb-5">
                         <div className={`w-14 h-14 rounded-xl border flex items-center justify-center shadow-inner ${queryType === 'change_detection' ? 'bg-[#2A151A] border-[#4A2228] text-[#FF5A65]' : 'bg-[#161628] border-[#2D2E4A] text-[#818CF8]'}`}>
                           <Building2 size={28} />
                         </div>
                         <div>
                           <div className="text-4xl font-light text-white leading-none mb-1">
                             {queryType === 'change_detection' ? '7' : '15'}
                           </div>
                           <div className="text-sm text-slate-300">
                             {queryType === 'change_detection' ? 'New Buildings Detected' : 'Buildings Detected'}
                           </div>
                         </div>
                       </div>
                       
                       <div className="space-y-1.5 mb-4">
                         <div className="flex items-center justify-between text-xs">
                           <span className="text-slate-400">Confidence</span>
                           <span className="text-white font-medium">92%</span>
                         </div>
                         <div className="h-1.5 w-full bg-[#222432] rounded-full overflow-hidden">
                           <motion.div 
                             initial={{ width: 0 }} 
                             animate={{ width: '92%' }} 
                             transition={{ duration: 1, delay: 0.2 }}
                             className="h-full bg-emerald-500 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.5)]" 
                           />
                         </div>
                       </div>
                       
                       <div className={`flex items-center justify-between text-xs text-slate-500 pt-4 border-t ${queryType === 'change_detection' ? 'border-[#3A1E24]/50' : 'border-[#2A2B44]/50'}`}>
                         <span>Processing Time</span>
                         <span className="text-slate-300 font-medium">2.41 sec</span>
                       </div>
                    </div>
                  )}
                </div>

                {/* DETECTED OBJECTS LIST */}
                {(queryType === 'grounding' || queryType === 'change_detection') && (
                  <div className="mt-8 px-5">
                    <h3 className="text-[10px] font-bold text-indigo-400 tracking-wider mb-4 uppercase">Detected Objects</h3>
                    <div className="space-y-3">
                       <div className="flex items-center justify-between text-sm group cursor-pointer hover:bg-[#1A1C24] p-2 -mx-2 rounded-lg transition-colors">
                         <div className="flex items-center gap-3 text-slate-300">
                           <Building2 size={16} className="text-[#FF5A65]" /> New Buildings
                         </div>
                         <span className="text-[#FF5A65] font-semibold">7</span>
                       </div>
                       <div className="flex items-center justify-between text-sm group cursor-pointer hover:bg-[#1A1C24] p-2 -mx-2 rounded-lg transition-colors">
                         <div className="flex items-center gap-3 text-slate-400">
                           <Building2 size={16} className="text-slate-500" /> Buildings (Existing)
                         </div>
                         <span className="text-slate-400 font-semibold">23</span>
                       </div>
                       <div className="flex items-center justify-between text-sm group cursor-pointer hover:bg-[#1A1C24] p-2 -mx-2 rounded-lg transition-colors">
                         <div className="flex items-center gap-3 text-slate-300">
                           <Navigation size={16} className="text-yellow-500" /> Roads
                         </div>
                         <span className="text-yellow-500 font-semibold">12</span>
                       </div>
                       <div className="flex items-center justify-between text-sm group cursor-pointer hover:bg-[#1A1C24] p-2 -mx-2 rounded-lg transition-colors">
                         <div className="flex items-center gap-3 text-slate-300">
                           <Trees size={16} className="text-emerald-500" /> Vegetation
                         </div>
                         <span className="text-emerald-500 font-semibold">41</span>
                       </div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center opacity-60">
           <div className="w-16 h-16 rounded-full bg-[#1A1C24] border border-[#222432] flex items-center justify-center text-indigo-500/50 mb-4">
             <Sparkles size={24} />
           </div>
           <h3 className="text-sm font-medium text-slate-300 mb-2">Ready to Analyze</h3>
           <p className="text-xs text-slate-500 leading-relaxed">
             Enter a query or select a quick action to begin satellite intelligence analysis.
           </p>
        </div>
      )}
    </aside>
  );
}
