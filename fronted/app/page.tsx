'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Sparkles } from 'lucide-react';
import { QueryType, ViewMode } from './types';
import Sidebar from '../components/Sidebar';
import Header from '../components/Header';
import ImageViewer from '../components/ImageViewer';
import QueryInput from '../components/QueryInput';
import ResultsPanel from '../components/ResultsPanel';

export default function SatQueryApp() {
  // Global App State
  const [query, setQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasResult, setHasResult] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [viewMode, setViewMode] = useState<ViewMode>('slider');
  const [activeTab, setActiveTab] = useState('compare');
  const [queryType, setQueryType] = useState<QueryType>('vqa');
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 3000);
  };

  // Mock Analysis Sequence
  const handleAnalyze = (text: string) => {
    if (!text.trim()) return;
    
    // Determine query type
    const lowerText = text.toLowerCase();
    let qType: QueryType = 'grounding';
    if (lowerText.includes('change') || lowerText.includes('between') || lowerText.includes('dates')) {
      qType = 'change_detection';
    } else if (lowerText.includes('describe') || lowerText.includes('what type') || lowerText.includes('land-cover')) {
      qType = 'vqa';
    } else if (lowerText.includes('use the optical and sar images together') || lowerText.includes('identify built-up and water')) {
      qType = 'cross_modal';
    } else {
      qType = 'grounding';
    }
    
    setQueryType(qType);
    setQuery(text);
    setIsAnalyzing(true);
    setHasResult(false);
    setAnalysisStep(0);

    // Simulate steps
    const steps = [
      setTimeout(() => setAnalysisStep(1), 800),
      setTimeout(() => setAnalysisStep(2), 1600),
      setTimeout(() => setAnalysisStep(3), 2600),
      setTimeout(() => {
        setIsAnalyzing(false);
        setHasResult(true);
        setAnalysisStep(4);
      }, 3500)
    ];

    return () => steps.forEach(clearTimeout);
  };

  const handleNewQuery = () => {
    setQuery('');
    setHasResult(false);
    setIsAnalyzing(false);
    setAnalysisStep(0);
  };

  return (
    <div className="h-screen w-full bg-[#0B0C10] text-slate-300 font-sans flex overflow-hidden selection:bg-indigo-500/30">
      
      <Sidebar 
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        showToast={showToast}
        onNewQuery={handleNewQuery}
      />

      {/* MAIN CONTENT CENTER */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#0B0C10] z-10 relative shadow-[-10px_0_30px_rgba(0,0,0,0.5)]">
        
        <Header showToast={showToast} />

        {/* Center Workspace */}
        <div className="flex-1 flex flex-col p-5 gap-5 overflow-hidden">
          
          <ImageViewer 
            queryType={queryType}
            viewMode={viewMode}
            setViewMode={setViewMode}
            sliderPosition={sliderPosition}
            setSliderPosition={setSliderPosition}
            isAnalyzing={isAnalyzing}
            hasResult={hasResult}
          />

          <QueryInput 
            query={query}
            setQuery={setQuery}
            isAnalyzing={isAnalyzing}
            handleAnalyze={handleAnalyze}
          />

        </div>
      </main>

      <ResultsPanel 
        query={query}
        isAnalyzing={isAnalyzing}
        hasResult={hasResult}
        analysisStep={analysisStep}
        queryType={queryType}
      />

      {/* Global Toast */}
      <AnimatePresence>
        {toastMsg && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.9 }}
            className="fixed bottom-10 left-1/2 -translate-x-1/2 bg-[#1A1525] border border-indigo-500/30 text-indigo-200 px-6 py-3 rounded-full shadow-[0_0_30px_rgba(99,102,241,0.2)] flex items-center gap-3 z-50 backdrop-blur-md"
          >
            <Sparkles size={16} className="text-indigo-400" />
            <span className="text-sm font-medium">{toastMsg}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
