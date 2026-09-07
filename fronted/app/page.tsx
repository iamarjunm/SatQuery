'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Sparkles } from 'lucide-react';
import { QueryResult, QueryType, SceneInfo, ViewMode } from './types';
import { fetchScenes, queryTypeOf, runQuery, scenesForSite } from '../lib/api';
import Sidebar from '../components/Sidebar';
import Header from '../components/Header';
import ImageViewer from '../components/ImageViewer';
import QueryInput from '../components/QueryInput';
import ResultsPanel from '../components/ResultsPanel';

const DEFAULT_SITE = 'jnpt_port';

export default function SatQueryApp() {
  // Data from the controller
  const [scenes, setScenes] = useState<SceneInfo[]>([]);
  const [site, setSite] = useState<string>(DEFAULT_SITE);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Query lifecycle
  const [query, setQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);

  // Viewer
  const [sliderPosition, setSliderPosition] = useState(50);
  const [viewMode, setViewMode] = useState<ViewMode>('slider');
  const [activeTab, setActiveTab] = useState('compare');
  const [toastMsg, setToastMsg] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 3000);
  };

  useEffect(() => {
    fetchScenes()
      .then(list => {
        setScenes(list);
        if (!list.some(s => s.site === DEFAULT_SITE) && list.length) setSite(list[0].site);
      })
      .catch(err => setLoadError(err.message));
  }, []);

  const sites = useMemo(() => Array.from(new Set(scenes.map(s => s.site))), [scenes]);
  const siteScenes = useMemo(() => scenesForSite(scenes, site), [scenes, site]);
  const hasResult = result !== null;
  const queryType: QueryType = queryTypeOf(result);

  const handleAnalyze = async (text: string) => {
    if (!text.trim() || isAnalyzing) return;
    setQuery(text);
    setResult(null);
    setQueryError(null);
    setIsAnalyzing(true);
    setAnalysisStep(0);
    // The real work is one request. The step indicator advances on a timer
    // while it is in flight and jumps to done when the answer lands.
    const timers = [setTimeout(() => setAnalysisStep(1), 500), setTimeout(() => setAnalysisStep(2), 1500)];
    try {
      const res = await runQuery(text, siteScenes.map(s => s.image_id));
      setResult(res);
      setAnalysisStep(4);
      if (res.status === 'partial') showToast('Partial answer: a step did not complete');
    } catch (err) {
      setQueryError(err instanceof Error ? err.message : String(err));
      setAnalysisStep(0);
    } finally {
      timers.forEach(clearTimeout);
      setIsAnalyzing(false);
    }
  };

  const handleNewQuery = () => {
    setQuery('');
    setResult(null);
    setQueryError(null);
    setIsAnalyzing(false);
    setAnalysisStep(0);
  };

  const handleSiteChange = (next: string) => {
    setSite(next);
    handleNewQuery();
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

        <Header sites={sites} site={site} onSiteChange={handleSiteChange} sceneCount={scenes.length} loadError={loadError} />

        {/* Center Workspace */}
        <div className="flex-1 flex flex-col p-5 gap-5 overflow-hidden">

          <ImageViewer
            scenes={siteScenes}
            result={result}
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
        result={result}
        error={queryError}
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
