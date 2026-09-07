'use client';

import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { CheckCircle2, Circle, AlertTriangle, Copy, Sparkles, Cpu, Boxes } from 'lucide-react';
import { QueryResult, QueryType } from '../app/types';

interface ResultsPanelProps {
  query: string;
  isAnalyzing: boolean;
  hasResult: boolean;
  analysisStep: number;
  queryType: QueryType;
  result: QueryResult | null;
  error: string | null;
}

const TOOL_LABEL: Record<string, string> = {
  vqa: 'Visual question answering',
  ground: 'Grounding',
  change_detect: 'Change detection',
  cross_modal: 'Optical + SAR fusion',
  filter_by_region: 'Geometric filter (objects inside regions)',
  count: 'Count',
};

function argSummary(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join(', ');
}

function StepRow({ done, active, label, detail, extra }: { done: boolean; active: boolean; label: string; detail?: string; extra?: string }) {
  return (
    <div className="flex items-start justify-between w-full relative gap-3">
      <div className="flex items-start gap-4 min-w-0">
        <div className="bg-[#111217] py-1 z-10 shrink-0">
          {done ? (
            <CheckCircle2 size={16} className="text-emerald-500" />
          ) : active ? (
            <div className="w-4 h-4 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin shadow-[0_0_10px_rgba(99,102,241,0.5)]" />
          ) : (
            <Circle size={16} className="text-[#3A3F58]" />
          )}
        </div>
        <div className="min-w-0">
          <span className={`text-sm ${done ? 'text-slate-300' : active ? 'text-white font-medium' : 'text-slate-600'}`}>{label}</span>
          {detail && <div className="text-[11px] text-slate-500 truncate font-mono" title={detail}>{detail}</div>}
        </div>
      </div>
      {extra && <span className="text-xs text-slate-500 shrink-0">{extra}</span>}
    </div>
  );
}

export default function ResultsPanel({ query, isAnalyzing, hasResult, analysisStep, queryType, result, error }: ResultsPanelProps) {
  const plan = result?.plan || null;
  const confidencePct = result?.confidence != null ? Math.round(result.confidence * 100) : null;

  // Detected objects, grouped by label across every geometry-carrying step.
  const answerLayer = plan ? result?.display?.[plan.answer_from] : undefined;
  const counts: Record<string, number> = {};
  for (const item of answerLayer?.items || []) counts[item.label] = (counts[item.label] || 0) + 1;

  return (
    <aside className="w-[340px] bg-[#111217] border-l border-[#1E1F27] flex flex-col shrink-0 overflow-y-auto z-20 shadow-[-10px_0_30px_rgba(0,0,0,0.5)]">
      <div className="p-5 border-b border-[#1E1F27] sticky top-0 bg-[#111217] z-10">
        <h2 className="text-xs font-bold text-white tracking-widest">AI ANALYSIS</h2>
      </div>

      {query || hasResult || isAnalyzing || error ? (
        <div className="pb-10">
          {/* Query Display */}
          <div className="mt-6 px-5">
            <h3 className="text-[10px] font-bold text-indigo-400 tracking-wider mb-3">QUERY</h3>
            <div className="bg-[#1A1C24] border border-[#222432] rounded-xl p-4 flex gap-3 relative group">
              <p className="text-sm text-slate-200 leading-relaxed flex-1 pr-6">"{query}"</p>
              <button suppressHydrationWarning onClick={() => navigator.clipboard?.writeText(query)} className="absolute right-3 top-3 text-slate-500 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity">
                <Copy size={16} />
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-6 mx-5 border border-[#3A1E24] bg-[#171216] rounded-xl p-4 flex gap-3 text-sm text-[#FF9AA2]">
              <AlertTriangle size={18} className="shrink-0 text-[#FF5A65]" />
              <span>{error}</span>
            </div>
          )}

          {/* Processing Steps: the planner, then the plan's actual steps */}
          <div className="mt-8 px-5">
            <h3 className="text-[10px] font-bold text-indigo-400 tracking-wider mb-4 uppercase">Processing Steps</h3>
            <div className="space-y-4 relative before:absolute before:inset-y-2 before:left-[7px] before:w-[2px] before:bg-[#222432]">
              <StepRow
                done={analysisStep > 0 || hasResult}
                active={isAnalyzing && analysisStep === 0}
                label="Understanding query"
              />
              <StepRow
                done={analysisStep > 1 || hasResult}
                active={isAnalyzing && analysisStep === 1}
                label="Planning workflow with the LLM"
                detail={plan ? `${plan.source === 'llm' ? 'planned by' : 'keyword fallback,'} ${plan.provider}` : undefined}
              />
              {plan ? (
                plan.steps.map(step => (
                  <StepRow
                    key={step.id}
                    done={hasResult}
                    active={false}
                    label={`${step.id}: ${TOOL_LABEL[step.tool] || step.tool}`}
                    detail={argSummary(step.args)}
                    extra={step.id === plan.answer_from ? 'answer' : undefined}
                  />
                ))
              ) : (
                <StepRow
                  done={analysisStep > 2 || hasResult}
                  active={isAnalyzing && analysisStep === 2}
                  label="Running the plan against the agents"
                />
              )}
              <StepRow
                done={hasResult}
                active={isAnalyzing && analysisStep >= 2}
                label="Validating geometry & preparing answer"
                extra={result ? `${result.elapsed_s.toFixed(1)}s` : undefined}
              />
            </div>

            {hasResult && plan && (
              <div className="mt-6 border border-[#222432] bg-[#14151C] rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-semibold text-slate-400 tracking-wider">TECHNICAL DETAILS</span>
                  <Cpu size={14} className="text-slate-500" />
                </div>
                <div className="space-y-3 text-xs">
                  <div className="flex justify-between gap-4">
                    <span className="text-slate-500">TASK</span>
                    <span className="text-slate-300 text-right">
                      {queryType === 'cross_modal' ? 'Optical + SAR fusion' : queryType === 'change_detection' ? 'Bi-temporal change detection' : queryType === 'vqa' ? 'VQA' : 'Grounding'}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-slate-500">PLANNER</span>
                    <span className="text-slate-300 text-right">{plan.provider}{plan.source === 'fallback' ? ' (degraded)' : ''}</span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-slate-500">STATUS</span>
                    <span className={`text-right ${result?.status === 'ok' ? 'text-emerald-400' : 'text-amber-400'}`}>{result?.status}</span>
                  </div>
                  <div>
                    <span className="text-slate-500">REASONING</span>
                    <p className="text-slate-300 mt-1 leading-relaxed">{plan.reasoning}</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          <AnimatePresence>
            {hasResult && result && (
              <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: 'easeOut' }}>
                {/* RESULT SUMMARY CARD */}
                <div className="mt-10 px-5">
                  <h3 className="text-[10px] font-bold text-indigo-400 tracking-wider mb-3 uppercase">Result</h3>
                  <div className={`border rounded-xl p-5 shadow-lg ${queryType === 'change_detection' ? 'bg-[#171216] border-[#3A1E24]' : 'bg-[#12131C] border-[#2A2B44]'}`}>
                    {answerLayer && answerLayer.items.length > 0 && (
                      <div className="flex items-center gap-5 mb-5">
                        <div className={`w-14 h-14 rounded-xl border flex items-center justify-center shadow-inner ${queryType === 'change_detection' ? 'bg-[#2A151A] border-[#4A2228] text-[#FF5A65]' : 'bg-[#161628] border-[#2D2E4A] text-[#818CF8]'}`}>
                          <Boxes size={28} />
                        </div>
                        <div>
                          <div className="text-4xl font-light text-white leading-none mb-1">{answerLayer.items.length}</div>
                          <div className="text-sm text-slate-300">{queryType === 'change_detection' ? 'changed regions' : 'objects located'}</div>
                        </div>
                      </div>
                    )}
                    <p className="text-sm text-slate-200 leading-relaxed">{result.answer}</p>
                    {confidencePct != null && (
                      <div className="space-y-1.5 mt-4">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-400">Confidence</span>
                          <span className="text-white font-medium">{confidencePct}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-[#222432] rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${confidencePct}%` }}
                            transition={{ duration: 1, delay: 0.2 }}
                            className={`h-full rounded-full ${confidencePct >= 70 ? 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]' : confidencePct >= 50 ? 'bg-amber-400' : 'bg-[#FF5A65]'}`}
                          />
                        </div>
                        {result.confidence_note && <div className="text-[11px] text-slate-500">{result.confidence_note}</div>}
                      </div>
                    )}
                    <div className={`flex items-center justify-between text-xs text-slate-500 pt-4 mt-4 border-t ${queryType === 'change_detection' ? 'border-[#3A1E24]/50' : 'border-[#2A2B44]/50'}`}>
                      <span>Processing time</span>
                      <span className="text-slate-300 font-medium">{result.elapsed_s.toFixed(2)} s</span>
                    </div>
                  </div>
                </div>

                {/* DETECTED OBJECTS LIST */}
                {Object.keys(counts).length > 0 && (
                  <div className="mt-8 px-5">
                    <h3 className="text-[10px] font-bold text-indigo-400 tracking-wider mb-4 uppercase">Detected</h3>
                    <div className="space-y-3">
                      {Object.entries(counts).map(([label, n]) => (
                        <div key={label} className="flex items-center justify-between text-sm p-2 -mx-2 rounded-lg hover:bg-[#1A1C24] transition-colors">
                          <div className="flex items-center gap-3 text-slate-300 capitalize">
                            <Boxes size={16} className={queryType === 'change_detection' ? 'text-[#FF5A65]' : 'text-[#818CF8]'} /> {label.replace('_', ' ')}
                          </div>
                          <span className={`font-semibold ${queryType === 'change_detection' ? 'text-[#FF5A65]' : 'text-[#818CF8]'}`}>{n}</span>
                        </div>
                      ))}
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
          <p className="text-xs text-slate-500 leading-relaxed">Pick a site, then ask a question about its imagery.</p>
        </div>
      )}
    </aside>
  );
}
