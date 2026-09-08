'use client';

import React, { useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChevronLeft, ChevronRight, CloudFog, ImageOff } from 'lucide-react';
import { DisplayLayer, QueryResult, QueryType, SceneInfo, ViewMode } from '../app/types';
import { formatDate } from '../lib/api';

interface ImageViewerProps {
  scenes: SceneInfo[];          // this site's chips, oldest first
  result: QueryResult | null;
  queryType: QueryType;
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  sliderPosition: number;
  setSliderPosition: (pos: number) => void;
  isAnalyzing: boolean;
  hasResult: boolean;
}

const STROKE: Record<string, string> = {
  change_detect: '#FF5A65',
  filter_by_region: '#FBBF24',
  ground: '#818CF8',
  cross_modal: '#34D399',
};

/** The step whose tool produced each display layer, for colouring. */
function toolOfStep(result: QueryResult | null, stepId: string): string {
  return result?.plan?.steps.find(s => s.id === stepId)?.tool || 'ground';
}

/** One chip with every result polygon that belongs to it, in its own pixel space. */
function Chip({ scene, result, dim }: { scene: SceneInfo; result: QueryResult | null; dim?: boolean }) {
  const [w, h] = scene.size_px;
  const layers: [string, DisplayLayer][] = result
    ? Object.entries(result.display || {}).filter(([, layer]) => layer.image_id === scene.image_id)
    : [];
  const answerStep = result?.plan?.answer_from;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" className="absolute inset-0 w-full h-full">
      <image href={scene.preview} width={w} height={h} style={{ filter: dim ? 'brightness(0.55) saturate(0.6)' : undefined }} />
      {layers.map(([stepId, layer]) => {
        const tool = toolOfStep(result, stepId);
        const stroke = STROKE[tool] || '#818CF8';
        const isAnswer = stepId === answerStep;
        return (
          <g key={stepId} opacity={isAnswer ? 1 : 0.55}>
            {layer.items.map((item, i) =>
              item.pixels.map((ring, j) => (
                <motion.polygon
                  key={`${item.id || i}-${j}`}
                  points={ring.map(p => p.join(',')).join(' ')}
                  fill={stroke}
                  fillOpacity={isAnswer ? 0.18 : 0.08}
                  stroke={stroke}
                  strokeWidth={Math.max(1.5, w / 400)}
                  vectorEffect="non-scaling-stroke"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: Math.min(i * 0.05, 1) }}
                  style={{ transformOrigin: 'center', transformBox: 'fill-box' }}
                >
                  <title>{`${item.label}${item.confidence != null ? ` ${Math.round(item.confidence * 100)}%` : ''}`}</title>
                </motion.polygon>
              ))
            )}
          </g>
        );
      })}
    </svg>
  );
}

// Sensor label for the viewer tag. A SAR scene not actually sourced from
// Sentinel-1 is a demo stand-in (no real geocoded SAR imagery is fetched
// today, see controller/fetch_chips.py) and must say so rather than being
// mislabelled as real satellite data.
function sensorLabel(scene: SceneInfo): string {
  if (scene.modality !== 'sar') return 'Sentinel-2';
  return scene.source === 'sentinel-1' ? 'Sentinel-1 SAR' : 'Synthetic SAR (demo)';
}

function Tag({ scene, side }: { scene: SceneInfo; side: 'left' | 'right' }) {
  return (
    <div className={`absolute top-4 ${side === 'left' ? 'left-16' : 'right-4'} bg-[#0B0C10]/80 backdrop-blur-md px-3 py-1.5 rounded-md text-xs font-medium text-white border border-[#222432] z-10 shadow-lg flex items-center gap-2`}>
      {formatDate(scene.acquired)}
      <span className="text-slate-400">| {sensorLabel(scene)}</span>
      {scene.cloudy && <CloudFog size={12} className="text-slate-400" />}
    </div>
  );
}

export default function ImageViewer({
  scenes,
  result,
  queryType,
  viewMode,
  setViewMode,
  sliderPosition,
  setSliderPosition,
  isAnalyzing,
  hasResult,
}: ImageViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const startSliding = (e: React.MouseEvent) => {
    e.preventDefault();
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

  // Which chips to show. The scene the answer was computed on is "after";
  // the earliest clear optical scene is "before" for a comparison.
  const optical = scenes.filter(s => s.modality === 'optical');
  const clear = optical.filter(s => !s.cloudy);
  const answerLayer = result?.plan ? result.display?.[result.plan.answer_from] : undefined;
  const answerImageId = answerLayer?.image_id
    || (result?.plan?.steps.map(s => s.args.image_id || s.args.image_id_t2 || s.args.optical_image_id).find(Boolean) as string | undefined);
  const after = scenes.find(s => s.image_id === answerImageId) || clear[clear.length - 1] || optical[optical.length - 1] || scenes[0];
  const before = clear.find(s => s.image_id !== after?.image_id) || optical.find(s => s.image_id !== after?.image_id);
  const compare = queryType === 'change_detection' && !!before && !!after;

  return (
    <div className="flex-1 bg-[#111217] rounded-xl border border-[#1E1F27] flex flex-col overflow-hidden relative shadow-lg">

      {/* Viewer Header */}
      <div className="h-14 border-b border-[#1E1F27] flex items-center justify-between px-5 shrink-0 bg-[#111217] z-20">
        <h2 className="text-xs font-bold text-white tracking-widest">IMAGE VIEWER</h2>

        <div className="flex items-center gap-6">
          {scenes.length > 0 && (
            <div className="text-xs text-slate-500">
              {scenes.length} scene{scenes.length !== 1 ? 's' : ''} loaded ·{' '}
              {scenes.map(s => formatDate(s.acquired)).join(' · ')}
            </div>
          )}
          {compare && (
            <div className="flex items-center bg-[#0B0C10] p-1 rounded-lg border border-[#1E1F27]">
              {(['before', 'after', 'split', 'slider'] as ViewMode[]).map(mode => (
                <button
                  key={mode}
                  suppressHydrationWarning
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-1 text-xs font-medium rounded transition-colors capitalize ${viewMode === mode ? 'bg-[#2D2B55] text-indigo-200 border border-indigo-500/30 shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                >
                  {mode}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Viewer Canvas */}
      <div className="flex-1 relative bg-[#1A1C23] overflow-hidden" ref={containerRef}>

        {!after && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-3">
            <ImageOff size={28} />
            <span className="text-sm">No scenes loaded. Is the controller running?</span>
          </div>
        )}

        {/* Base layer: the "after" chip with its polygons */}
        {after && (
          <div className={`absolute inset-0 ${compare && viewMode === 'before' ? 'hidden' : ''}`}>
            <Chip scene={after} result={result} />
            <Tag scene={after} side="right" />

            {/* Scanning Animation */}
            <AnimatePresence>
              {isAnalyzing && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 pointer-events-none z-10">
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
          </div>
        )}

        {/* "Before" chip on top, clipped for split / slider */}
        {compare && before && (
          <div
            className={`absolute inset-0 z-20 ${viewMode === 'after' ? 'hidden' : ''}`}
            style={{
              clipPath: viewMode === 'slider' ? `inset(0 ${100 - sliderPosition}% 0 0)` : viewMode === 'split' ? 'inset(0 50% 0 0)' : 'inset(0 0 0 0)',
            }}
          >
            <Chip scene={before} result={result} dim={viewMode !== 'before'} />
            <div className={`absolute inset-y-0 right-0 w-px bg-white/20 shadow-[2px_0_15px_rgba(0,0,0,0.8)] ${viewMode === 'slider' || viewMode === 'split' ? '' : 'hidden'}`} />
            <Tag scene={before} side="left" />
          </div>
        )}

        {/* Slider Handle */}
        {compare && viewMode === 'slider' && (
          <div className="absolute top-0 bottom-0 w-px bg-white/50 cursor-ew-resize z-30 group" style={{ left: `${sliderPosition}%` }} onMouseDown={startSliding}>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 bg-[#111217] rounded-full border border-[#2A2B35] flex items-center justify-center text-slate-300 shadow-[0_0_20px_rgba(0,0,0,0.5)] group-hover:scale-110 group-hover:border-indigo-500/50 transition-all">
              <ChevronLeft size={14} className="-mr-0.5" />
              <ChevronRight size={14} className="-ml-0.5" />
            </div>
          </div>
        )}

        {/* Legend */}
        {hasResult && result?.display && Object.keys(result.display).length > 0 && (
          <div className="absolute left-4 bottom-4 z-30 bg-[#0B0C10]/85 backdrop-blur-md rounded-lg border border-[#222432] px-3 py-2 text-[11px] text-slate-300 flex flex-col gap-1 shadow-lg">
            {Object.entries(result.display).map(([stepId, layer]) => {
              const tool = toolOfStep(result, stepId);
              return (
                <div key={stepId} className="flex items-center gap-2">
                  <span className="inline-block w-3 h-3 rounded-sm border-2" style={{ borderColor: STROKE[tool] || '#818CF8', background: `${STROKE[tool] || '#818CF8'}30` }} />
                  <span className="text-slate-400">{stepId}</span> {tool.replace('_', ' ')} · {layer.items.length}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
