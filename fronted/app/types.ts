export type ViewMode = 'before' | 'after' | 'split' | 'slider';
export type QueryType = 'grounding' | 'change_detection' | 'vqa' | 'cross_modal';

/** One loaded chip, as listed by the controller's GET /images. */
export interface SceneInfo {
  image_id: string;
  site: string;
  purpose: string;
  acquired: string;
  modality: 'optical' | 'sar';
  source: string | null;
  cloudy: boolean;
  cloud_cover_pct: number | null;
  size_px: [number, number];
  bounds_epsg4326: number[];
  preview: string;
}

export interface PlanStep {
  id: string;
  tool: string;
  args: Record<string, unknown>;
}

export interface Plan {
  source: 'llm' | 'fallback';
  provider: string;
  reasoning: string;
  answer_from: string;
  steps: PlanStep[];
}

/** A geometry item projected into the pixel space of one chip, for drawing. */
export interface DisplayItem {
  id?: string;
  label: string;
  confidence?: number;
  pixels: number[][][]; // rings of [x, y]
}

export interface DisplayLayer {
  image_id: string;
  size_px: [number, number];
  items: DisplayItem[];
}

/** The controller's handle_query() result, plus the server's display layer. */
export interface QueryResult {
  query_id?: string | null;
  query: string;
  plan: Plan | null;
  status: 'ok' | 'partial';
  answer: string;
  confidence: number | null;
  confidence_note: string;
  results: Record<string, Record<string, unknown>>;
  trace: unknown[];
  elapsed_s: number;
  display: Record<string, DisplayLayer>;
  images: SceneInfo[];
}

export interface ApiError {
  status: 'error';
  error: { code: string; message: string };
}
