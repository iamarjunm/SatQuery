import { ApiError, QueryResult, QueryType, SceneInfo } from '../app/types';

/** The Express backend, which proxies to the controller. */
export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';

export async function fetchScenes(): Promise<SceneInfo[]> {
  const res = await fetch(`${BACKEND_URL}/query/images`);
  const body = await res.json();
  if (!res.ok || body.status !== 'ok') {
    throw new Error((body as ApiError).error?.message || `backend returned ${res.status}`);
  }
  return body.images as SceneInfo[];
}

export async function runQuery(query: string, imageIds: string[]): Promise<QueryResult> {
  const res = await fetch(`${BACKEND_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query_id: `q_${Date.now()}`, query, image_ids: imageIds }),
  });
  const body = await res.json();
  if (!res.ok || body.status === 'error') {
    throw new Error((body as ApiError).error?.message || `backend returned ${res.status}`);
  }
  return body as QueryResult;
}

/** Which kind of evidence the plan produced, for the viewer and the panel. */
export function queryTypeOf(result: QueryResult | null): QueryType {
  const plan = result?.plan;
  if (!plan) return 'vqa';
  const tools = plan.steps.map(s => s.tool);
  if (tools.includes('cross_modal')) return 'cross_modal';
  if (tools.includes('change_detect')) return 'change_detection';
  if (tools.includes('ground')) return 'grounding';
  return 'vqa';
}

/** The chips a site has, oldest first. */
export function scenesForSite(scenes: SceneInfo[], site: string): SceneInfo[] {
  return scenes.filter(s => s.site === site).sort((a, b) => a.acquired.localeCompare(b.acquired));
}

export function siteLabel(site: string): string {
  const names: Record<string, string> = {
    vienna: 'Vienna, Austria',
    navi_mumbai: 'Navi Mumbai Airport',
    jewar: 'Jewar Airport, Noida',
    jnpt_port: 'JNPT Port, Mumbai',
    rondonia: 'Rondonia, Brazil',
    udaipur: 'Udaipur Lakes',
    bhadla: 'Bhadla Solar Park',
    singrauli: 'Singrauli Coal Mine',
    singapore: 'Singapore Anchorage',
    punjab: 'Punjab Cropland',
    sundarbans: 'Sundarbans',
  };
  return names[site] || site;
}

export function formatDate(iso: string): string {
  const d = new Date(iso + 'T00:00:00');
  return isNaN(d.getTime()) ? iso : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}
