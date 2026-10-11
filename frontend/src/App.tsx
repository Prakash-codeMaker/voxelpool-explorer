import { useEffect, useMemo, useState } from 'react'
import {
  Activity, ArrowUpRight, Beaker, Check, ChevronDown,
  CircleHelp, Clock3, Cpu, Download, ExternalLink, FlaskConical, Grid3X3,
  Layers3, Play, RefreshCw, ShieldCheck, SlidersHorizontal, Sparkles,
  Terminal, Waypoints, XCircle, GitCompare, Telescope, FileSpreadsheet, BrainCircuit,
} from 'lucide-react'

const API = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

type Pattern = 'uniform' | 'hotspot' | 'unique'
type Timing = { samples_ms: number[]; median_ms: number; min_ms: number; max_ms: number; p95_ms: number; standard_deviation_ms?: number; coefficient_of_variation_percent?: number }
type Strategy = { id: string; label: string; correct: boolean; max_abs_error: number | null; timing: Timing | null; status: string; speedup_vs_reference: number | null }
type Experiment = {
  experiment_id: string; created_at: string; schema_version: string; measurement_backend: string;
  measurement_unit: string; warning: string;
  environment?: { python_version: string; numpy_version: string; os: string; machine: string; processor: string };
  workload: { points: number; channels: number; grid_size: number; pattern: Pattern; seed: number; num_voxels: number };
  occupancy: { num_voxels: number; occupied_voxels: number; empty_voxels: number; occupancy_percent: number; collision_points: number; collision_rate_percent: number; max_points_per_voxel: number; mean_points_per_occupied_voxel: number; xy_projection: number[][] };
  correctness: { reference: string; rtol: number; atol: number; all_candidates_passed: boolean };
  strategies: Strategy[];
  methodology: { warmup_runs_per_strategy: number; measured_repeats: number; summary_statistic: string; timing_scope: string; reference_output_dtype: string };
}
type Config = { points: number; channels: number; grid_size: number; pattern: Pattern; repeats: number; seed: number }
type Tab = 'explore' | 'results' | 'sweep' | 'landscape' | 'methodology'
type Preview = { workload: Experiment['workload']; occupancy: Experiment['occupancy']; preview_only: boolean; warning: string }
type PreviewStatus = 'waiting' | 'loading' | 'ready' | 'offline' | 'invalid' | 'error'

type SweepConfig = { points: number[]; channels: number[]; grid_sizes: number[]; patterns: Pattern[]; seeds: number[]; repeats: number }
type SweepCase = { case_index: number; config: { points: number; channels: number; grid_size: number; pattern: Pattern; seed: number; repeats: number }; status: 'completed' | 'skipped' | 'error' | 'running'; reason: string | null; result: Experiment | null }
type SweepJob = { sweep_id: string; status: 'queued' | 'running' | 'completed' | 'failed'; created_at?: string; finished_at?: string | null; request: SweepConfig; total_cases: number; progress: number; completed_cases: number; skipped_cases: number; error_cases: number; cases?: SweepCase[]; error?: string }
type SelectorEvaluation = { method: string; status: string; split_policy: string; training_configurations: number; heldout_configurations: number; input_configurations: number; selector_accuracy_percent: number; selector_mean_regret_percent: number; selector_median_regret_percent: number; baseline_strategy: string; baseline_label: string; baseline_accuracy_percent: number; baseline_mean_regret_percent: number; evaluation_cases: { workload: { pattern: Pattern; points: number; channels: number; grid_size: number }; selected_strategy: string; selected_label: string; oracle_fastest_strategy: string; oracle_fastest_label: string; selected_median_ms: number; oracle_median_ms: number; regret_percent: number; baseline_strategy: string; baseline_label: string; baseline_regret_percent: number; nearest_neighbors: number }[]; interpretation: string }

const initialConfig: Config = { points: 1200, channels: 32, grid_size: 32, pattern: 'uniform', repeats: 5, seed: 42 }
const initialSweepConfig: SweepConfig = { points: [256, 1200], channels: [8, 32], grid_sizes: [16, 32], patterns: ['uniform', 'hotspot', 'unique'], seeds: [42, 43], repeats: 3 }
const STORAGE_PREFIX = 'voxelpool-explorer:'

function readStored<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(`${STORAGE_PREFIX}${key}`)
    return raw ? JSON.parse(raw) as T : fallback
  } catch { return fallback }
}
function writeStored(key: string, value: unknown) {
  try { window.localStorage.setItem(`${STORAGE_PREFIX}${key}`, JSON.stringify(value)) } catch { /* Storage may be disabled or full. */ }
}

function formatMs(v?: number | null) {
  if (v == null || !Number.isFinite(v)) return '—'
  if (v < 1) return `${v.toFixed(3)} ms`
  return `${v.toFixed(2)} ms`
}
function fmt(n: number) { return n.toLocaleString('en-US') }
function downloadText(name: string, content: string, type: string) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = name; a.style.display = 'none'
  document.body.appendChild(a); a.click(); a.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
function csvCell(value: string | number | boolean | null | undefined): string {
  if (value == null) return ''
  const text = String(value)
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}
function buildHistoryCSV(experiments: Experiment[]): string {
  const columns = [
    'experiment_id', 'created_at', 'backend', 'points', 'channels', 'grid_size', 'pattern', 'seed',
    'repeats', 'occupied_voxels', 'occupancy_percent', 'collision_rate_percent', 'strategy', 'correct',
    'median_ms', 'min_ms', 'max_ms', 'p95_ms', 'standard_deviation_ms', 'coefficient_of_variation_percent', 'speedup_vs_reference', 'max_abs_error', 'status',
    'python_version', 'numpy_version', 'os', 'machine',
  ]
  const rows: (string | number | boolean | null | undefined)[][] = []
  for (const e of experiments) {
    for (const strategy of e.strategies) {
      rows.push([
        e.experiment_id, e.created_at, e.measurement_backend, e.workload.points, e.workload.channels,
        e.workload.grid_size, e.workload.pattern, e.workload.seed, e.methodology.measured_repeats,
        e.occupancy.occupied_voxels, e.occupancy.occupancy_percent, e.occupancy.collision_rate_percent,
        strategy.id, strategy.correct, strategy.timing?.median_ms, strategy.timing?.min_ms,
        strategy.timing?.max_ms, strategy.timing?.p95_ms, strategy.timing?.standard_deviation_ms, strategy.timing?.coefficient_of_variation_percent, strategy.speedup_vs_reference,
        strategy.max_abs_error, strategy.status, e.environment?.python_version, e.environment?.numpy_version,
        e.environment?.os, e.environment?.machine,
      ])
    }
  }
  return [columns, ...rows].map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n'
}
function downloadJSON(data: unknown, filename = 'voxelpool-experiment.json') {
  downloadText(filename, JSON.stringify(data, null, 2), 'application/json')
}
function chooseWinner(result: Experiment | null) {
  if (!result) return null
  return [...result.strategies].filter(s => s.correct && s.timing && s.timing.median_ms > 0)
    .sort((a, b) => (a.timing?.median_ms ?? Infinity) - (b.timing?.median_ms ?? Infinity))[0] ?? null
}
function sweepCaseCsv(sweep: SweepJob): string {
  const columns = ['sweep_id', 'case_index', 'case_status', 'reason', 'points', 'channels', 'grid_size', 'pattern', 'seed', 'repeats', 'occupied_voxels', 'occupancy_percent', 'collision_rate_percent', 'strategy', 'correct', 'median_ms', 'min_ms', 'max_ms', 'p95_ms', 'standard_deviation_ms', 'coefficient_of_variation_percent', 'speedup_vs_reference', 'max_abs_error']
  const rows: (string | number | boolean | null | undefined)[][] = []
  for (const c of sweep.cases ?? []) {
    const e = c.result
    if (!e) {
      rows.push([sweep.sweep_id, c.case_index, c.status, c.reason, c.config.points, c.config.channels, c.config.grid_size, c.config.pattern, c.config.seed, c.config.repeats])
      continue
    }
    for (const strategy of e.strategies) rows.push([
      sweep.sweep_id, c.case_index, c.status, c.reason, e.workload.points, e.workload.channels, e.workload.grid_size,
      e.workload.pattern, e.workload.seed, e.methodology.measured_repeats, e.occupancy.occupied_voxels,
      e.occupancy.occupancy_percent, e.occupancy.collision_rate_percent, strategy.id, strategy.correct,
      strategy.timing?.median_ms, strategy.timing?.min_ms, strategy.timing?.max_ms, strategy.timing?.p95_ms, strategy.timing?.standard_deviation_ms, strategy.timing?.coefficient_of_variation_percent,
      strategy.speedup_vs_reference, strategy.max_abs_error,
    ])
  }
  return [columns, ...rows].map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n'
}
function ChoiceChips<T extends string | number>({ label, options, selected, onToggle }: {
  label: string; options: { value: T; label: string }[]; selected: T[]; onToggle: (value: T) => void
}) {
  return <div className="sweep-choice-group"><div className="sweep-choice-label">{label}</div><div className="choice-chips">
    {options.map(option => <button type="button" key={String(option.value)} className={`choice-chip ${selected.includes(option.value) ? 'selected' : ''}`} onClick={() => onToggle(option.value)} aria-pressed={selected.includes(option.value)}>{selected.includes(option.value) && <Check size={12} />}{option.label}</button>)}
  </div></div>
}

function VoxelMap({ projection, gridSize }: { projection: number[][]; gridSize: number }) {
  const max = Math.max(1, ...projection.flat())
  const cell = Math.max(5, Math.min(15, 300 / gridSize))
  const gap = gridSize > 32 ? 1 : 2
  const size = gridSize * (cell + gap)
  return <div className="voxel-map-wrap">
    <div className="voxel-map" style={{ width: size, gridTemplateColumns: `repeat(${gridSize}, ${cell}px)` }}>
      {projection.flatMap((row, y) => row.map((v, x) => {
        const intensity = v / max
        return <div key={`${x}-${y}`} title={`x=${x}, y=${y}: ${v} points`} className="voxel-cell"
          style={{ width: cell, height: cell, background: v === 0 ? '#171d2a' : `rgba(76, 221, 177, ${0.16 + intensity * 0.84})`, outline: v === max && v > 0 ? '1px solid #c0ffe9' : 'none' }} />
      }))}
    </div>
    <div className="map-legend"><span>Low occupancy</span><span className="legend-gradient" /><span>High occupancy</span></div>
    <p className="micro muted">XY projection · counts summed across Z · {gridSize} × {gridSize}</p>
  </div>
}

function TimingChart({ strategies }: { strategies: Strategy[] }) {
  const values = strategies.map(s => s.timing?.median_ms ?? 0)
  const max = Math.max(...values, 0.001)
  return <div className="timing-chart">
    {strategies.map((s, i) => {
      const val = s.timing?.median_ms ?? 0
      const width = Math.max(2, val / max * 100)
      return <div className="bar-row" key={s.id}>
        <div className="bar-label">{s.label}</div>
        <div className="bar-track"><div className={`bar-fill bar-${i}`} style={{ width: `${width}%` }} /></div>
        <div className="bar-value">{formatMs(val)}</div>
      </div>
    })}
  </div>
}

function App() {
  const [config, setConfig] = useState<Config>(() => readStored('config', initialConfig))
  const [result, setResult] = useState<Experiment | null>(() => readStored<Experiment | null>('current', null))
  const [history, setHistory] = useState<Experiment[]>(() => readStored<Experiment[]>('history', []))
  const [preview, setPreview] = useState<Preview | null>(null)
  const [previewStatus, setPreviewStatus] = useState<PreviewStatus>('waiting')
  const [tab, setTab] = useState<Tab>('explore')
  const [busy, setBusy] = useState(false)
  const [apiState, setApiState] = useState<'checking' | 'online' | 'offline'>('checking')
  const [error, setError] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [sweepConfig, setSweepConfig] = useState<SweepConfig>(() => readStored('sweep-config', initialSweepConfig))
  const [sweepResults, setSweepResults] = useState<SweepJob | null>(() => readStored<SweepJob | null>('sweep-results', null))
  const [sweepProgress, setSweepProgress] = useState<SweepJob | null>(null)
  const [sweepBusy, setSweepBusy] = useState(false)
  const [selectorBusy, setSelectorBusy] = useState(false)
  const [selectorResult, setSelectorResult] = useState<SelectorEvaluation | null>(() => readStored<SelectorEvaluation | null>('selector-result', null))
  const [sweepError, setSweepError] = useState('')

  const update = <K extends keyof Config>(key: K, value: Config[K]) => setConfig(prev => ({ ...prev, [key]: value }))

  async function checkHealth() {
    try {
      const res = await fetch(`${API}/api/health`)
      if (!res.ok) throw new Error('API unavailable')
      setApiState('online')
    } catch { setApiState('offline') }
  }

  useEffect(() => { checkHealth() }, [])

  useEffect(() => { writeStored('config', config) }, [config])
  useEffect(() => { writeStored('history', history) }, [history])
  useEffect(() => { writeStored('current', result) }, [result])
  useEffect(() => { writeStored('sweep-config', sweepConfig) }, [sweepConfig])
  useEffect(() => { writeStored('sweep-results', sweepResults) }, [sweepResults])
  useEffect(() => { writeStored('selector-result', selectorResult) }, [selectorResult])

  // Preview uses the same seeded Python workload generator as the benchmark,
  // but does not time any strategy. Debouncing avoids sending a request for
  // every tiny range-slider movement.
  useEffect(() => {
    if (apiState === 'offline') { setPreviewStatus('offline'); return }
    if (apiState !== 'online') { setPreviewStatus('waiting'); return }
    if (config.pattern === 'unique' && config.points > config.grid_size ** 3) {
      setPreview(null); setPreviewStatus('invalid'); return
    }
    const controller = new AbortController()
    setPreviewStatus('loading')
    const timeout = window.setTimeout(async () => {
      try {
        const res = await fetch(`${API}/api/preview`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...config, repeats: 1 }), signal: controller.signal,
        })
        const data = await res.json()
        if (!res.ok) throw new Error(data.detail || `Preview failed (${res.status})`)
        setPreview(data as Preview); setPreviewStatus('ready')
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') return
        setPreviewStatus('error')
      }
    }, 180)
    return () => { window.clearTimeout(timeout); controller.abort() }
  }, [apiState, config.points, config.channels, config.grid_size, config.pattern, config.seed])

  const canRun = config.pattern !== 'unique' || config.points <= config.grid_size ** 3
  const current = result
  const previewMatchesConfig = Boolean(preview && preview.workload.points === config.points &&
    preview.workload.channels === config.channels && preview.workload.grid_size === config.grid_size &&
    preview.workload.pattern === config.pattern && preview.workload.seed === config.seed)
  const best = useMemo(() => {
    if (!current) return null
    return [...current.strategies].filter(s => s.correct && s.timing).sort((a, b) => (a.timing?.median_ms ?? Infinity) - (b.timing?.median_ms ?? Infinity))[0] ?? null
  }, [current])

  async function runExperiment() {
    setBusy(true); setError('')
    try {
      const res = await fetch(`${API}/api/experiment`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`)
      setResult(data)
      setHistory(prev => [data, ...prev.filter(x => x.experiment_id !== data.experiment_id)].slice(0, 25))
      setTab('results')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unable to run experiment')
    } finally { setBusy(false); checkHealth() }
  }

  const sweepCaseCount = sweepConfig.points.length * sweepConfig.channels.length * sweepConfig.grid_sizes.length * sweepConfig.patterns.length * sweepConfig.seeds.length

  function toggleSweepDimension(key: 'points' | 'channels' | 'grid_sizes' | 'patterns' | 'seeds', value: number | Pattern) {
    setSweepConfig(prev => {
      const values = prev[key] as (number | Pattern)[]
      if (values.includes(value)) {
        if (values.length === 1) return prev
        return { ...prev, [key]: values.filter(item => item !== value) }
      }
      return { ...prev, [key]: [...values, value].sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true })) }
    })
  }

  async function runSweep() {
    if (sweepCaseCount > 48 || sweepBusy || apiState !== 'online') return
    setSweepBusy(true); setSweepError(''); setSelectorResult(null)
    setSweepProgress(null)
    try {
      const response = await fetch(`${API}/api/sweeps`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(sweepConfig),
      })
      const started = await response.json()
      if (!response.ok) throw new Error(started.detail || `Sweep request failed (${response.status})`)
      let done: SweepJob | null = null
      for (let attempt = 0; attempt < 900; attempt++) {
        const poll = await fetch(`${API}/api/sweeps/${started.sweep_id}`)
        const payload = await poll.json()
        if (!poll.ok) throw new Error(payload.detail || `Unable to read sweep status (${poll.status})`)
        setSweepProgress(payload as SweepJob)
        if (payload.status === 'completed' || payload.status === 'failed') {
          done = payload as SweepJob
          break
        }
        await new Promise(resolve => window.setTimeout(resolve, 650))
      }
      if (!done) throw new Error('Sweep is still running on the backend. Its progress can be checked again after refreshing the API status.')
      if (done.status === 'failed') throw new Error(done.error || 'Sweep failed. Check the backend terminal for details.')
      setSweepResults(done)
      setSelectorResult(null)
      setTab('sweep')
    } catch (e) {
      setSweepError(e instanceof Error ? e.message : 'Unable to run the workload sweep')
    } finally { setSweepBusy(false); checkHealth() }
  }

  async function evaluateSelector() {
    if (!sweepResults?.cases?.length || selectorBusy) return
    setSelectorBusy(true); setSweepError('')
    try {
      const response = await fetch(`${API}/api/selector/evaluate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cases: sweepResults.cases }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || `Selector evaluation failed (${response.status})`)
      setSelectorResult(payload as SelectorEvaluation)
      setTab('landscape')
    } catch (e) {
      setSweepError(e instanceof Error ? e.message : 'Unable to evaluate the strategy selector')
    } finally { setSelectorBusy(false) }
  }

  function exportSweepJSON() {
    if (sweepResults) downloadJSON({ project: 'VoxelPool Explorer', schema_version: 'voxelpool-sweep/v1', sweep: sweepResults, selector_evaluation: selectorResult }, 'voxelpool-sweep-research-bundle.json')
  }
  function exportResearchMarkdown() {
    if (!sweepResults?.cases) return
    const completed = sweepResults.cases.filter(c => c.status === 'completed' && c.result)
    const winners = new Map<string, number>()
    for (const item of completed) {
      const winner = chooseWinner(item.result)
      if (winner) winners.set(winner.label, (winners.get(winner.label) ?? 0) + 1)
    }
    const environment = completed[0]?.result?.environment
    const lines: string[] = [
      '# VoxelPool Explorer — Workload Sweep Report',
      '',
      `Generated: ${new Date().toISOString()}`,
      `Sweep ID: \`${sweepResults.sweep_id}\``,
      '',
      '> CPU-only exploratory measurements. This report does not contain Quadric, CUDA, GPU, or accelerator performance claims.',
      '',
      '## Research question',
      '',
      'Can workload features help predict which of the currently implemented voxel-pooling strategies performs best on held-out workload configurations?',
      '',
      '## Study design',
      '',
      `- Point counts: ${sweepResults.request.points.join(', ')}`,
      `- Feature channels: ${sweepResults.request.channels.join(', ')}`,
      `- Grid sizes: ${sweepResults.request.grid_sizes.map(v => `${v}³`).join(', ')}`,
      `- Distributions: ${sweepResults.request.patterns.join(', ')}`,
      `- Seeds: ${sweepResults.request.seeds.join(', ')}`,
      `- Measured repeats per strategy: ${sweepResults.request.repeats} (plus one warm-up per strategy)`,
      `- Requested / completed / skipped / error cases: ${sweepResults.total_cases} / ${sweepResults.completed_cases} / ${sweepResults.skipped_cases} / ${sweepResults.error_cases}`,
      '',
      '## Environment',
      '',
      `- Backend: ${completed[0]?.result?.measurement_backend ?? 'No completed measurements'}`,
      `- Python: ${environment?.python_version ?? '—'}`,
      `- NumPy: ${environment?.numpy_version ?? '—'}`,
      `- OS: ${environment?.os ?? '—'}`,
      `- Machine: ${environment?.machine ?? '—'}`,
      '',
      '## Observed winners by case',
      '',
      '| Strategy selected as fastest valid | Case count | Share of completed cases |',
      '|---|---:|---:|',
      ...[...winners.entries()].sort((a, b) => b[1] - a[1]).map(([name, count]) => `| ${name} | ${count} | ${(100 * count / Math.max(1, completed.length)).toFixed(1)}% |`),
      '',
      'Winner frequency is descriptive for this exact experiment matrix. It is not an aggregate runtime score and depends on the distribution of workloads included in the sweep.',
      '',
      '## Per-case measurements',
      '',
      '| # | Pattern | Points | Channels | Grid | Seed | Collision rate | Fastest valid strategy | Median | Correctness gate |',
      '|---:|---|---:|---:|---:|---:|---:|---|---:|---|',
      ...sweepResults.cases.map(c => {
        const e = c.result
        const winner = chooseWinner(e)
        return `| ${c.case_index} | ${c.config.pattern} | ${c.config.points} | ${c.config.channels} | ${c.config.grid_size}³ | ${c.config.seed} | ${e ? `${e.occupancy.collision_rate_percent.toFixed(2)}%` : '—'} | ${winner?.label ?? c.status} | ${winner ? formatMs(winner.timing?.median_ms) : '—'} | ${e ? (e.correctness.all_candidates_passed ? 'all passed' : 'review required') : (c.reason ?? c.status)} |`
      }),
      '',
      '## Correctness and timing protocol',
      '',
      '- The sequential Python implementation is the correctness reference.',
      '- Candidate results are compared using `rtol=1e-4` and `atol=1e-4`.',
      '- A candidate that fails correctness is not eligible for a timing-based winner.',
      '- The measured timing includes strategy-specific preprocessing.',
      '- Each strategy is warmed up once; the table reports the median from the requested repeat count.',
      '- The sequential Python reference is intentionally simple and is not a strong optimized CPU baseline.',
      '',
      '## Workload-aware selector evaluation',
      '',
    ]
    if (selectorResult) {
      lines.push(
        `- Method: ${selectorResult.method}`,
        `- Training / held-out configurations: ${selectorResult.training_configurations} / ${selectorResult.heldout_configurations}`,
        `- Winner accuracy: ${selectorResult.selector_accuracy_percent.toFixed(2)}%`,
        `- Selector mean / median regret: ${selectorResult.selector_mean_regret_percent.toFixed(3)}% / ${selectorResult.selector_median_regret_percent.toFixed(3)}%`,
        `- Training-only baseline: ${selectorResult.baseline_label}`,
        `- Baseline accuracy: ${selectorResult.baseline_accuracy_percent.toFixed(2)}%`,
        `- Baseline mean regret: ${selectorResult.baseline_mean_regret_percent.toFixed(3)}%`,
        `- Split policy: ${selectorResult.split_policy}`,
        '',
        '| Held-out workload | Selector choice | Measured winner | Selector regret | Baseline regret |',
        '|---|---|---|---:|---:|',
        ...selectorResult.evaluation_cases.map(item => `| ${item.workload.pattern}, ${item.workload.points} pts, ${item.workload.channels} ch, grid ${item.workload.grid_size}³ | ${item.selected_label} | ${item.oracle_fastest_label} | ${item.regret_percent.toFixed(3)}% | ${item.baseline_regret_percent.toFixed(3)}% |`),
      )
    } else {
      lines.push('The selector has not been evaluated for this sweep. Run the held-out evaluation from the Performance Landscape screen.', '')
    }
    lines.push(
      '',
      '## Limitations and interpretation',
      '',
      '- These are measurements on one local CPU/software environment; timing varies with system load and machine setup.',
      '- Synthetic distributions are not a substitute for workload traces taken from a real perception pipeline.',
      '- The selector evaluation uses a small deterministic configuration-level holdout. A stronger study should repeat group-aware splits, broaden workload dimensions and include more seeds.',
      '- Regret compares the selected implementation with the fastest measured valid candidate on the same held-out workload. It is not a forecast for unmeasured hardware.',
      '- No Quadric CCL kernel, CUDA implementation, or accelerator result is included.',
      '',
      '## Related work and references',
      '',
      '- [KernelBench](https://github.com/ScalingIntelligence/KernelBench)',
      '- [Kernel Tuner](https://github.com/KernelTuner/kernel_tuner)',
      '- [TritonBench](https://github.com/meta-pytorch/tritonbench)',
      '- [Quadric custom-op tutorials](https://app.quadric.ai/docs/latest/chimera-software-user-guide/tutorials-model-demos/custom-op-tutorials/index)',
      '',
      'This report is generated from the selected sweep records. It does not invent or infer missing measurements.',
      '',
    )
    downloadText('voxelpool-workload-sweep-report.md', lines.join('\n'), 'text/markdown;charset=utf-8')
  }
  function exportSweepCSV() {
    if (sweepResults) downloadText('voxelpool-sweep-results.csv', sweepCaseCsv(sweepResults), 'text/csv;charset=utf-8')
  }

  const landscapeCells = useMemo(() => {
    const completed = (sweepResults?.cases ?? []).filter(c => c.status === 'completed' && c.result)
    const buckets = new Map<string, { pattern: Pattern; points: number; channels: number; grid_size: number; seeds: number[]; values: Map<string, number[]> }>()
    for (const c of completed) {
      const e = c.result!
      const key = `${e.workload.pattern}|${e.workload.grid_size}|${e.workload.channels}|${e.workload.points}`
      if (!buckets.has(key)) buckets.set(key, { pattern: e.workload.pattern, points: e.workload.points, channels: e.workload.channels, grid_size: e.workload.grid_size, seeds: [], values: new Map() })
      const bucket = buckets.get(key)!
      bucket.seeds.push(e.workload.seed)
      for (const strategy of e.strategies) {
        if (!strategy.correct || !strategy.timing) continue
        const values = bucket.values.get(strategy.id) ?? []
        values.push(strategy.timing.median_ms)
        bucket.values.set(strategy.id, values)
      }
    }
    return [...buckets.values()].map(bucket => {
      const medians = [...bucket.values.entries()].map(([id, values]) => {
        const sorted = [...values].sort((a, b) => a - b)
        const mid = Math.floor(sorted.length / 2)
        return { id, median: sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2 }
      }).sort((a, b) => a.median - b.median)
      return { ...bucket, medians, winner: medians[0] ?? null, runnerUp: medians[1] ?? null }
    }).sort((a, b) => a.pattern.localeCompare(b.pattern) || a.grid_size - b.grid_size || a.channels - b.channels || a.points - b.points)
  }, [sweepResults])

  function exportCSV() {
    if (!history.length) return
    // Export from the browser's saved history, not the backend's volatile cache.
    downloadText('voxelpool-experiments.csv', buildHistoryCSV(history), 'text/csv;charset=utf-8')
  }

  function clearHistory() {
    if (!window.confirm('Clear the locally saved experiments from this browser? Export them first if you need a backup.')) return
    setHistory([]); setResult(null)
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark"><Waypoints size={21} /></div>
        <div><div className="brand-name">voxelpool<span>.</span></div><div className="brand-caption">RESEARCH LAB</div></div>
      </div>
      <div className="side-label">WORKSPACE</div>
      <button className={`nav-item ${tab === 'explore' ? 'active' : ''}`} onClick={() => setTab('explore')}><Grid3X3 size={17} /> Workload explorer</button>
      <button className={`nav-item ${tab === 'results' ? 'active' : ''}`} onClick={() => setTab('results')}><Activity size={17} /> Experiment results <span className="nav-count">{history.length}</span></button>
      <button className={`nav-item ${tab === 'sweep' ? 'active' : ''}`} onClick={() => setTab('sweep')}><Telescope size={17} /> Workload sweeps <span className="nav-count">{sweepResults?.completed_cases ?? 0}</span></button>
      <button className={`nav-item ${tab === 'landscape' ? 'active' : ''}`} onClick={() => setTab('landscape')}><GitCompare size={17} /> Performance landscape</button>
      <button className={`nav-item ${tab === 'methodology' ? 'active' : ''}`} onClick={() => setTab('methodology')}><FlaskConical size={17} /> Methodology</button>
      <div className="sidebar-spacer" />
      <div className="sidebar-status">
        <div className="side-label">RUNTIME STATUS</div>
        <div className="status-line"><span className={`status-dot ${apiState}`} />{apiState === 'online' ? 'CPU worker connected' : apiState === 'offline' ? 'API disconnected' : 'Checking API...'}</div>
        <div className="runtime-chip"><Cpu size={14} /><span>Portable CPU · NumPy</span></div>
      </div>
      <div className="sidebar-foot">VOXELPOOL EXPLORER <span>v1.2.0</span></div>
    </aside>

    <main className="main-content">
      <header className="topbar">
        <div className="breadcrumbs"><span>Workspace</span><span className="crumb-sep">/</span><strong>{tab === 'explore' ? 'Workload explorer' : tab === 'results' ? 'Experiment results' : tab === 'sweep' ? 'Workload sweeps' : tab === 'landscape' ? 'Performance landscape' : 'Methodology'}</strong></div>
        <div className="topbar-right">
          <div className={`connection-pill ${apiState}`}><span className="status-dot" />{apiState === 'online' ? 'Local engine online' : apiState === 'offline' ? 'Engine offline' : 'Connecting'}</div>
          <button className="icon-button" title="Refresh API status" onClick={checkHealth}><RefreshCw size={16} /></button>
        </div>
      </header>

      <div className="page-wrap">
        <section className="page-heading">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> EDGE PERCEPTION / KERNEL RESEARCH</div>
            <h1>{tab === 'explore' ? <>Understand the <span className="gradient-text">workload.</span></> : tab === 'results' ? <>Measure what <span className="gradient-text">matters.</span></> : tab === 'sweep' ? <>Explore the <span className="gradient-text">landscape.</span></> : tab === 'landscape' ? <>Find the <span className="gradient-text">pattern.</span></> : <>Research with <span className="gradient-text">evidence.</span></>}</h1>
            <p>{tab === 'explore' ? 'Explore destination collisions, feature shapes, and accumulation behavior before committing to an optimization.' : tab === 'results' ? 'Correctness-gated CPU measurements across the current workload and saved experiments.' : tab === 'sweep' ? 'Run a controlled matrix of workloads, preserve every result, and build evidence for the strategy that wins.' : tab === 'landscape' ? 'Compare strategy winners across workload configurations and evaluate a selector on held-out configurations.' : 'A transparent, reproducible workflow inspired by established kernel benchmark systems.'}</p>
          </div>
          <div className="heading-chip"><Beaker size={16} /><span>EXPERIMENTAL · CPU ONLY</span></div>
        </section>

        <div className="notice"><ShieldCheck size={17} /><div><strong>Research integrity is on.</strong> Every candidate is checked against a sequential reference before its timing is accepted. <span>CPU measurements only — no Quadric or GPU claims.</span></div></div>
        {error && <div className="error-banner"><XCircle size={17} /><span>{error}</span><button onClick={() => setError('')}><XCircle size={15} /></button></div>}

        {tab === 'explore' && <div className="explorer-layout">
          <section className="panel config-panel">
            <div className="panel-head"><div><div className="panel-kicker">01 / CONFIGURATION</div><h2>Workload parameters</h2></div><SlidersHorizontal size={18} className="muted-icon" /></div>
            <div className="field-block">
              <div className="field-top"><label htmlFor="points">Point count</label><strong>{fmt(config.points)}</strong></div>
              <input id="points" type="range" min="64" max="6000" step="64" value={config.points} onChange={e => update('points', Number(e.target.value))} />
              <div className="range-labels"><span>64 points</span><span>6,000 points</span></div>
            </div>
            <div className="field-block">
              <div className="field-top"><label htmlFor="channels">Feature channels</label><strong>{config.channels}</strong></div>
              <input id="channels" type="range" min="4" max="128" step="4" value={config.channels} onChange={e => update('channels', Number(e.target.value))} />
              <div className="range-labels"><span>4 channels</span><span>128 channels</span></div>
            </div>
            <div className="field-block">
              <div className="field-top"><label htmlFor="grid">Voxel grid dimension</label><strong>{config.grid_size}³</strong></div>
              <input id="grid" type="range" min="4" max="48" step="4" value={config.grid_size} onChange={e => update('grid_size', Number(e.target.value))} />
              <div className="range-labels"><span>4 × 4 × 4</span><span>48 × 48 × 48</span></div>
            </div>
            <div className="field-block">
              <div className="field-top"><label>Destination distribution</label><span className="field-hint">Select a scenario</span></div>
              <div className="pattern-options">
                <button className={`pattern-option ${config.pattern === 'uniform' ? 'selected' : ''}`} onClick={() => update('pattern', 'uniform')}>
                  <div className="pattern-icon uniform"><Grid3X3 size={17} /></div><div><strong>Uniform</strong><span>Spread across grid</span></div>{config.pattern === 'uniform' && <Check size={15} className="pattern-check" />}
                </button>
                <button className={`pattern-option ${config.pattern === 'hotspot' ? 'selected' : ''}`} onClick={() => update('pattern', 'hotspot')}>
                  <div className="pattern-icon hotspot"><Waypoints size={17} /></div><div><strong>Hotspot</strong><span>Concentrated collisions</span></div>{config.pattern === 'hotspot' && <Check size={15} className="pattern-check" />}
                </button>
                <button className={`pattern-option ${config.pattern === 'unique' ? 'selected' : ''}`} onClick={() => update('pattern', 'unique')}>
                  <div className="pattern-icon unique"><Layers3 size={17} /></div><div><strong>Unique</strong><span>No repeated targets</span></div>{config.pattern === 'unique' && <Check size={15} className="pattern-check" />}
                </button>
              </div>
              {config.pattern === 'unique' && !canRun && <div className="field-warning">Unique mode requires points ≤ grid³ ({fmt(config.grid_size ** 3)}). Increase grid size or reduce point count.</div>}
            </div>
            <button className="advanced-toggle" onClick={() => setShowAdvanced(v => !v)}>{showAdvanced ? 'Hide' : 'Show'} advanced settings <ChevronDown size={15} className={showAdvanced ? 'rotate' : ''} /></button>
            {showAdvanced && <div className="advanced-fields">
              <div className="inline-field"><label htmlFor="repeats">Timing repeats</label><select id="repeats" value={config.repeats} onChange={e => update('repeats', Number(e.target.value))}>{[3,5,7,10,15,20].map(n => <option key={n} value={n}>{n} runs</option>)}</select></div>
              <div className="inline-field"><label htmlFor="seed">Random seed</label><input id="seed" type="number" min="0" max="2147483647" value={config.seed} onChange={e => update('seed', Math.max(0, Number(e.target.value) || 0))} /></div>
            </div>}
            <div className="config-footer">
              <div className="config-summary"><span>Grid capacity</span><strong>{fmt(config.grid_size ** 3)} voxels</strong></div>
              <button className="primary-button" onClick={runExperiment} disabled={busy || !canRun || apiState === 'offline'}>
                {busy ? <><span className="spinner" /> Running experiment</> : <><Play size={16} fill="currentColor" /> Run experiment <ArrowUpRight size={15} /></>}
              </button>
              {apiState === 'offline' && <div className="field-warning">Start the backend first. See README.md.</div>}
            </div>
          </section>

          <div className="right-column">
            <section className="panel preview-panel">
              <div className="panel-head"><div><div className="panel-kicker">02 / SPATIAL DISTRIBUTION</div><h2>Voxel occupancy map</h2></div><span className={`live-tag ${previewStatus === 'loading' ? 'is-loading' : ''}`}><span /> {previewStatus === 'ready' && previewMatchesConfig ? 'LIVE PREVIEW' : previewStatus === 'loading' ? 'UPDATING' : previewStatus === 'offline' ? 'ENGINE OFFLINE' : previewStatus === 'invalid' ? 'INVALID CONFIG' : previewStatus === 'error' ? 'PREVIEW ERROR' : 'CONNECTING'}</span></div>
              {preview && previewMatchesConfig ? <VoxelMap projection={preview.occupancy.xy_projection} gridSize={preview.workload.grid_size} /> :
                <div className="empty-map"><div className="empty-map-grid"><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /><span /></div><div><strong>{previewStatus === 'loading' ? 'Updating occupancy preview' : previewStatus === 'offline' ? 'CPU worker is offline' : previewStatus === 'invalid' ? 'No preview for this configuration' : previewStatus === 'error' ? 'Preview unavailable' : 'Preparing occupancy preview'}</strong><p>{previewStatus === 'invalid' ? 'Unique destinations require point count to be no greater than grid capacity.' : previewStatus === 'offline' ? 'Start the backend to generate the seeded preview.' : 'This preview uses the same seeded workload generator as the benchmark; no strategy timing has run yet.'}</p></div></div>}
              <div className="map-foot"><div><span className="map-dot mint" />Occupied voxels</div><div><span className="map-dot gray" />Empty voxels</div><span className="projection-label">SUM OVER Z</span></div>
            </section>

            <section className="panel stats-panel">
              <div className="panel-head"><div><div className="panel-kicker">03 / COLLISION PROFILE</div><h2>Workload fingerprint</h2></div><Activity size={18} className="muted-icon" /></div>
              <div className="fingerprint-grid">
                <div className="fingerprint-stat"><span>Grid capacity</span><strong>{fmt(config.grid_size ** 3)}</strong><small>total destinations</small></div>
                <div className="fingerprint-stat"><span>Feature volume</span><strong>{fmt(config.points * config.channels)}</strong><small>float values</small></div>
                <div className="fingerprint-stat"><span>Occupancy</span><strong>{previewMatchesConfig && preview ? `${preview.occupancy.occupancy_percent.toFixed(2)}%` : '—'}</strong><small>voxels reached</small></div>
                <div className="fingerprint-stat"><span>Collision points</span><strong>{previewMatchesConfig && preview ? `${preview.occupancy.collision_rate_percent.toFixed(1)}%` : '—'}</strong><small>points beyond first hit</small></div>
              </div>
              <div className="fingerprint-note"><Sparkles size={15} /><span>Hotspot mode deliberately increases repeated destinations. Profile before assuming collisions are the bottleneck.</span></div>
            </section>

            <section className="hypothesis-card">
              <div className="hypothesis-icon"><CircleHelp size={18} /></div><div><div className="panel-kicker">RESEARCH HYPOTHESIS</div><strong>Does destination structure change the best strategy?</strong><p>Compare uniform, hotspot and unique distributions while holding the other parameters fixed. Validate any winner on unseen workloads.</p></div>
            </section>
          </div>
        </div>}

        {tab === 'results' && <div className="results-layout">
          {!current ? <section className="panel no-results"><div className="no-results-icon"><Activity size={24} /></div><h2>No experiment yet</h2><p>Configure a workload and run your first correctness-gated experiment.</p><button className="primary-button" onClick={() => setTab('explore')}>Open workload explorer <ArrowUpRight size={15} /></button></section> : <>
            <section className="metrics-grid">
              <div className="metric-card"><div className="metric-top"><span>REFERENCE MEDIAN</span><Clock3 size={16} /></div><strong>{formatMs(current.strategies[0]?.timing?.median_ms)}</strong><small>Sequential Python reference</small></div>
              <div className="metric-card"><div className="metric-top"><span>FASTEST VALID</span><Activity size={16} /></div><strong>{best ? formatMs(best.timing?.median_ms) : '—'}</strong><small>{best?.label ?? 'No valid strategy'}</small></div>
              <div className="metric-card"><div className="metric-top"><span>COLLISION RATE</span><Waypoints size={16} /></div><strong>{current.occupancy.collision_rate_percent.toFixed(2)}%</strong><small>{fmt(current.occupancy.collision_points)} points beyond first hit</small></div>
              <div className="metric-card"><div className="metric-top"><span>CORRECTNESS GATE</span><ShieldCheck size={16} /></div><strong className={current.correctness.all_candidates_passed ? 'success-text' : 'warning-text'}>{current.correctness.all_candidates_passed ? 'All passed' : 'Review needed'}</strong><small>rtol {current.correctness.rtol} · atol {current.correctness.atol}</small></div>
            </section>
            <section className="panel">
              <div className="panel-head"><div><div className="panel-kicker">04 / PERFORMANCE</div><h2>Strategy comparison</h2></div><button className="secondary-button" onClick={() => downloadJSON(current)}><Download size={15} /> Export JSON</button></div>
              <div className="chart-caption"><span>Median wall-clock time · lower is better</span><span className="unit-tag">MILLISECONDS</span></div>
              <TimingChart strategies={current.strategies} />
              <div className="table-wrap"><table><thead><tr><th>Implementation</th><th>Correctness</th><th>Median</th><th>Min / max</th><th>CV</th><th>vs. reference</th></tr></thead><tbody>
                {current.strategies.map(s => <tr key={s.id}><td><div className="strategy-name">{s.label}<small>{s.id}</small></div></td><td><span className={`result-status ${s.correct ? 'pass' : 'fail'}`}>{s.correct ? <Check size={12} /> : <XCircle size={12} />}{s.correct ? 'PASS' : 'FAIL'}</span></td><td className="mono">{formatMs(s.timing?.median_ms)}</td><td className="mono">{s.timing ? `${formatMs(s.timing.min_ms)} / ${formatMs(s.timing.max_ms)}` : '—'}</td><td className="mono">{s.timing?.coefficient_of_variation_percent == null ? '—' : `${s.timing.coefficient_of_variation_percent.toFixed(1)}%`}</td><td>{s.speedup_vs_reference == null ? '—' : <span className={s.speedup_vs_reference >= 1 ? 'success-text' : 'warning-text'}>{s.speedup_vs_reference.toFixed(3)}×</span>}</td></tr>)}
              </tbody></table></div>
              <div className="result-note"><ShieldCheck size={15} /><span>Incorrect candidates are not timed for scoring. Speedup is relative to the sequential Python reference on this same CPU only.</span></div>
            </section>
            <section className="panel">
              <div className="panel-head"><div><div className="panel-kicker">05 / WORKLOAD RECORD</div><h2>Reproducibility metadata</h2></div><button className="secondary-button" onClick={exportCSV}><Download size={15} /> Export history CSV</button></div>
              <div className="metadata-grid">
                <div><span>Experiment ID</span><code>{current.experiment_id.slice(0, 18)}…</code></div><div><span>Backend</span><code>{current.measurement_backend}</code></div>
                <div><span>Software environment</span><code>Python {current.environment?.python_version ?? '—'} · NumPy {current.environment?.numpy_version ?? '—'}</code></div>
                <div><span>Machine</span><code>{current.environment?.machine ?? '—'} · {(current.environment?.os ?? 'OS unknown').split(' ').slice(0, 2).join(' ')}</code></div>
                <div><span>Input shape</span><code>[{current.workload.points}, {current.workload.channels}]</code></div><div><span>Grid</span><code>{current.workload.grid_size}³ = {fmt(current.workload.num_voxels)} voxels</code></div>
                <div><span>Distribution / seed</span><code>{current.workload.pattern} / {current.workload.seed}</code></div><div><span>Timing policy</span><code>1 warm-up + {current.methodology.measured_repeats} measured runs</code></div>
              </div>
              <div className="warning-note"><Terminal size={16} /><span>{current.warning}</span></div>
            </section>
            {history.length > 0 && <section className="panel"><div className="panel-head"><div><div className="panel-kicker">06 / LOCAL HISTORY</div><h2>Recent experiments</h2></div><div className="panel-actions"><span className="field-hint">{history.length} saved in this browser</span><button className="secondary-button" onClick={clearHistory}>Clear history</button></div></div>
              <div className="history-list">{history.map(h => <button key={h.experiment_id} className={`history-item ${h.experiment_id === current.experiment_id ? 'selected' : ''}`} onClick={() => setResult(h)}>
                <div className="history-icon"><Activity size={16} /></div><div className="history-main"><strong>{fmt(h.workload.points)} points · {h.workload.channels} channels</strong><span>{h.workload.pattern} · grid {h.workload.grid_size}³ · seed {h.workload.seed}</span></div><div className="history-time">{formatMs(h.strategies[0]?.timing?.median_ms)}<small>reference</small></div><ChevronDown size={15} className="history-chevron" /></button>)}</div>
            </section>}
          </>}
        </div>}

        {tab === 'sweep' && <div className="sweep-layout">
          {sweepError && <div className="error-banner"><XCircle size={17} /><span>{sweepError}</span><button onClick={() => setSweepError('')}><XCircle size={15} /></button></div>}
          <section className="panel sweep-config-panel">
            <div className="panel-head"><div><div className="panel-kicker">01 / STUDY DESIGN</div><h2>Build a controlled workload matrix</h2></div><Telescope size={18} className="muted-icon" /></div>
            <p className="panel-description">Each combination gets deterministic inputs and a correctness-gated run for every strategy. Seeds are separate repeats of a configuration, not independent workload shapes.</p>
            <div className="sweep-choice-grid">
              <ChoiceChips label="Point counts" options={[128, 256, 1200, 2500].map(value => ({ value, label: fmt(value) }))} selected={sweepConfig.points} onToggle={value => toggleSweepDimension('points', value)} />
              <ChoiceChips label="Feature channels" options={[4, 8, 16, 32, 64].map(value => ({ value, label: String(value) }))} selected={sweepConfig.channels} onToggle={value => toggleSweepDimension('channels', value)} />
              <ChoiceChips label="Voxel grid size" options={[8, 16, 32].map(value => ({ value, label: `${value}³` }))} selected={sweepConfig.grid_sizes} onToggle={value => toggleSweepDimension('grid_sizes', value)} />
              <ChoiceChips label="Destination distributions" options={[{ value: 'uniform' as Pattern, label: 'Uniform' }, { value: 'hotspot' as Pattern, label: 'Hotspot' }, { value: 'unique' as Pattern, label: 'Unique' }]} selected={sweepConfig.patterns} onToggle={value => toggleSweepDimension('patterns', value)} />
              <ChoiceChips label="Random seeds" options={[42, 43, 44].map(value => ({ value, label: `Seed ${value}` }))} selected={sweepConfig.seeds} onToggle={value => toggleSweepDimension('seeds', value)} />
              <div className="sweep-choice-group"><div className="sweep-choice-label">Timing repeats per strategy</div><select className="sweep-repeats" value={sweepConfig.repeats} onChange={e => setSweepConfig(prev => ({ ...prev, repeats: Number(e.target.value) }))}>{[1, 3, 5].map(value => <option key={value} value={value}>{value} {value === 1 ? 'repeat' : 'repeats'}</option>)}</select><p className="micro muted">Each strategy also receives one warm-up call.</p></div>
            </div>
            <div className="sweep-launch-row">
              <div className="sweep-case-summary"><strong>{fmt(sweepCaseCount)}</strong><span>planned workload cases</span><small>{sweepConfig.points.length} point sizes × {sweepConfig.channels.length} channel counts × {sweepConfig.grid_sizes.length} grids × {sweepConfig.patterns.length} patterns × {sweepConfig.seeds.length} seeds</small></div>
              <div className="sweep-launch-actions"><span className={`case-cap ${sweepCaseCount > 48 ? 'over-cap' : ''}`}>{sweepCaseCount > 48 ? 'Limit: 48 cases' : `${48 - sweepCaseCount} case slots available`}</span><button className="primary-button" onClick={runSweep} disabled={sweepBusy || apiState !== 'online' || sweepCaseCount > 48}>{sweepBusy ? <><span className="spinner" /> Running {sweepProgress?.progress ?? 0}/{sweepProgress?.total_cases ?? sweepCaseCount}</> : <><Play size={16} fill="currentColor" /> Run workload sweep <ArrowUpRight size={15} /></>}</button></div>
            </div>
            {apiState !== 'online' && <div className="field-warning">Connect the local CPU worker to run a sweep.</div>}
          </section>

          {sweepBusy && <section className="panel sweep-progress-panel"><div className="panel-head"><div><div className="panel-kicker">02 / EXECUTION</div><h2>Running controlled experiments</h2></div><span className="live-tag is-loading"><span /> IN PROGRESS</span></div><div className="progress-meta"><span>{sweepProgress?.completed_cases ?? 0} completed · {sweepProgress?.skipped_cases ?? 0} skipped · {sweepProgress?.error_cases ?? 0} errors</span><strong>{sweepProgress ? Math.round(100 * sweepProgress.progress / Math.max(1, sweepProgress.total_cases)) : 0}%</strong></div><progress value={sweepProgress?.progress ?? 0} max={sweepProgress?.total_cases ?? sweepCaseCount} /><p className="micro muted">Do not compare runs collected on different hardware as if they were a single controlled study. Keep this machine under a reasonably steady load while the sweep runs.</p></section>}

          {sweepResults?.cases && <>
            <section className="metrics-grid sweep-metrics">
              <div className="metric-card"><div className="metric-top"><span>CASES FINISHED</span><Activity size={16} /></div><strong>{fmt(sweepResults.completed_cases)}</strong><small>of {sweepResults.total_cases} requested</small></div>
              <div className="metric-card"><div className="metric-top"><span>SKIPPED</span><Layers3 size={16} /></div><strong>{fmt(sweepResults.skipped_cases)}</strong><small>constraints or memory guardrail</small></div>
              <div className="metric-card"><div className="metric-top"><span>EXECUTION ERRORS</span><XCircle size={16} /></div><strong>{fmt(sweepResults.error_cases)}</strong><small>inspect rows for details</small></div>
              <div className="metric-card"><div className="metric-top"><span>SWEEP ID</span><ShieldCheck size={16} /></div><strong className="sweep-id-value">{sweepResults.sweep_id.slice(0, 8)}</strong><small>stored in this browser</small></div>
            </section>
            <section className="panel">
              <div className="panel-head"><div><div className="panel-kicker">03 / EXPERIMENT MATRIX</div><h2>Per-workload results</h2></div><div className="panel-actions"><button className="secondary-button" onClick={exportSweepCSV}><FileSpreadsheet size={15} /> Export CSV</button><button className="secondary-button" onClick={exportResearchMarkdown}><FileSpreadsheet size={15} /> Research report</button><button className="secondary-button" onClick={exportSweepJSON}><Download size={15} /> Research bundle</button></div></div>
              <p className="panel-description">Rows represent one seed/configuration combination. The winner is the fastest correctly measured candidate for that workload; it is not a universal winner.</p>
              <div className="table-wrap"><table className="sweep-results-table"><thead><tr><th>Case</th><th>Workload</th><th>Seed</th><th>Status</th><th>Fastest valid</th><th>Median</th><th>Collision</th><th>Details</th></tr></thead><tbody>
                {sweepResults.cases.map(c => { const winner = chooseWinner(c.result); return <tr key={c.case_index}><td className="mono">{String(c.case_index).padStart(2, '0')}</td><td><div className="strategy-name">{c.config.pattern}<small>{fmt(c.config.points)} points · {c.config.channels} ch · grid {c.config.grid_size}³</small></div></td><td className="mono">{c.config.seed}</td><td><span className={`result-status ${c.status === 'completed' ? 'pass' : c.status === 'skipped' ? 'skip' : 'fail'}`}>{c.status.toUpperCase()}</span></td><td>{winner ? <span className={`winner-label winner-${winner.id}`}>{winner.label}</span> : '—'}</td><td className="mono">{formatMs(winner?.timing?.median_ms)}</td><td className="mono">{c.result ? `${c.result.occupancy.collision_rate_percent.toFixed(1)}%` : '—'}</td><td className="table-reason">{c.reason ?? (c.result?.correctness.all_candidates_passed ? 'All timed candidates passed' : c.result ? 'Review correctness gate' : '—')}</td></tr> })}
              </tbody></table></div>
              <div className="result-note"><ShieldCheck size={15} /><span>The performance landscape below aggregates seed replicates by workload configuration. The selector evaluation keeps all replicates for a configuration together in either training or testing.</span></div>
            </section>
            <section className="panel selector-callout"><div className="selector-callout-icon"><BrainCircuit size={21} /></div><div className="selector-callout-content"><div className="panel-kicker">NEXT / HELD-OUT EVALUATION</div><h2>Can workload features predict the winner?</h2><p>Evaluate a three-nearest-neighbour selector against a training-only global-best baseline. Entire workload configurations—not individual seed rows—are held out from training.</p>{selectorResult && <span className="field-hint">Latest selector result: {selectorResult.heldout_configurations} held-out configurations.</span>}</div><button className="primary-button" onClick={evaluateSelector} disabled={selectorBusy || sweepBusy || sweepResults.completed_cases < 4}>{selectorBusy ? <><span className="spinner" /> Evaluating</> : <><BrainCircuit size={16} /> Evaluate selector <ArrowUpRight size={15} /></>}</button></section>
          </>}
          {!sweepResults?.cases && !sweepBusy && <section className="panel no-results"><div className="no-results-icon"><Telescope size={24} /></div><h2>No workload sweep yet</h2><p>Start with the default 48-case study to compare point counts, channel counts, grid sizes, destination distributions and random seeds.</p><button className="primary-button" onClick={runSweep} disabled={apiState !== 'online' || sweepCaseCount > 48}><Play size={16} /> Run starter study</button></section>}
        </div>}

        {tab === 'landscape' && <div className="landscape-layout">
          {sweepError && <div className="error-banner"><XCircle size={17} /><span>{sweepError}</span><button onClick={() => setSweepError('')}><XCircle size={15} /></button></div>}
          {!sweepResults?.cases && <section className="panel no-results"><div className="no-results-icon"><GitCompare size={24} /></div><h2>Landscape not available yet</h2><p>Run a workload sweep first. The landscape only displays measured, correctness-eligible CPU results.</p><button className="primary-button" onClick={() => setTab('sweep')}>Open workload sweeps <ArrowUpRight size={15} /></button></section>}
          {sweepResults?.cases && <>
            <section className="panel landscape-intro"><div className="panel-head"><div><div className="panel-kicker">01 / STRATEGY WINNERS</div><h2>Performance landscape</h2></div><GitCompare size={18} className="muted-icon" /></div><p className="panel-description">Each cell compares the median of seed-level timing medians for one (pattern, grid, channels, point count) configuration. Tint indicates the fastest correct strategy. If the second-place timing is close, treat the apparent win cautiously.</p><div className="landscape-legend"><span><i className="legend-swatch strategy-python" /> Sequential Python</span><span><i className="legend-swatch strategy-numpy" /> NumPy scatter</span><span><i className="legend-swatch strategy-sort" /> Sort + segmented</span></div></section>
            {['uniform', 'hotspot', 'unique'].map(pattern => [8, 16, 32].map(gridSize => {
              const selected = landscapeCells.filter(cell => cell.pattern === pattern && cell.grid_size === gridSize)
              if (!selected.length) return null
              const points = [...new Set(selected.map(cell => cell.points))].sort((a, b) => a - b)
              const channels = [...new Set(selected.map(cell => cell.channels))].sort((a, b) => a - b)
              return <section className="panel landscape-panel" key={`${pattern}-${gridSize}`}><div className="panel-head"><div><div className="panel-kicker">{pattern.toUpperCase()} DISTRIBUTION · GRID {gridSize}³</div><h2>Point count × feature channels</h2></div><span className="unit-tag">MEDIAN MS</span></div><div className="heatmap-scroll"><div className="landscape-matrix" style={{ gridTemplateColumns: `minmax(92px, 1.15fr) repeat(${points.length}, minmax(115px, 1fr))` }}><div className="landscape-axis corner">Channels ↓ / Points →</div>{points.map(point => <div className="landscape-axis" key={point}>{fmt(point)} pts</div>)}{channels.flatMap(channel => [<div className="landscape-row-label" key={`row-${channel}`}>{channel} channels</div>, ...points.map(point => { const cell = selected.find(item => item.channels === channel && item.points === point); const winner = cell?.winner; const short = winner?.id === 'python_reference' ? 'PYTHON' : winner?.id === 'numpy_add_at' ? 'ADD.AT' : winner?.id === 'sort_segmented' ? 'SORT + SEG' : 'NO RESULT'; const ratio = winner && cell?.runnerUp ? cell.runnerUp.median / winner.median : null; return <div key={`${channel}-${point}`} className={`landscape-cell ${winner ? `strategy-${winner.id}` : 'strategy-none'}`} title={cell ? `${pattern}; grid ${gridSize}³; ${channel} channels; ${point} points; winner ${winner?.id ?? 'none'}; ${winner ? formatMs(winner.median) : 'no timing'}; ${cell.seeds.length} seed(s)` : 'No completed result for this configuration'}><strong>{short}</strong><span>{winner ? formatMs(winner.median) : '—'}</span><small>{ratio == null ? 'single candidate' : `${ratio.toFixed(2)}× vs next`}</small></div> })])}</div></div></section>
            }))}
            <section className="panel selector-panel"><div className="panel-head"><div><div className="panel-kicker">02 / HELD-OUT STRATEGY SELECTOR</div><h2>Does workload awareness help?</h2></div><BrainCircuit size={18} className="muted-icon" /></div>{!selectorResult ? <div className="selector-empty"><div><strong>Selector has not been evaluated for this sweep.</strong><p>Run the deterministic configuration-level evaluation to compare a nearest-neighbour selector against the best single strategy selected using training data only.</p></div><button className="primary-button" onClick={evaluateSelector} disabled={selectorBusy || sweepBusy || sweepResults.completed_cases < 4}>{selectorBusy ? <><span className="spinner" /> Evaluating</> : 'Evaluate held-out cases'}</button></div> : <>
              <div className="metrics-grid selector-metrics"><div className="metric-card"><div className="metric-top"><span>HELD-OUT CONFIGURATIONS</span><ShieldCheck size={16} /></div><strong>{selectorResult.heldout_configurations}</strong><small>{selectorResult.training_configurations} training configurations</small></div><div className="metric-card"><div className="metric-top"><span>SELECTOR ACCURACY</span><BrainCircuit size={16} /></div><strong>{selectorResult.selector_accuracy_percent.toFixed(1)}%</strong><small>matches the measured winner</small></div><div className="metric-card"><div className="metric-top"><span>SELECTOR MEAN REGRET</span><Activity size={16} /></div><strong>{selectorResult.selector_mean_regret_percent.toFixed(2)}%</strong><small>over fastest measured strategy</small></div><div className="metric-card"><div className="metric-top"><span>GLOBAL-BASELINE REGRET</span><GitCompare size={16} /></div><strong>{selectorResult.baseline_mean_regret_percent.toFixed(2)}%</strong><small>{selectorResult.baseline_label} on training data</small></div></div>
              <div className="selector-verdict"><BrainCircuit size={18} /><div><strong>{selectorResult.selector_mean_regret_percent < selectorResult.baseline_mean_regret_percent ? 'Selector improved on the training-only baseline in this test set.' : selectorResult.selector_mean_regret_percent > selectorResult.baseline_mean_regret_percent ? 'The simple global baseline performed better on this test set.' : 'The selector and baseline had equal mean regret on this test set.'}</strong><p>Do not generalize from a small sweep. Repeat with more seeds and held-out configurations before treating this as a reliable rule.</p></div></div>
              <div className="table-wrap"><table><thead><tr><th>Held-out workload</th><th>Selector chooses</th><th>Measured winner</th><th>Selector regret</th><th>Baseline regret</th></tr></thead><tbody>{selectorResult.evaluation_cases.map((item, index) => <tr key={`${item.workload.pattern}-${item.workload.points}-${item.workload.channels}-${item.workload.grid_size}-${index}`}><td><div className="strategy-name">{item.workload.pattern}<small>{item.workload.points} pts · {item.workload.channels} channels · grid {item.workload.grid_size}³</small></div></td><td>{item.selected_label}<small className="table-subline">{formatMs(item.selected_median_ms)}</small></td><td>{item.oracle_fastest_label}<small className="table-subline">{formatMs(item.oracle_median_ms)}</small></td><td className={item.regret_percent > 0.01 ? 'warning-text mono' : 'success-text mono'}>{item.regret_percent.toFixed(2)}%</td><td className="mono">{item.baseline_regret_percent.toFixed(2)}%</td></tr>)}</tbody></table></div>
              <div className="result-note"><ShieldCheck size={15} /><span>{selectorResult.interpretation}</span></div><button className="secondary-button selector-reexport" onClick={exportSweepJSON}><Download size={15} /> Export results and selector evaluation</button>
            </>}</section>
          </>}
        </div>}

        {tab === 'methodology' && <div className="method-layout">
          <section className="panel method-intro"><div className="panel-kicker">RESEARCH PROTOCOL / V2</div><h2>A result is only useful if it is reproducible.</h2><p>VoxelPool Explorer separates workload generation, implementation, correctness, timing and interpretation. It is designed to make weak or workload-specific results visible rather than hide them.</p><div className="method-pills"><span><ShieldCheck size={14} /> Correctness first</span><span><RefreshCw size={14} /> Seeded workloads</span><span><Activity size={14} /> Repeated timing</span></div></section>
          <div className="method-cards">
            <section className="panel method-card"><div className="method-number">01</div><div><h3>Generate deterministic input</h3><p>Feature vectors use float32 values generated from a fixed seed. Destination IDs are generated according to uniform, hotspot, or unique patterns. The full configuration is included in each result.</p></div></section>
            <section className="panel method-card"><div className="method-number">02</div><div><h3>Validate against the reference</h3><p>The sequential reference computes the expected output. Each candidate is compared with <code>rtol=1e-4</code> and <code>atol=1e-4</code>. Timing is withheld if the correctness check fails.</p></div></section>
            <section className="panel method-card"><div className="method-number">03</div><div><h3>Measure the complete strategy</h3><p>Each strategy is warmed up once, then measured in a seeded pseudorandom interleaving order. Sorting and segmented reduction include preprocessing cost. Median, minimum, maximum, p95, raw samples and timing variation are preserved.</p></div></section>
            <section className="panel method-card"><div className="method-number">04</div><div><h3>Interpret without overclaiming</h3><p>Results describe this local CPU and software environment. They do not prove behavior on GPUs or Quadric Chimera. A future target backend must have its own verified reference, profiler and measurement contract.</p></div></section>
          </div>
          <section className="panel">
            <div className="panel-head"><div><div className="panel-kicker">RESEARCH LINEAGE</div><h2>What this project reuses</h2></div><ExternalLink size={17} className="muted-icon" /></div>
            <div className="reference-list">
              <a href="https://github.com/ScalingIntelligence/KernelBench" target="_blank" rel="noreferrer"><div><strong>KernelBench</strong><span>Correctness-gated reference vs. candidate evaluation and repeated timing.</span></div><ExternalLink size={14} /></a>
              <a href="https://github.com/KernelTuner/kernel_tuner" target="_blank" rel="noreferrer"><div><strong>Kernel Tuner</strong><span>Experiment configurations, parameter search and output verification.</span></div><ExternalLink size={14} /></a>
              <a href="https://github.com/meta-pytorch/tritonbench" target="_blank" rel="noreferrer"><div><strong>TritonBench</strong><span>Reusable operator cases and common benchmark runner patterns.</span></div><ExternalLink size={14} /></a>
              <a href="https://github.com/albertozeni/starlight" target="_blank" rel="noreferrer"><div><strong>Starlight</strong><span>Profiling-led analysis and bottleneck explanation.</span></div><ExternalLink size={14} /></a>
              <a href="https://app.quadric.ai/docs/latest/chimera-software-user-guide/tutorials-model-demos/custom-op-tutorials/index" target="_blank" rel="noreferrer"><div><strong>Quadric custom-op tutorials</strong><span>Required follow-up for a genuine Quadric-specific backend.</span></div><ExternalLink size={14} /></a>
            </div>
          </section>
          <div className="limitation-card"><CircleHelp size={19} /><div><strong>Current limitations</strong><p>Only three portable CPU strategies are included. Browser-local history and sweep records survive refreshes in this browser; the API job cache itself is volatile. Export the research bundle/CSV/Markdown report for portable backups. The selector is a preliminary nearest-neighbour baseline, not a mature autotuner, and no Quadric CCL implementation is included.</p></div></div>
        </div>}

        <footer className="footer"><div>VOXELPOOL EXPLORER <span>·</span> RESEARCH PROTOTYPE</div><div>Built for reproducible experiments <span>·</span> CPU results only</div></footer>
      </div>
    </main>
  </div>
}

export default App
