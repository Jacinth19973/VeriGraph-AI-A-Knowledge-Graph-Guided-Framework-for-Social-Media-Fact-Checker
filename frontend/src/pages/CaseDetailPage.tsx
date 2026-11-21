import type { ReactNode } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Shield, FileText, Layers, Users, GitBranch, Network, MessageCircle,
  ClipboardList, ExternalLink, AlertTriangle,
  RefreshCw, ArrowLeft
} from 'lucide-react'
import { casesApi, getApiError } from '../services/api'
import type { CaseDetail, EvidenceItem, NarrativeCluster, SpecialistAgent } from '../types'

type Tab = 'overview' | 'evidence' | 'narratives' | 'specialists' | 'debate' | 'kg' | 'explanation' | 'audit'

const tabs: Array<{ id: Tab; label: string; icon: typeof Shield }> = [
  { id: 'overview', label: 'Overview', icon: Shield },
  { id: 'evidence', label: 'Evidence', icon: FileText },
  { id: 'narratives', label: 'Narratives', icon: Layers },
  { id: 'specialists', label: 'Specialists', icon: Users },
  { id: 'debate', label: 'Debate', icon: GitBranch },
  { id: 'kg', label: 'Knowledge Graph', icon: Network },
  { id: 'explanation', label: 'Explanation', icon: MessageCircle },
  { id: 'audit', label: 'Audit', icon: ClipboardList },
]

function verdictClass(verdict: string) {
  const v = verdict?.toLowerCase()
  return v === 'true' ? 'badge-true' : v === 'false' ? 'badge-false' : v === 'misleading' ? 'badge-misleading' : 'badge-insufficient'
}
function display(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
function pct(value: unknown) {
  return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—'
}
function safeArray<T = any>(value: unknown): T[] {
  return Array.isArray(value) ? value : []
}

export function CaseDetailPage() {
  const { caseId } = useParams<{ caseId: string }>()
  const navigate = useNavigate()
  const [data, setData] = useState<CaseDetail | null>(null)
  const [tab, setTab] = useState<Tab>('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const retryTimer = useRef<number | undefined>(undefined)
  const [progress, setProgress] = useState<{ state: string; stage?: string; message?: string } | null>(null)

  const load = async () => {
    if (!caseId) return
    setLoading(true); setError(null)
    try {
      const res: any = await casesApi.get(caseId)
      if (res && res.state && !res.judge) {          // not finished yet: show progress instead of an error
        setProgress({ state: res.state, stage: res.stage, message: res.message })
        if (res.state === 'running') retryTimer.current = window.setTimeout(() => void load(), 8000)
        else setError(res.message || 'This case did not finish.')
      } else {
        setProgress(null)
        setData(res)
      }
    }
    catch (err) {
      const message = getApiError(err, 'Unable to load this verification case.')
      setError(message)
      // A run that is still in progress opens by itself as soon as it finishes.
      if (message.includes('still running')) retryTimer.current = window.setTimeout(() => void load(), 10000)
    }
    finally { setLoading(false) }
  }

  useEffect(() => {
    void load()
    return () => { if (retryTimer.current) window.clearTimeout(retryTimer.current) }
  }, [caseId])

  const evidence = useMemo<EvidenceItem[]>(() => {
    const fromRetrieval = safeArray<EvidenceItem>(data?.retrieval?.fused_evidence)
    if (fromRetrieval.length) return fromRetrieval
    return safeArray<EvidenceItem>(data?.narratives?.fused_evidence)
  }, [data])

  if (!caseId) return <PageError message="No case ID was supplied." onBack={() => navigate('/history')} />
  if (progress?.state === 'running' && !data) {
    return <RunningCase caseId={caseId} stage={progress.stage} onBack={() => navigate('/history')} />
  }
  if (loading && !data) return <PageState text="Loading verification case..." />
  if (error && !data) return <PageError message={error} onBack={() => navigate('/history')} retry={() => void load()} />
  if (!data) return <PageError message="The case record is empty." onBack={() => navigate('/history')} />

  const verdict = display(data.judge?.final_verdict || (data as any).verdict || 'INSUFFICIENT')
  const confidence = data.judge?.confidence ?? (data as any).confidence
  const clusters = safeArray<NarrativeCluster>(data.narratives?.clusters)
  const specialists = data.specialists || {}
  const kg = data.kg || ((data as any).knowledge_graph as any) || {}
  const kgContext = kg.kg_context || (data as any).kg_context || {}
  const explanation = data.explanation || {}
  const debate = data.debate || {}
  const audit = safeArray(data.audit)

  return (
    <div className="min-h-screen bg-gray-50 py-6 px-4">
      <div className="max-w-7xl mx-auto">
        <button className="text-sm text-primary-600 hover:underline flex items-center gap-2 mb-5" onClick={() => navigate('/history')}>
          <ArrowLeft className="w-4 h-4" /> Back to history
        </button>

        <section className="card mb-6">
          <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-5">
            <div className="min-w-0">
              <div className="flex flex-wrap gap-2 items-center mb-3">
                <span className={`badge ${verdictClass(verdict)}`}>{verdict}</span>
                <span className="text-xs font-mono text-gray-500">{data.case_id}</span>
              </div>
              <h1 className="text-2xl font-bold text-gray-900 break-words">{data.claim}</h1>
              <p className="text-sm text-gray-500 mt-2">Created {data.created_at ? new Intl.DateTimeFormat("en-IN", {timeZone: "Asia/Kolkata", hour12: false}).format(new Date(data.created_at)) + " IST" : '—'}</p>
            </div>
            <div className="flex gap-5 text-sm flex-shrink-0">
              <Stat label="Confidence" value={pct(confidence)} />
              <Stat label="Evidence" value={String(evidence.length || data.retrieval?.total_evidence || 0)} />
              <Stat label="Narratives" value={String(clusters.length)} />
            </div>
          </div>
        </section>

        {data.judge?.judge_status && !['success', 'pending'].includes(data.judge.judge_status) && (
          <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
            <strong>Judge status: {data.judge.judge_status}.</strong>{' '}
            {data.judge.judge_status === 'degraded'
              ? 'The Judge answered but cited no valid evidence IDs, so treat this verdict with caution.'
              : 'The language-model Judge did not complete, so this verdict comes from the rule-based fallback arbiter, not from the model.'}
            {data.judge.judge_error ? ` Detail: ${data.judge.judge_error}` : ''}
          </div>
        )}

        {(data as any).multimodal?.consistency === 'mismatch' && (
          <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
            <strong>The attached image does not match the claim.</strong>{' '}
            {((data as any).multimodal.reasons || []).join('; ')}
          </div>
        )}

        <div className="card p-0 overflow-hidden">
          <div className="flex overflow-x-auto border-b border-gray-200">
            {tabs.map(item => {
              const Icon = item.icon
              return <button key={item.id} type="button" onClick={() => setTab(item.id)} className={`flex-shrink-0 px-4 py-3 text-sm font-medium flex items-center gap-2 border-b-2 ${tab === item.id ? 'border-primary-600 text-primary-600' : 'border-transparent text-gray-500 hover:text-gray-800'}`}><Icon className="w-4 h-4" />{item.label}</button>
            })}
          </div>
          <div className="p-6">
            {tab === 'overview' && <Overview data={data} evidence={evidence} verdict={verdict} confidence={confidence} />}
            {tab === 'evidence' && <EvidencePanel evidence={evidence} />}
            {tab === 'narratives' && <NarrativesPanel clusters={clusters} />}
            {tab === 'specialists' && <SpecialistsPanel specialists={specialists} />}
            {tab === 'debate' && <DebatePanel debate={debate} />}
            {tab === 'kg' && <KGPanel kg={kg} context={kgContext} />}
            {tab === 'explanation' && <ExplanationPanel explanation={explanation} />}
            {tab === 'audit' && <AuditPanel audit={audit} metrics={data.processing_metrics || {}} />}
          </div>
        </div>
      </div>
    </div>
  )
}

function Overview({ data, evidence, verdict, confidence }: { data: CaseDetail; evidence: EvidenceItem[]; verdict: string; confidence: unknown }) {
  const input = data.input_metadata || {}
  const uncertainty = data.uncertainty || {}
  return (
    <div className="space-y-7">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="card bg-gray-50"><h2 className="font-semibold mb-3">Verification result</h2><span className={`badge ${verdictClass(verdict)}`}>{verdict}</span><p className="mt-3 text-sm text-gray-600">Judge confidence: <strong>{pct(confidence)}</strong>. This is a verification factor, not a calibrated probability.</p></div>
        <div className="card bg-gray-50"><h2 className="font-semibold mb-3">Input provenance</h2><p className="text-sm"><strong>Platform:</strong> {display(input.source_platform || input.platform || 'local')}</p><p className="text-sm mt-1 break-all"><strong>Source:</strong> {display(input.source_url || input.post_url)}</p></div>
        <div className="card bg-gray-50"><h2 className="font-semibold mb-3">Uncertainty</h2><p className="text-sm"><strong>Debate priority:</strong> {display(uncertainty.debate_priority)}</p><p className="text-sm mt-1"><strong>Weighted uncertainty:</strong> {pct(uncertainty.weighted_uncertainty_score)}</p></div>
      </div>

      {data.judge?.reasoning && <Section title="Judge reasoning"><p className="text-gray-700 whitespace-pre-wrap">{data.judge.reasoning}</p></Section>}

      <Section title="Pipeline coverage">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Mini label="Historical text" value={display(data.retrieval?.historical_text_count)} />
          <Mini label="Historical images" value={display(data.retrieval?.historical_image_count)} />
          <Mini label="Live evidence" value={display(data.retrieval?.live_count)} />
          <Mini label="KG used" value={data.kg?.kg_context?.neo4j_used ? 'Yes' : 'No / unavailable'} />
        </div>
      </Section>

      <Section title="Key evidence">
        {evidence.slice(0, 5).map(item => <EvidenceRow key={item.evidence_id} item={item} />)}
        {!evidence.length && <Empty text="No evidence records are available in this case response." />}
      </Section>
    </div>
  )
}

function EvidencePanel({ evidence }: { evidence: EvidenceItem[] }) {
  const [filter, setFilter] = useState('')
  const shown = evidence.filter(item => `${item.title || ''} ${item.domain || ''} ${item.snippet || ''}`.toLowerCase().includes(filter.toLowerCase()))
  return <div className="space-y-5"><div className="flex items-center gap-3"><input className="input-field" value={filter} onChange={(e: { target: { value: string } }) => setFilter(e.target.value)} placeholder="Filter evidence by title, domain or snippet..." /><span className="text-sm text-gray-500 whitespace-nowrap">{shown.length}/{evidence.length}</span></div>{shown.map(item => <EvidenceRow key={item.evidence_id} item={item} detailed />)}{!shown.length && <Empty text="No evidence matches this filter." />}</div>
}

function EvidenceRow({ item, detailed = false }: { item: EvidenceItem; detailed?: boolean }) {
  const url = item.url || item.display_url
  return <article className="border border-gray-200 rounded-lg p-4">
    <div className="flex flex-col md:flex-row md:justify-between gap-2">
      <div className="min-w-0"><p className="font-semibold text-gray-900">{item.title || item.display_text || 'Untitled evidence'}</p><p className="text-xs text-gray-500 mt-1">{item.domain || item.source || 'Provenance unavailable'} · {item.provider || item.retrieval_channel || '—'}</p></div>
      <div className="flex gap-2 text-xs"><span className="px-2 py-1 bg-gray-100 rounded">Relevance {pct(item.relevance_score)}</span><span className="px-2 py-1 bg-gray-100 rounded">Credibility {pct(item.credibility_score)}</span>{typeof item.fusion_score === 'number' && <span className="px-2 py-1 bg-primary-50 text-primary-700 rounded">Fusion {pct(item.fusion_score)}</span>}</div>
    </div>
    {item.snippet && <p className={`text-sm text-gray-700 mt-3 ${detailed ? '' : 'line-clamp-2'}`}>{item.snippet}</p>}
    <div className="mt-3 flex flex-wrap gap-3 text-xs text-gray-500"><span>{item.provenance_type || '—'}</span><span>{item.published_at || 'Date unavailable'}</span>{item.dataset && <span>Dataset: {item.dataset}</span>}{url && /^https?:\/\//i.test(url) && <a className="text-primary-600 hover:underline inline-flex gap-1 items-center" href={url} target="_blank" rel="noopener noreferrer">Open source <ExternalLink className="w-3 h-3" /></a>}</div>
  </article>
}

function NarrativesPanel({ clusters }: { clusters: NarrativeCluster[] }) {
  return <div className="space-y-4">{clusters.map(cluster => <article key={cluster.cluster_id} className="border border-gray-200 rounded-lg p-5"><div className="flex justify-between gap-3"><h3 className="font-semibold">Narrative {cluster.cluster_id}</h3><span className="text-sm text-gray-500">{cluster.size ?? cluster.evidence_ids?.length ?? 0} evidence items</span></div><p className="mt-2 text-gray-800">{cluster.theme || cluster.theme_keywords?.join(', ') || 'Theme unavailable'}</p>{cluster.theme_keywords?.length ? <div className="flex flex-wrap gap-2 mt-3">{cluster.theme_keywords.map(k => <span key={k} className="px-2 py-1 bg-gray-100 rounded text-xs">{k}</span>)}</div> : null}<div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 text-sm"><Mini label="Coherence" value={pct(cluster.mean_cluster_coherence)} /><Mini label="Credibility" value={pct(cluster.mean_credibility)} /><Mini label="Relevance" value={pct(cluster.mean_relevance)} /><Mini label="Sources" value={display(cluster.source_count)} /></div>{cluster.conflict_summary && <p className="mt-4 text-sm text-amber-700">{cluster.conflict_summary}</p>}</article>)}{!clusters.length && <Empty text="No narrative clusters were returned for this case." />}</div>
}

function SpecialistsPanel({ specialists }: { specialists: Record<string, SpecialistAgent> }) {
  const entries = Object.entries(specialists).filter(([key, value]) => value && key !== 'specialist_summary' && key !== 'evidence_assessment_matrix')
  return <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">{entries.map(([name, agent]) => <article key={name} className="border border-gray-200 rounded-lg p-5"><div className="flex justify-between gap-3"><h3 className="font-semibold capitalize">{name .split('_').join(' ')}</h3><span className="text-sm text-gray-500">{pct(agent.confidence)}</span></div><p className="mt-2 font-medium">{display(agent.assessment || agent.overall_assessment || agent.status)}</p>{agent.key_findings?.length ? <ul className="mt-3 list-disc pl-5 text-sm text-gray-700 space-y-1">{agent.key_findings.slice(0, 6).map((x, i) => <li key={i}>{x}</li>)}</ul> : null}<div className="grid grid-cols-2 gap-3 mt-4 text-sm"><Mini label="Evidence coverage" value={pct(agent.evidence_coverage)} /><Mini label="Retrieval support" value={pct(agent.retrieval_support)} /></div></article>)}{!entries.length && <Empty text="No specialist results are available." />}</div>
}

function DebatePanel({ debate }: { debate: Record<string, any> }) {
  const turns = safeArray<any>(debate.turns || debate.debate_history)
  return <div className="space-y-5"><div className="grid grid-cols-2 md:grid-cols-4 gap-3"><Mini label="Max rounds" value={display(debate.max_rounds)} /><Mini label="Executed" value={display(debate.actual_rounds || turns.length)} /><Mini label="Stop reason" value={display(debate.stop_reason)} /><Mini label="Final confidence" value={pct(debate.final_confidence)} /></div>{turns.map((turn, i) => <article key={i} className="border border-gray-200 rounded-lg p-4"><div className="flex justify-between"><strong>{display(turn.speaker)}</strong><span className="text-sm text-gray-500">Round {display(turn.round || i + 1)} · {pct(turn.confidence)}</span></div><p className="mt-2 text-sm text-gray-700">{display(turn.key_summary || turn.position)}</p>{turn.key_evidence_summary && <p className="mt-2 text-xs text-gray-500">Evidence: {turn.key_evidence_summary}</p>}</article>)}{debate.judge && <Section title="Judge summary"><p className="text-sm whitespace-pre-wrap">{display(debate.judge.reasoning || debate.judge.summary)}</p></Section>}{!turns.length && <Empty text="No structured debate turns were stored." />}</div>
}

function KGPanel({ kg, context }: { kg: Record<string, any>; context: Record<string, any> }) {
  const findings = safeArray<string>(kg.kg_findings)
  const gaps = safeArray<string>(kg.kg_gaps)
  const candidates = safeArray<any>(kg.expansion_candidates || kg.kg_expansion?.candidates)
  const listFields = ['entities','persons','organizations','locations','events','dates','sources']
  return <div className="space-y-6"><div className="grid grid-cols-2 md:grid-cols-4 gap-3"><Mini label="Status" value={display(context.status)} /><Mini label="Neo4j used" value={context.neo4j_used ? 'Yes' : 'No'} /><Mini label="Nodes returned" value={display(context.nodes_returned)} /><Mini label="Relationships" value={display(context.relationships_returned)} /></div><div className="grid grid-cols-1 md:grid-cols-2 gap-5">{listFields.map(field => <div className="border rounded-lg p-4" key={field}><h3 className="font-semibold capitalize mb-2">{field}</h3>{safeArray<string>(context[field]).length ? <ul className="text-sm text-gray-700 list-disc pl-5 space-y-1">{safeArray<string>(context[field]).slice(0, 20).map((x,i)=><li key={i}>{x}</li>)}</ul> : <p className="text-sm text-gray-500">No data returned.</p>}</div>)}</div><Section title="KG findings">{findings.length ? findings.map((x,i)=><p key={i} className="text-sm mb-2">{x}</p>) : <Empty text="No KG findings were returned for this case." />}</Section><Section title="KG gaps">{gaps.length ? gaps.map((x,i)=><p key={i} className="text-sm mb-2">{x}</p>) : <Empty text="No unresolved KG gaps were identified for this case." />}</Section><Section title="Expansion candidates">{candidates.length ? candidates.map((x,i)=><pre key={i} className="text-xs bg-gray-50 rounded p-3 overflow-auto">{JSON.stringify(x,null,2)}</pre>) : <Empty text="No expansion candidates were returned." />}</Section></div>
}

function ExplanationPanel({ explanation }: { explanation: Record<string, any> }) {
  const factors = safeArray<any>(explanation.verification_factors)
  const sections: Array<[string, unknown]> = [
    ['Summary', explanation.summary || explanation.natural_language_summary],
    ['Verdict reason', explanation.verdict_reason],
    ['Temporal findings', explanation.temporal_findings],
    ['Location findings', explanation.location_findings],
    ['Source findings', explanation.source_findings],
    ['KG findings', explanation.kg_findings],
    ['Debate findings', explanation.debate_findings],
    ['Knowledge gaps', explanation.knowledge_gaps],
  ]
  return <div className="space-y-6"><Section title="Human-readable explanation"><p className="whitespace-pre-wrap text-gray-700">{display(explanation.natural_language_summary || explanation.summary || 'No natural-language explanation was returned.')}</p></Section><Section title="Verification factors"><div className="grid grid-cols-1 md:grid-cols-2 gap-4">{factors.map((factor,i)=><div key={i} className="border rounded-lg p-4"><div className="flex justify-between"><strong>{display(factor.factor)}</strong><span>{pct(factor.confidence)}</span></div><p className="text-sm mt-2">{display(factor.detail || factor.assessment)}</p></div>)}{!factors.length && <Empty text="No verification-factor records were returned." />}</div></Section>{sections.slice(2).map(([title,value])=><Section key={title} title={title}>{safeArray(value).length ? safeArray(value).map((x,i)=><p className="text-sm mb-2" key={i}>{display(x)}</p>) : <Empty text="No records returned." />}</Section>)}</div>
}

function AuditPanel({ audit, metrics }: { audit: any[]; metrics: Record<string, any> }) {
  return <div className="space-y-6"><Section title="Processing metrics"><pre className="text-xs bg-gray-50 rounded-lg p-4 overflow-auto">{JSON.stringify(metrics,null,2)}</pre></Section><Section title="Pipeline stages">{audit.length ? audit.map((stage,i)=><div key={i} className="border-b py-3 flex flex-col md:flex-row md:items-center gap-2 md:justify-between"><div><strong>{display(stage.stage)}</strong><p className="text-sm text-gray-500">{display(stage.message)}</p>{stage.error && <p className="text-sm text-red-600">{stage.error}</p>}</div><div className="text-sm text-gray-500">{display(stage.status)} · {display(stage.duration_seconds)}s</div></div>) : <Empty text="No audit stages were returned." />}</Section></div>
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section><h2 className="text-lg font-semibold mb-3">{title}</h2>{children}</section>
}
function Stat({ label, value }: { label: string; value: string }) { return <div><p className="text-xs text-gray-500">{label}</p><p className="font-semibold mt-1">{value}</p></div> }
function Mini({ label, value }: { label: string; value: string }) { return <div className="bg-gray-50 rounded-lg p-3"><p className="text-xs text-gray-500">{label}</p><p className="font-semibold mt-1 break-words">{value}</p></div> }
function Empty({ text }: { text: string }) { return <p className="text-sm text-gray-500">{text}</p> }
function PageState({ text }: { text: string }) { return <div className="min-h-screen flex items-center justify-center bg-gray-50"><div className="text-gray-500">{text}</div></div> }
function PageError({ message, onBack, retry }: { message: string; onBack: () => void; retry?: () => void }) { return <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4"><div className="card max-w-lg w-full text-center"><AlertTriangle className="w-10 h-10 text-amber-500 mx-auto mb-3" /><p className="text-gray-700">{message}</p><div className="flex justify-center gap-3 mt-5"><button className="btn-secondary" onClick={onBack}>Back</button>{retry && <button className="btn-primary flex items-center gap-2" onClick={retry}><RefreshCw className="w-4 h-4" />Retry</button>}</div></div></div> }

function RunningCase({ caseId, stage, onBack }: { caseId?: string; stage?: string; onBack: () => void }) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="card max-w-lg w-full text-center">
        <RefreshCw className="w-10 h-10 text-primary-600 mx-auto mb-3 animate-spin" />
        <h2 className="text-lg font-semibold text-gray-900">Verification in progress</h2>
        <p className="text-gray-600 mt-2">
          Case {caseId} is being analysed (current stage: <strong>{stage || 'starting'}</strong>).
        </p>
        <p className="text-sm text-gray-500 mt-2">
          This page updates by itself and opens the result when the run finishes. On a CPU a run can take 15-20 minutes.
        </p>
        <button className="btn-secondary mt-5" onClick={onBack}>Back to history</button>
      </div>
    </div>
  )
}
