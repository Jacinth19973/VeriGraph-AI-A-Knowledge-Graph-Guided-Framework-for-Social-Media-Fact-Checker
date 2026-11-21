import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'
import { Activity, Brain, Clock, Network, RefreshCw, Users } from 'lucide-react'
import { analyticsApi, getApiError, healthApi } from '../services/api'
import type { AnalyticsResponse, SystemHealthResponse } from '../types'

export function SystemAnalyticsPage() {
  const [data, setData] = useState<AnalyticsResponse | null>(null)
  const [health, setHealth] = useState<SystemHealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true); setError(null)
    try {
      const [analytics, systemHealth] = await Promise.all([analyticsApi.overview(), healthApi.check()])
      setData(analytics); setHealth(systemHealth)
    } catch (err) {
      setError(getApiError(err, 'Unable to load analytics.'))
    } finally { setLoading(false) }
  }

  useEffect(() => { void load() }, [])

  if (loading && !data) return <PageState text="Loading analytics..." />
  if (!data) return <PageState text={error || 'No analytics data available.'} error />

  const verdicts = Object.entries(data.verdict_distribution || {})
  const platforms = Object.entries(data.platform_distribution || {})
  const sources = Object.entries(data.source_distribution || {}).sort((a, b) => b[1] - a[1])

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-end justify-between mb-7">
          <div><h1 className="text-3xl font-bold">System Analytics</h1><p className="text-gray-600 mt-1">Backend-derived verification metrics. No frontend estimates are generated.</p></div>
          <button className="btn-secondary flex gap-2 items-center" onClick={() => void load()} disabled={loading}><RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />Refresh</button>
        </div>
        {error && <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">{error}</div>}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-7">
          <Metric icon={<Users />} value={data.total_cases} label="Total cases" />
          <Metric icon={<Clock />} value={`${num(data.average_processing_time)}s`} label="Average processing time" />
          <Metric icon={<Brain />} value={`${pct(data.agent_agreement_rate)}`} label="Agent agreement" />
          <Metric icon={<Network />} value={num(data.average_evidence_count)} label="Average evidence/case" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-7">
          <DistributionCard title="Verdict distribution" entries={verdicts} total={data.total_cases} />
          <DistributionCard title="Platform distribution" entries={platforms} total={data.total_cases} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-7">
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Source distribution</h2>
            <div className="space-y-3">
              {sources.length ? sources.slice(0, 12).map(([name, count]) => <BarRow key={name} name={name} value={count} max={Math.max(...sources.map(x => x[1]), 1)} />) : <Empty />}
            </div>
          </div>
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Verdict by platform</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm"><thead><tr className="border-b text-xs uppercase text-gray-500"><th className="text-left py-2">Platform</th>{['True','False','Misleading','INSUFFICIENT'].map(v => <th key={v} className="text-right py-2">{v}</th>)}</tr></thead>
                <tbody>{Object.entries(data.verdict_by_platform || {}).map(([platform, counts]) => <tr key={platform} className="border-b border-gray-100"><td className="py-2 capitalize">{platform}</td>{['True','False','Misleading','INSUFFICIENT'].map(v => <td key={v} className="py-2 text-right">{counts[v] || 0}</td>)}</tr>)}</tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <SimpleStat title="Average debate rounds" value={num(data.average_debate_rounds)} />
          <SimpleStat title="Average narratives/case" value={num(data.average_narrative_count)} />
          <div className="card"><div className="flex items-center gap-2 mb-3"><Activity className="w-5 h-5 text-primary-600" /><h2 className="font-semibold">System health</h2></div>{health?.services ? Object.entries(health.services).map(([key, service]) => <div key={key} className="flex justify-between py-1 text-sm"><span className="text-gray-600">{service.name}</span><span className="capitalize">{service.status}</span></div>) : <Empty />}</div>
        </div>
      </div>
    </div>
  )
}

function Metric({ icon, value, label }: { icon: ReactNode; value: ReactNode; label: string }) {
  return <div className="card flex items-center gap-4"><div className="p-3 bg-primary-50 rounded-xl text-primary-600">{icon}</div><div><div className="text-2xl font-bold">{value}</div><div className="text-sm text-gray-500">{label}</div></div></div>
}
function DistributionCard({ title, entries, total }: { title: string; entries: [string, number][]; total: number }) {
  return <div className="card"><h2 className="text-lg font-semibold mb-4">{title}</h2><div className="space-y-4">{entries.length ? entries.map(([name, value]) => <BarRow key={name} name={name} value={value} max={Math.max(total,1)} suffix={total ? `${((value/total)*100).toFixed(1)}%` : '0%'} />) : <Empty />}</div></div>
}
function BarRow({ name, value, max, suffix }: { name: string; value: number; max: number; suffix?: string }) {
  return <div><div className="flex justify-between text-sm mb-1"><span className="truncate">{name}</span><span className="text-gray-500">{value}{suffix ? ` (${suffix})` : ''}</span></div><div className="h-2 bg-gray-100 rounded-full"><div className="h-2 bg-primary-500 rounded-full" style={{width:`${Math.min(100,(value/max)*100)}%`}} /></div></div>
}
function SimpleStat({ title, value }: { title: string; value: string }) { return <div className="card"><p className="text-sm text-gray-500">{title}</p><p className="text-2xl font-bold mt-2">{value}</p></div> }
function Empty() { return <p className="text-sm text-gray-500">No data available.</p> }
function num(value: number | undefined) { return typeof value === 'number' ? value.toFixed(1) : '—' }
function pct(value: number | undefined) { return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—' }
function PageState({ text, error = false }: { text: string; error?: boolean }) { return <div className="min-h-screen flex items-center justify-center bg-gray-50"><p className={error ? 'text-red-600' : 'text-gray-500'}>{text}</p></div> }
