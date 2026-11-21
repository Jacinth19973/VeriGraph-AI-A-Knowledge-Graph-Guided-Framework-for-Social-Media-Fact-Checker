import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Shield, Image, Globe, Folder, BarChart2, CheckCircle, Brain,
  TrendingUp, AlertCircle, RefreshCw
} from 'lucide-react'
import { casesApi, getApiError } from '../services/api'
import type { CaseSummary } from '../types'

function verdictClass(verdict: string): string {
  switch (verdict?.toLowerCase()) {
    case 'true': return 'badge-true'
    case 'false': return 'badge-false'
    case 'misleading': return 'badge-misleading'
    default: return 'badge-insufficient'
  }
}

export function HomePage() {
  const navigate = useNavigate()
  const [cases, setCases] = useState<CaseSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await casesApi.list({ page: 1, page_size: 10 })
      setCases(response.cases ?? [])
    } catch (err) {
      setError(getApiError(err, 'Unable to load recent verification cases.'))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [])

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <section className="card mb-8 bg-gradient-to-r from-primary-50 to-white">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
            <div>
              <div className="flex items-center gap-3 mb-3">
                <Shield className="w-9 h-9 text-primary-600" />
                <span className="text-sm font-semibold uppercase tracking-wide text-primary-700">VeriGraph AI</span>
              </div>
              <h1 className="text-3xl md:text-4xl font-bold text-gray-900">Knowledge-Guided Multimodal Fact Verification</h1>
              <p className="text-gray-600 mt-3 max-w-3xl">
                Verify an image and claim using multimodal retrieval, evidence fusion, Knowledge Graph context,
                specialist verification agents, adaptive debate and explainable final reasoning.
              </p>
            </div>
            <button className="btn-primary whitespace-nowrap" onClick={() => navigate('/verify/local')}>
              Start Verification
            </button>
          </div>
        </section>

        {error && (
          <div className="mb-6 p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span className="flex-1">{error}</span>
            <button type="button" onClick={() => void load()} className="text-sm font-medium hover:underline">
              Retry
            </button>
          </div>
        )}

        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
          <Metric icon={<Folder className="w-6 h-6" />} value={cases.length} label="Recent cases loaded" />
          <Metric icon={<CheckCircle className="w-6 h-6" />} value="NB1–NB9" label="Verification pipeline" />
          <Metric icon={<Brain className="w-6 h-6" />} value="4" label="Specialist verification roles" />
          <Metric icon={<TrendingUp className="w-6 h-6" />} value="Adaptive" label="Debate controller" />
        </section>

        <section className="grid grid-cols-1 md:grid-cols-4 gap-5 mb-8">
          <Action title="Verify Local" text="Upload image + claim" icon={<Image />} onClick={() => navigate('/verify/local')} />
          <Action title="Verify Social" text="Fetch a public post" icon={<Globe />} onClick={() => navigate('/verify/social')} />
          <Action title="History" text="Browse completed cases" icon={<Folder />} onClick={() => navigate('/history')} />
          <Action title="Analytics" text="Review system metrics" icon={<BarChart2 />} onClick={() => navigate('/analytics')} />
        </section>

        <section className="card">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-xl font-semibold">Recent Verification Cases</h2>
              <p className="text-sm text-gray-500 mt-1">Data is loaded from the backend case store.</p>
            </div>
            <button type="button" onClick={() => void load()} className="btn-secondary flex items-center gap-2" disabled={loading}>
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
            </button>
          </div>

          {loading ? (
            <div className="py-12 text-center text-gray-500">Loading cases...</div>
          ) : cases.length === 0 ? (
            <div className="py-12 text-center text-gray-500">No verification cases have been stored yet.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-gray-200 text-left text-xs uppercase text-gray-500">
                  <tr>
                    <th className="px-4 py-3">Case</th>
                    <th className="px-4 py-3">Claim</th>
                    <th className="px-4 py-3">Verdict</th>
                    <th className="px-4 py-3">Confidence</th>
                    <th className="px-4 py-3">Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {cases.map(item => (
                    <tr key={item.case_id} className="hover:bg-gray-50 cursor-pointer" onClick={() => navigate(`/case/${item.case_id}`)}>
                      <td className="px-4 py-3 font-mono text-xs">{item.case_id}</td>
                      <td className="px-4 py-3 max-w-md truncate">{item.claim || '—'}</td>
                      <td className="px-4 py-3"><span className={`badge ${verdictClass(item.verdict)}`}>{item.verdict || 'Unknown'}</span></td>
                      <td className="px-4 py-3">{typeof item.confidence === 'number' ? `${(item.confidence * 100).toFixed(1)}%` : '—'}</td>
                      <td className="px-4 py-3 text-gray-500">{item.created_at ? new Intl.DateTimeFormat("en-IN", {timeZone: "Asia/Kolkata", hour12: false}).format(new Date(item.created_at)) + " IST" : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

function Metric({ icon, value, label }: { icon: ReactNode; value: ReactNode; label: string }) {
  return (
    <div className="card">
      <div className="flex items-center gap-4">
        <div className="p-3 bg-primary-50 rounded-xl text-primary-600">{icon}</div>
        <div>
          <div className="font-bold text-xl text-gray-900">{value}</div>
          <div className="text-sm text-gray-500">{label}</div>
        </div>
      </div>
    </div>
  )
}

function Action({ title, text, icon, onClick }: { title: string; text: string; icon: ReactNode; onClick: () => void }) {
  return (
    <button type="button" className="card text-left hover:shadow-md transition-shadow" onClick={onClick}>
      <div className="w-11 h-11 rounded-xl bg-primary-50 text-primary-600 flex items-center justify-center mb-4">{icon}</div>
      <h3 className="font-semibold text-gray-900">{title}</h3>
      <p className="text-sm text-gray-500 mt-1">{text}</p>
    </button>
  )
}
