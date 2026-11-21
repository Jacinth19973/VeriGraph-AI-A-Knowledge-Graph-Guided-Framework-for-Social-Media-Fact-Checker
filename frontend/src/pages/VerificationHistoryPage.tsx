import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ChevronLeft, ChevronRight, Download, RefreshCw, Search } from 'lucide-react'
import { casesApi, getApiError } from '../services/api'
import type { CaseSummary } from '../types'

function badge(verdict: string) {
  const v = verdict?.toLowerCase()
  return v === 'true' ? 'badge-true' : v === 'false' ? 'badge-false' : v === 'misleading' ? 'badge-misleading' : 'badge-insufficient'
}

export function VerificationHistoryPage() {
  const [cases, setCases] = useState<CaseSummary[]>([])
  const [page, setPage] = useState(1)
  const pageSize = 20
  const [total, setTotal] = useState(0)
  const [search, setSearch] = useState('')
  const [verdict, setVerdict] = useState('')
  const [platform, setPlatform] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await casesApi.list({
        page, page_size: pageSize,
        search: search.trim() || undefined,
        verdict: verdict || undefined,
        platform: platform || undefined,
      })
      setCases(result.cases ?? [])
      setTotal(result.total ?? 0)
    } catch (err) {
      setError(getApiError(err, 'Unable to load verification history.'))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [page, verdict, platform])

  const onSearch = (event: any) => {
    event.preventDefault()
    setPage(1)
    void load()
  }

  const exportCsv = () => {
    const headers = ['case_id', 'claim', 'verdict', 'confidence', 'platform', 'created_at', 'evidence']
    const rows = cases.map(item => [
      item.case_id, item.claim, item.verdict, item.confidence,
      item.source_platform || 'local', item.created_at, item.total_evidence ?? '',
    ])
    const csv = [headers, ...rows].map(row => row.map(value => `"${String(value ?? '').split('"').join('""')}"`).join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'verigraph-verification-history.csv'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-end gap-4 mb-7">
          <div>
            <h1 className="text-3xl font-bold">Verification History</h1>
            <p className="text-gray-600 mt-1">Browse completed verification cases stored by the backend.</p>
          </div>
          <div className="flex gap-2">
            <button type="button" className="btn-secondary flex items-center gap-2" onClick={() => void load()} disabled={loading}><RefreshCw className="w-4 h-4" />Refresh</button>
            <button type="button" className="btn-secondary flex items-center gap-2" onClick={exportCsv} disabled={!cases.length}><Download className="w-4 h-4" />Export current page</button>
          </div>
        </div>

        <form onSubmit={onSearch} className="card mb-6 flex flex-col md:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input className="input-field pl-10" value={search} onChange={(e: { target: { value: string } }) => setSearch(e.target.value)} placeholder="Search claims..." />
          </div>
          <select className="input-field md:w-48" value={verdict} onChange={(e: { target: { value: string } }) => { setVerdict(e.target.value); setPage(1) }}>
            <option value="">All verdicts</option><option value="True">True</option><option value="False">False</option><option value="Misleading">Misleading</option><option value="INSUFFICIENT">Insufficient</option>
          </select>
          <select className="input-field md:w-48" value={platform} onChange={(e: { target: { value: string } }) => { setPlatform(e.target.value); setPage(1) }}>
            <option value="">All platforms</option><option value="local">Local</option><option value="instagram">Instagram</option><option value="facebook">Facebook</option><option value="x">X</option><option value="web">Web</option>
          </select>
          <button className="btn-primary" type="submit">Search</button>
        </form>

        {error && <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">{error}</div>}

        <div className="card overflow-hidden p-0">
          {loading ? (
            <div className="py-16 text-center text-gray-500">Loading cases...</div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 border-b border-gray-200 text-xs uppercase text-gray-500">
                    <tr>{['Case ID', 'Claim', 'Verdict', 'Confidence', 'Platform', 'Created', 'Evidence', ''].map(h => <th key={h} className="px-5 py-3 text-left">{h}</th>)}</tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {cases.length === 0 ? (
                      <tr><td colSpan={8} className="px-5 py-12 text-center text-gray-500">No cases match the current filters.</td></tr>
                    ) : cases.map(item => (
                      <tr key={item.case_id} className="hover:bg-gray-50">
                        <td className="px-5 py-4 font-mono text-xs">{item.case_id}</td>
                        <td className="px-5 py-4 max-w-sm truncate" title={item.claim}>{item.claim || '—'}</td>
                        <td className="px-5 py-4"><span className={`badge ${badge(item.verdict)}`}>{item.verdict || 'Unknown'}</span></td>
                        <td className="px-5 py-4">{typeof item.confidence === 'number' ? `${(item.confidence * 100).toFixed(1)}%` : '—'}</td>
                        <td className="px-5 py-4 capitalize">{item.source_platform || 'local'}</td>
                        <td className="px-5 py-4 text-gray-500">{item.created_at ? new Intl.DateTimeFormat("en-IN", {timeZone: "Asia/Kolkata", hour12: false}).format(new Date(item.created_at)) + " IST" : '—'}</td>
                        <td className="px-5 py-4">{item.total_evidence ?? '—'}</td>
                        <td className="px-5 py-4"><Link className="text-primary-600 hover:underline font-medium" to={`/case/${item.case_id}`}>Open</Link></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="border-t border-gray-200 px-5 py-4 flex items-center justify-between">
                <span className="text-sm text-gray-500">{total ? `${(page - 1) * pageSize + 1}–${Math.min(page * pageSize, total)} of ${total}` : '0 cases'}</span>
                <div className="flex gap-2">
                  <button className="btn-secondary p-2" disabled={page <= 1 || loading} onClick={() => setPage((value: number) => Math.max(1, value - 1))}><ChevronLeft className="w-4 h-4" /></button>
                  <button className="btn-secondary p-2" disabled={page * pageSize >= total || loading} onClick={() => setPage((value: number) => value + 1)}><ChevronRight className="w-4 h-4" /></button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
