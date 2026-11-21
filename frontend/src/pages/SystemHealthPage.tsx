import { useEffect, useState } from 'react'
import { RefreshCw, Server, Database, Cpu, Globe, Shield, Brain, Network, CheckCircle, XCircle, AlertTriangle, Loader2 } from 'lucide-react'
import { getApiError, healthApi } from '../services/api'
import type { SystemHealthResponse } from '../types'

function statusClass(status: string) {
  if (status === 'available') return 'bg-green-100 text-green-800'
  if (status === 'degraded') return 'bg-amber-100 text-amber-800'
  if (status === 'unavailable') return 'bg-red-100 text-red-800'
  return 'bg-gray-100 text-gray-700'
}

function serviceIcon(name: string) {
  const n = name.toLowerCase()
  if (n.includes('ollama') || n.includes('llm')) return <Brain className="w-5 h-5" />
  if (n.includes('neo4j') || n.includes('database')) return <Database className="w-5 h-5" />
  if (n.includes('faiss') || n.includes('vector')) return <Cpu className="w-5 h-5" />
  if (n.includes('search') || n.includes('news') || n.includes('api')) return <Globe className="w-5 h-5" />
  if (n.includes('social')) return <Network className="w-5 h-5" />
  if (n.includes('verify')) return <Shield className="w-5 h-5" />
  return <Server className="w-5 h-5" />
}

function statusIcon(status: string) {
  if (status === 'available') return <CheckCircle className="w-5 h-5 text-green-600" />
  if (status === 'degraded') return <AlertTriangle className="w-5 h-5 text-amber-600" />
  if (status === 'unavailable') return <XCircle className="w-5 h-5 text-red-600" />
  return <span className="text-gray-400">?</span>
}

export function SystemHealthPage() {
  const [health, setHealth] = useState<SystemHealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)

  const load = async () => {
    setLoading(true); setError(null)
    try { setHealth(await healthApi.check()) }
    catch (err) { setError(getApiError(err, 'Unable to reach the backend health endpoint.')) }
    finally { setLoading(false) }
  }

  useEffect(() => {
    void load()
    if (!autoRefresh) return
    const timer = window.setInterval(() => void load(), 30000)
    return () => window.clearInterval(timer)
  }, [autoRefresh])

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-7">
          <div><h1 className="text-3xl font-bold">System Health</h1><p className="text-gray-600 mt-1">Live status reported by the VeriGraph backend.</p></div>
          <div className="flex items-center gap-4">
            <label className="text-sm text-gray-600 flex items-center gap-2"><input type="checkbox" checked={autoRefresh} onChange={(e: { target: { checked: boolean } }) => setAutoRefresh(e.target.checked)} /> Auto-refresh</label>
            <button className="btn-secondary flex items-center gap-2" onClick={() => void load()} disabled={loading}><RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />Refresh</button>
          </div>
        </div>

        {error && <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">{error}</div>}

        {loading && !health ? (
          <div className="card text-center py-16"><Loader2 className="w-8 h-8 animate-spin mx-auto text-primary-600" /><p className="mt-3 text-gray-500">Checking backend services...</p></div>
        ) : health ? (
          <>
            <div className="card mb-7 flex items-center justify-between">
              <div className="flex items-center gap-4"><div className="p-3 bg-primary-50 rounded-xl"><Server className="w-6 h-6 text-primary-600" /></div><div><h2 className="font-semibold text-lg">Overall System Status</h2><p className="text-sm text-gray-500">{health.timestamp || 'Timestamp unavailable'}</p></div></div>
              <span className={`px-4 py-2 rounded-full font-semibold text-sm ${health.status === 'healthy' ? 'bg-green-100 text-green-800' : 'bg-amber-100 text-amber-800'}`}>{health.status || 'unknown'}</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {Object.entries(health.services || {}).map(([key, service]) => (
                <div className="card" key={key}>
                  <div className="flex gap-4">
                    <div className="p-3 bg-gray-100 rounded-xl h-fit">{serviceIcon(service.name)}</div>
                    <div className="min-w-0 flex-1">
                      <div className="flex justify-between gap-2"><h3 className="font-semibold truncate">{service.name}</h3>{statusIcon(service.status)}</div>
                      <span className={`inline-block px-2 py-1 rounded-full text-xs mt-2 ${statusClass(service.status)}`}>{service.status}</span>
                      {service.details && <p className="text-sm text-gray-500 mt-2">{service.details}</p>}
                      {typeof service.latency_ms === 'number' && <p className="text-xs text-gray-500 mt-2">{service.latency_ms.toFixed(1)} ms</p>}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="card mt-7">
              <h2 className="text-lg font-semibold mb-4">Health Summary</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-5 text-sm">
                <div><p className="text-gray-500">Services</p><p className="font-semibold text-lg">{Object.keys(health.services || {}).length}</p></div>
                <div><p className="text-gray-500">Available</p><p className="font-semibold text-lg">{Object.values(health.services || {}).filter(s => s.status === 'available').length}</p></div>
                <div><p className="text-gray-500">Degraded</p><p className="font-semibold text-lg">{Object.values(health.services || {}).filter(s => s.status === 'degraded').length}</p></div>
                <div><p className="text-gray-500">Unavailable</p><p className="font-semibold text-lg">{Object.values(health.services || {}).filter(s => s.status === 'unavailable').length}</p></div>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  )
}
