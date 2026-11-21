import { Outlet, Link, useLocation } from 'react-router-dom'
import {
  Home, History, BarChart2, Activity, ChevronLeft, ChevronRight,
  Menu, X, Shield, Network, Image, Settings, BarChart3, FileText, Download, Check
} from 'lucide-react'
import { useState } from 'react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home, description: 'Overview & quick actions' },
  { name: 'Verify Local Content', href: '/verify/local', icon: Image, description: 'Image + text claim' },
  { name: 'Verify Social Media', href: '/verify/social', icon: Network, description: 'Public post extraction' },
  { name: 'Accuracy Benchmark', href: '/benchmark', icon: BarChart3, description: '100-case ablation study' },
  { name: 'Verification History', href: '/history', icon: History, description: 'Previous cases' },
  { name: 'System Analytics', href: '/analytics', icon: BarChart2, description: 'System statistics' },
  { name: 'System Health', href: '/health', icon: Activity, description: 'Service status' },
  { name: 'Research Paper (.tex)', href: '/paper', icon: FileText, description: 'IEEE LaTeX Conference Paper' },
]

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const location = useLocation()

  const closeMobile = () => {
    if (window.innerWidth < 1024) setSidebarOpen(false)
  }

  const handleDownloadZip = async () => {
    setDownloading(true)
    try {
      const res = await fetch('/VeriGraph_AI_Fixed_Files.zip')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'VeriGraph_AI_Fixed_Files.zip'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      setTimeout(() => window.URL.revokeObjectURL(url), 10000)
    } catch (err) {
      console.warn('Direct blob download failed, falling back:', err)
      window.location.href = '/VeriGraph_AI_Fixed_Files.zip'
    } finally {
      setTimeout(() => setDownloading(false), 1500)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close navigation"
          className="fixed inset-0 bg-black/40 z-40 lg:hidden cursor-default"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed lg:static inset-y-0 left-0 z-50 bg-white border-r border-gray-200
          transition-all duration-300 flex flex-col
          ${collapsed ? 'w-20' : 'w-72'}
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}
      >
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
          <Link to="/" onClick={closeMobile} className="flex items-center gap-2 min-w-0">
            <div className="w-9 h-9 bg-primary-600 rounded-lg flex items-center justify-center flex-shrink-0">
              <Shield className="w-5 h-5 text-white" />
            </div>
            {!collapsed && <span className="font-semibold text-xl text-gray-900 truncate">VeriGraph AI</span>}
          </Link>
          <button
            type="button"
            onClick={() => setCollapsed((value: boolean) => !value)}
            className="p-2 rounded-lg hover:bg-gray-100 text-gray-500"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
          </button>
        </div>

        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto" aria-label="Main navigation">
          {navigation.map(item => {
            const active = item.href === '/'
              ? location.pathname === '/'
              : location.pathname === item.href || location.pathname.startsWith(`${item.href}/`)
            const Icon = item.icon

            return (
              <Link
                key={item.href}
                to={item.href}
                onClick={closeMobile}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                  active
                    ? 'bg-primary-50 text-primary-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                } ${collapsed ? 'justify-center' : ''}`}
                title={collapsed ? item.name : undefined}
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                {!collapsed && (
                  <div className="min-w-0">
                    <p className="truncate">{item.name}</p>
                    <p className="truncate text-xs text-gray-500">{item.description}</p>
                  </div>
                )}
              </Link>
            )
          })}
        </nav>

        <div className="p-4 border-t border-gray-200">
          {!collapsed ? (
            <div className="space-y-2">
              <button
                onClick={handleDownloadZip}
                disabled={downloading}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-xs font-semibold shadow-sm transition disabled:opacity-50"
              >
                <Download className="w-4 h-4" />
                <span>{downloading ? 'Downloading...' : 'Export Project (.zip)'}</span>
              </button>
              <div className="flex items-center justify-center gap-1.5 text-[11px] text-gray-500">
                <Settings className="w-3 h-3" />
                <span>VeriGraph AI v1.0.0</span>
              </div>
            </div>
          ) : null}
        </div>
      </aside>

      <button
        type="button"
        className="lg:hidden fixed bottom-4 right-4 z-50 p-3 bg-primary-600 text-white rounded-full shadow-lg"
        onClick={() => setSidebarOpen((value: boolean) => !value)}
        aria-label={sidebarOpen ? 'Close menu' : 'Open menu'}
      >
        {sidebarOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      <main className="flex-1 min-w-0 lg:ml-0 min-h-screen">
        <Outlet />
      </main>
    </div>
  )
}
