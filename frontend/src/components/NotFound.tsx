import { Link } from 'react-router-dom'
import { Home, RotateCcw, Search } from 'lucide-react'

export function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="text-center max-w-md mx-auto px-4">
        <div className="mb-8">
          <Search className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Page Not Found</h1>
          <p className="text-gray-600 text-lg">
            The page you're looking for doesn't exist or has been moved.
          </p>
        </div>
        
        <div className="space-y-3">
          <Link 
            to="/" 
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            <Home className="w-5 h-5" />
            Go to Dashboard
          </Link>
          
          <Link 
            to="/verify/local" 
            className="btn-secondary w-full flex items-center justify-center gap-2"
          >
            <RotateCcw className="w-5 h-5" />
            Start New Verification
          </Link>
        </div>
      </div>
    </div>
  )
}