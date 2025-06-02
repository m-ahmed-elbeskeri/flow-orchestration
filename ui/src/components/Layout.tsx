// src/components/Layout.tsx
import { Link, useLocation } from 'react-router-dom'
import { Workflow, Play, LayoutDashboard } from 'lucide-react'

export default function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  
  const isActive = (path: string) => location.pathname === path
  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Workflow className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-semibold">Workflow Orchestrator</span>
            </div>
            
            <nav className="flex space-x-8">
              <Link
                to="/"
                className={`flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                  isActive('/') ? 'text-blue-600 bg-blue-50' : 'text-gray-700 hover:text-gray-900'
                }`}
              >
                <LayoutDashboard className="h-4 w-4 mr-2" />
                Workflows
              </Link>
            </nav>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="flex-1">
        {children}
      </main>
    </div>
  )
}