// src/components/ExecutionMonitor.tsx
import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, RefreshCw, CheckCircle, XCircle, Clock } from 'lucide-react'

interface ExecutionLog {
  timestamp: string
  state: string
  message: string
  level: 'info' | 'warning' | 'error'
}

export default function ExecutionMonitor() {
  const { id } = useParams()
  const [status, setStatus] = useState<'running' | 'completed' | 'failed'>('running')
  const [logs, setLogs] = useState<ExecutionLog[]>([])
  const [currentState, setCurrentState] = useState('initializing')
  
  useEffect(() => {
    // Simulate execution logs
    const mockLogs: ExecutionLog[] = [
      { timestamp: new Date().toISOString(), state: 'start', message: 'Workflow execution started', level: 'info' },
      { timestamp: new Date().toISOString(), state: 'fetch_data', message: 'Fetching data from API...', level: 'info' },
      { timestamp: new Date().toISOString(), state: 'fetch_data', message: 'Retrieved 100 records', level: 'info' },
      { timestamp: new Date().toISOString(), state: 'process', message: 'Processing data...', level: 'info' },
    ]
    
    // Simulate real-time logs
    let index = 0
    const interval = setInterval(() => {
      if (index < mockLogs.length) {
        setLogs((prev) => [...prev, mockLogs[index]])
        setCurrentState(mockLogs[index].state)
        index++
      } else {
        setStatus('completed')
        clearInterval(interval)
      }
    }, 1000)
    
    return () => clearInterval(interval)
  }, [id])
  
  const getStatusIcon = () => {
    switch (status) {
      case 'running':
        return <RefreshCw className="h-5 w-5 text-blue-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />
    }
  }
  
  const getLogIcon = (level: string) => {
    switch (level) {
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />
      case 'warning':
        return <Clock className="h-4 w-4 text-yellow-500" />
      default:
        return <CheckCircle className="h-4 w-4 text-green-500" />
    }
  }
  
  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              to="/workflows"
              className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700"
            >
              <ArrowLeft className="h-4 w-4 mr-1" />
              Back to Workflows
            </Link>
            <h1 className="text-xl font-semibold">Workflow Execution</h1>
          </div>
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <span className="text-sm font-medium capitalize">{status}</span>
          </div>
        </div>
      </div>
      
      {/* Execution Details */}
      <div className="flex-1 flex">
        {/* States Progress */}
        <div className="w-64 bg-gray-50 border-r border-gray-200 p-4">
          <h2 className="font-semibold mb-4">Execution Progress</h2>
          <div className="space-y-2">
            {['start', 'fetch_data', 'process', 'validate', 'save', 'end'].map((state) => (
              <div
                key={state}
                className={`p-2 rounded text-sm ${
                  currentState === state
                    ? 'bg-blue-100 text-blue-700 font-medium'
                    : logs.some((l) => l.state === state)
                    ? 'bg-green-50 text-green-700'
                    : 'text-gray-500'
                }`}
              >
                {state.replace('_', ' ').charAt(0).toUpperCase() + state.slice(1).replace('_', ' ')}
              </div>
            ))}
          </div>
        </div>
        
        {/* Logs */}
        <div className="flex-1 p-4">
          <h2 className="font-semibold mb-4">Execution Logs</h2>
          <div className="bg-gray-900 text-gray-100 rounded-lg p-4 h-full overflow-auto font-mono text-sm">
            {logs.map((log, index) => (
              <div key={index} className="flex items-start gap-2 mb-2">
                <span className="text-gray-500 text-xs">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                {getLogIcon(log.level)}
                <span className="text-blue-400">[{log.state}]</span>
                <span>{log.message}</span>
              </div>
            ))}
            {status === 'running' && (
              <div className="flex items-center gap-2 text-gray-400">
                <RefreshCw className="h-4 w-4 animate-spin" />
                <span>Waiting for next event...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
} 