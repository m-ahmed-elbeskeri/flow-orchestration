// ui/src/components/ExecutionMonitor.tsx
import React, { useEffect, useState, useCallback, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft, RefreshCw, CheckCircle, XCircle, Clock, Pause, Play, 
  Square, AlertTriangle, Activity, Cpu, MemoryStick, Network,
  BarChart3, TrendingUp, Settings, Download, Filter, Search,
  Zap, Target, Timer, AlertCircle, Info, GitBranch
} from 'lucide-react'
import { workflowApi } from '../api/client'
import WorkflowFlowChart from './WorkflowFlowChart'
import MetricsChart from './MetricsChart'
import AlertPanel from './AlertPanel'

interface ExecutionState {
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'retrying'
  startTime?: string
  endTime?: string
  duration?: number
  attempts: number
  error?: string
  metrics?: StateMetrics
  dependencies: string[]
  transitions: string[]
}

interface StateMetrics {
  cpuUsage: number
  memoryUsage: number
  networkIO: number
  executionTime: number
  retryCount: number
}

interface ExecutionMetrics {
  totalStates: number
  completedStates: number
  failedStates: number
  activeStates: number
  totalExecutionTime: number
  avgStateTime: number
  resourceUtilization: {
    cpu: number
    memory: number
    network: number
  }
  throughput: number
  errorRate: number
}

interface ExecutionEvent {
  id: string
  timestamp: string
  type: 'state_started' | 'state_completed' | 'state_failed' | 'workflow_paused' | 'alert' | 'resource_warning'
  state?: string
  message: string
  level: 'info' | 'warning' | 'error' | 'success'
  metadata?: Record<string, any>
}

interface Alert {
  id: string
  severity: 'info' | 'warning' | 'error' | 'critical'
  title: string
  message: string
  timestamp: string
  resolved: boolean
  source: string
}

const ExecutionMonitor: React.FC = () => {
  const { workflowId, executionId } = useParams<{ workflowId: string, executionId: string }>()
  const queryClient = useQueryClient()
  
  // State management
  const [activeTab, setActiveTab] = useState<'overview' | 'states' | 'metrics' | 'logs' | 'alerts'>('overview')
  const [selectedState, setSelectedState] = useState<string | null>(null)
  const [showFilters, setShowFilters] = useState(false)
  const [eventFilter, setEventFilter] = useState({ level: 'all', type: 'all', search: '' })
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshInterval, setRefreshInterval] = useState(2000)

  // Real-time data fetching
  const { data: execution, isLoading } = useQuery({
    queryKey: ['execution', workflowId, executionId],
    queryFn: () => workflowApi.getExecution(workflowId!, executionId!),
    refetchInterval: autoRefresh ? refreshInterval : false,
    enabled: !!workflowId && !!executionId
  })

  const { data: states } = useQuery({
    queryKey: ['execution-states', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionStates(workflowId!, executionId!),
    refetchInterval: autoRefresh ? refreshInterval : false,
    enabled: !!workflowId && !!executionId
  })

  const { data: metrics } = useQuery({
    queryKey: ['execution-metrics', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionMetrics(workflowId!, executionId!),
    refetchInterval: autoRefresh ? refreshInterval : false,
    enabled: !!workflowId && !!executionId
  })

  const { data: events } = useQuery({
    queryKey: ['execution-events', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionEvents(workflowId!, executionId!),
    refetchInterval: autoRefresh ? refreshInterval : false,
    enabled: !!workflowId && !!executionId
  })

  const { data: alerts } = useQuery({
    queryKey: ['execution-alerts', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionAlerts(workflowId!, executionId!),
    refetchInterval: autoRefresh ? refreshInterval : false,
    enabled: !!workflowId && !!executionId
  })

  // Control mutations
  const pauseMutation = useMutation({
    mutationFn: () => workflowApi.pauseExecution(workflowId!, executionId!),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['execution'] })
  })

  const resumeMutation = useMutation({
    mutationFn: () => workflowApi.resumeExecution(workflowId!, executionId!),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['execution'] })
  })

  const cancelMutation = useMutation({
    mutationFn: () => workflowApi.cancelExecution(workflowId!, executionId!),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['execution'] })
  })

  const retryStateMutation = useMutation({
    mutationFn: (stateName: string) => workflowApi.retryState(workflowId!, executionId!, stateName),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['execution'] })
  })

  // Helper functions
  const getStatusIcon = useCallback((status: string) => {
    switch (status) {
      case 'running': return <Activity className="w-5 h-5 text-blue-500 animate-pulse" />
      case 'completed': return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed': return <XCircle className="w-5 h-5 text-red-500" />
      case 'paused': return <Pause className="w-5 h-5 text-yellow-500" />
      case 'pending': return <Clock className="w-5 h-5 text-gray-400" />
      default: return <Clock className="w-5 h-5 text-gray-400" />
    }
  }, [])

  const getProgressPercentage = useCallback(() => {
    if (!metrics) return 0
    return Math.round((metrics.completedStates / metrics.totalStates) * 100)
  }, [metrics])

  const filteredEvents = useMemo(() => {
    if (!events) return []
    return events.filter(event => {
      const matchesLevel = eventFilter.level === 'all' || event.level === eventFilter.level
      const matchesType = eventFilter.type === 'all' || event.type === eventFilter.type
      const matchesSearch = !eventFilter.search || 
        event.message.toLowerCase().includes(eventFilter.search.toLowerCase()) ||
        event.state?.toLowerCase().includes(eventFilter.search.toLowerCase())
      return matchesLevel && matchesType && matchesSearch
    })
  }, [events, eventFilter])

  const formatDuration = useCallback((ms: number) => {
    if (ms < 1000) return `${ms}ms`
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
    if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`
    return `${(ms / 3600000).toFixed(1)}h`
  }, [])

  const exportData = useCallback(async () => {
    try {
      const exportData = {
        execution,
        states,
        metrics,
        events: filteredEvents,
        alerts,
        exportTime: new Date().toISOString()
      }
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `execution-${executionId}-${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Export failed:', error)
    }
  }, [execution, states, metrics, filteredEvents, alerts, executionId])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading execution details...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <Link to="/workflows" className="flex items-center text-gray-600 hover:text-gray-900">
                <ArrowLeft className="w-5 h-5 mr-2" />
                Back to Workflows
              </Link>
              <div className="h-6 border-l border-gray-300" />
              <div>
                <h1 className="text-xl font-semibold text-gray-900">{execution?.workflow_name}</h1>
                <p className="text-sm text-gray-500">Execution ID: {executionId}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Status Badge */}
              <div className="flex items-center space-x-2">
                {getStatusIcon(execution?.status || 'pending')}
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  execution?.status === 'running' ? 'bg-blue-100 text-blue-800' :
                  execution?.status === 'completed' ? 'bg-green-100 text-green-800' :
                  execution?.status === 'failed' ? 'bg-red-100 text-red-800' :
                  execution?.status === 'paused' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {execution?.status?.charAt(0).toUpperCase() + execution?.status?.slice(1)}
                </span>
              </div>

              {/* Control Buttons */}
              <div className="flex items-center space-x-2">
                {execution?.status === 'running' && (
                  <button
                    onClick={() => pauseMutation.mutate()}
                    disabled={pauseMutation.isPending}
                    className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg"
                    title="Pause Execution"
                  >
                    <Pause className="w-5 h-5" />
                  </button>
                )}
                
                {execution?.status === 'paused' && (
                  <button
                    onClick={() => resumeMutation.mutate()}
                    disabled={resumeMutation.isPending}
                    className="p-2 text-green-600 hover:text-green-700 hover:bg-green-50 rounded-lg"
                    title="Resume Execution"
                  >
                    <Play className="w-5 h-5" />
                  </button>
                )}

                {['running', 'paused'].includes(execution?.status || '') && (
                  <button
                    onClick={() => cancelMutation.mutate()}
                    disabled={cancelMutation.isPending}
                    className="p-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg"
                    title="Cancel Execution"
                  >
                    <Square className="w-5 h-5" />
                  </button>
                )}

                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`p-2 rounded-lg ${autoRefresh ? 'text-blue-600 bg-blue-50' : 'text-gray-600 hover:bg-gray-100'}`}
                  title={autoRefresh ? 'Disable Auto-refresh' : 'Enable Auto-refresh'}
                >
                  <RefreshCw className={`w-5 h-5 ${autoRefresh ? 'animate-spin' : ''}`} />
                </button>

                <button
                  onClick={exportData}
                  className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg"
                  title="Export Data"
                >
                  <Download className="w-5 h-5" />
                </button>

                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className={`p-2 rounded-lg ${showFilters ? 'text-blue-600 bg-blue-50' : 'text-gray-600 hover:bg-gray-100'}`}
                  title="Filters"
                >
                  <Filter className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          {metrics && (
            <div className="pb-4">
              <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                <span>Progress: {metrics.completedStates} of {metrics.totalStates} states</span>
                <span>{getProgressPercentage()}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-300 ${
                    execution?.status === 'failed' ? 'bg-red-500' :
                    execution?.status === 'completed' ? 'bg-green-500' :
                    'bg-blue-500'
                  }`}
                  style={{ width: `${getProgressPercentage()}%` }}
                />
              </div>
            </div>
          )}

          {/* Filter Panel */}
          {showFilters && (
            <div className="pb-4 border-t border-gray-200 pt-4">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <label className="text-sm font-medium text-gray-700">Level:</label>
                  <select
                    value={eventFilter.level}
                    onChange={(e) => setEventFilter(prev => ({ ...prev, level: e.target.value }))}
                    className="border border-gray-300 rounded-md px-3 py-1 text-sm"
                  >
                    <option value="all">All</option>
                    <option value="info">Info</option>
                    <option value="warning">Warning</option>
                    <option value="error">Error</option>
                    <option value="success">Success</option>
                  </select>
                </div>

                <div className="flex items-center space-x-2">
                  <label className="text-sm font-medium text-gray-700">Type:</label>
                  <select
                    value={eventFilter.type}
                    onChange={(e) => setEventFilter(prev => ({ ...prev, type: e.target.value }))}
                    className="border border-gray-300 rounded-md px-3 py-1 text-sm"
                  >
                    <option value="all">All</option>
                    <option value="state_started">State Started</option>
                    <option value="state_completed">State Completed</option>
                    <option value="state_failed">State Failed</option>
                    <option value="alert">Alert</option>
                  </select>
                </div>

                <div className="flex items-center space-x-2">
                  <Search className="w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search events..."
                    value={eventFilter.search}
                    onChange={(e) => setEventFilter(prev => ({ ...prev, search: e.target.value }))}
                    className="border border-gray-300 rounded-md px-3 py-1 text-sm w-64"
                  />
                </div>

                <div className="flex items-center space-x-2">
                  <label className="text-sm font-medium text-gray-700">Refresh:</label>
                  <select
                    value={refreshInterval}
                    onChange={(e) => setRefreshInterval(Number(e.target.value))}
                    className="border border-gray-300 rounded-md px-3 py-1 text-sm"
                  >
                    <option value={1000}>1s</option>
                    <option value={2000}>2s</option>
                    <option value={5000}>5s</option>
                    <option value={10000}>10s</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* Tab Navigation */}
          <div className="flex space-x-8 border-b border-gray-200">
            {[
              { id: 'overview', label: 'Overview', icon: Activity },
              { id: 'states', label: 'States', icon: GitBranch },
              { id: 'metrics', label: 'Metrics', icon: BarChart3 },
              { id: 'logs', label: 'Events', icon: Info },
              { id: 'alerts', label: 'Alerts', icon: AlertTriangle }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
                {tab.id === 'alerts' && alerts && alerts.filter(a => !a.resolved).length > 0 && (
                  <span className="bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                    {alerts.filter(a => !a.resolved).length}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && (
          <OverviewTab
            execution={execution}
            metrics={metrics}
            states={states}
            alerts={alerts}
            onStateSelect={setSelectedState}
          />
        )}

        {activeTab === 'states' && (
          <StatesTab
            states={states}
            selectedState={selectedState}
            onStateSelect={setSelectedState}
            onRetryState={retryStateMutation.mutate}
            isRetrying={retryStateMutation.isPending}
          />
        )}

        {activeTab === 'metrics' && (
          <MetricsTab
            metrics={metrics}
            execution={execution}
            formatDuration={formatDuration}
          />
        )}

        {activeTab === 'logs' && (
          <EventsTab
            events={filteredEvents}
            formatDuration={formatDuration}
          />
        )}

        {activeTab === 'alerts' && (
          <AlertsTab
            alerts={alerts}
            workflowId={workflowId!}
            executionId={executionId!}
          />
        )}
      </div>
    </div>
  )
}

export default ExecutionMonitor