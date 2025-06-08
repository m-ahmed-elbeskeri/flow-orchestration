import React, { useEffect, useState, useCallback, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft, RefreshCw, CheckCircle, XCircle, Clock, Pause, Play,
  Square, AlertTriangle, Activity, Cpu, MemoryStick, Network,
  BarChart3, TrendingUp, Settings, Download, Filter, Search,
  Zap, Target, Timer, AlertCircle, Info, GitBranch, Eye, Layers,
  PlayCircle, StopCircle, RotateCcw
} from 'lucide-react'
import { workflowApi } from '../api/client'

interface ExecutionState {
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'retrying' | 'paused'
  startTime?: string
  endTime?: string
  duration?: number
  attempts: number
  error?: string
  metrics?: StateMetrics
  dependencies: string[]
  transitions: string[]
  data?: any
  logs?: LogEntry[]
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

interface LogEntry {
  id: string
  timestamp: string
  level: 'debug' | 'info' | 'warning' | 'error'
  message: string
  state?: string
  data?: any
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
  type: 'warning' | 'error' | 'info'
  message: string
  timestamp: string
  state?: string
  severity: 'low' | 'medium' | 'high' | 'critical'
}

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />
    case 'failed': return <XCircle className="w-4 h-4 text-red-500" />
    case 'running': return <Activity className="w-4 h-4 text-blue-500 animate-pulse" />
    case 'paused': return <Pause className="w-4 h-4 text-yellow-500" />
    case 'pending': return <Clock className="w-4 h-4 text-gray-500" />
    case 'retrying': return <RotateCcw className="w-4 h-4 text-orange-500 animate-spin" />
    default: return <Clock className="w-4 h-4 text-gray-500" />
  }
}

const getStateStatusColor = (status: string) => {
  switch (status) {
    case 'completed': return 'border-green-500 bg-green-50'
    case 'failed': return 'border-red-500 bg-red-50'
    case 'running': return 'border-blue-500 bg-blue-50'
    case 'paused': return 'border-yellow-500 bg-yellow-50'
    case 'pending': return 'border-gray-300 bg-gray-50'
    case 'retrying': return 'border-orange-500 bg-orange-50'
    default: return 'border-gray-300 bg-gray-50'
  }
}

const formatDuration = (ms: number) => {
  if (!ms) return '0ms'
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}m`
}

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleTimeString()
}

export default function ExecutionMonitor() {
  const { workflowId, executionId } = useParams<{
    workflowId: string
    executionId: string
  }>()
  const queryClient = useQueryClient()
  
  const [selectedState, setSelectedState] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'overview' | 'states' | 'logs' | 'events' | 'metrics'>('overview')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [logFilter, setLogFilter] = useState<'all' | 'error' | 'warning' | 'info'>('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [websocket, setWebsocket] = useState<WebSocket | null>(null)

  // Queries
  const {
    data: execution,
    isLoading: executionLoading,
    error: executionError,
    refetch: refetchExecution
  } = useQuery({
    queryKey: ['execution', workflowId, executionId],
    queryFn: () => workflowApi.getExecution(workflowId!, executionId!),
    enabled: !!workflowId && !!executionId,
    refetchInterval: autoRefresh ? 2000 : false,
  })

  const {
    data: states = [],
    isLoading: statesLoading,
    refetch: refetchStates
  } = useQuery({
    queryKey: ['execution-states', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionStates(workflowId!, executionId!),
    enabled: !!workflowId && !!executionId,
    refetchInterval: autoRefresh ? 2000 : false,
  })

  const {
    data: metrics,
    isLoading: metricsLoading,
    refetch: refetchMetrics
  } = useQuery({
    queryKey: ['execution-metrics', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionMetrics(workflowId!, executionId!),
    enabled: !!workflowId && !!executionId,
    refetchInterval: autoRefresh ? 3000 : false,
  })

  const {
    data: events = [],
    refetch: refetchEvents
  } = useQuery({
    queryKey: ['execution-events', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionEvents(workflowId!, executionId!, { limit: 100 }),
    enabled: !!workflowId && !!executionId,
    refetchInterval: autoRefresh ? 2000 : false,
  })

  const {
    data: alerts = [],
    refetch: refetchAlerts
  } = useQuery({
    queryKey: ['execution-alerts', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionAlerts(workflowId!, executionId!),
    enabled: !!workflowId && !!executionId,
    refetchInterval: autoRefresh ? 5000 : false,
  })

  const {
    data: logs = [],
    refetch: refetchLogs
  } = useQuery({
    queryKey: ['execution-logs', workflowId, executionId, logFilter],
    queryFn: () => workflowApi.getExecutionLogs(workflowId!, executionId!, { 
      limit: 500,
      level: logFilter === 'all' ? undefined : logFilter
    }),
    enabled: !!workflowId && !!executionId,
    refetchInterval: autoRefresh ? 2000 : false,
  })

  // Mutations
  const pauseMutation = useMutation({
    mutationFn: () => workflowApi.pauseExecution(workflowId!, executionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['execution'] })
    }
  })

  const resumeMutation = useMutation({
    mutationFn: () => workflowApi.resumeExecution(workflowId!, executionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['execution'] })
    }
  })

  const cancelMutation = useMutation({
    mutationFn: () => workflowApi.cancelExecution(workflowId!, executionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['execution'] })
    }
  })

  const retryMutation = useMutation({
    mutationFn: () => workflowApi.retryExecution(workflowId!, executionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['execution'] })
    }
  })

  const retryStateMutation = useMutation({
    mutationFn: (stateName: string) => workflowApi.retryState(workflowId!, executionId!, stateName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['execution-states'] })
    }
  })

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!workflowId || !executionId || !autoRefresh) return

    try {
      const wsUrl = `ws://localhost:8000/api/v1/workflows/${workflowId}/executions/${executionId}/ws`
      const ws = new WebSocket(wsUrl)
      
      ws.onopen = () => {
        console.log('WebSocket connected')
        setWebsocket(ws)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          // Invalidate relevant queries when we receive updates
          if (data.type === 'execution_update') {
            queryClient.invalidateQueries({ queryKey: ['execution'] })
          }
          if (data.type === 'state_update') {
            queryClient.invalidateQueries({ queryKey: ['execution-states'] })
          }
          if (data.type === 'metrics_update') {
            queryClient.invalidateQueries({ queryKey: ['execution-metrics'] })
          }
          if (data.type === 'event') {
            queryClient.invalidateQueries({ queryKey: ['execution-events'] })
          }
          if (data.type === 'alert') {
            queryClient.invalidateQueries({ queryKey: ['execution-alerts'] })
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setWebsocket(null)
      }

      return () => {
        ws.close()
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
    }
  }, [workflowId, executionId, autoRefresh, queryClient])

  // Computed values
  const filteredLogs = useMemo(() => {
    if (!logs) return []
    return logs.filter(log => 
      searchTerm === '' || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.state?.toLowerCase().includes(searchTerm.toLowerCase())
    )
  }, [logs, searchTerm])

  const filteredEvents = useMemo(() => {
    if (!events) return []
    return events.filter(event => 
      searchTerm === '' || 
      event.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      event.state?.toLowerCase().includes(searchTerm.toLowerCase())
    )
  }, [events, searchTerm])

  const progressPercentage = useMemo(() => {
    if (!metrics) return 0
    return Math.round((metrics.completedStates / metrics.totalStates) * 100)
  }, [metrics])

  // Event handlers
  const handleRefresh = useCallback(() => {
    refetchExecution()
    refetchStates()
    refetchMetrics()
    refetchEvents()
    refetchAlerts()
    refetchLogs()
  }, [refetchExecution, refetchStates, refetchMetrics, refetchEvents, refetchAlerts, refetchLogs])

  const handleStateClick = useCallback((stateName: string) => {
    setSelectedState(selectedState === stateName ? null : stateName)
  }, [selectedState])

  const handleControlAction = useCallback(async (action: string) => {
    try {
      switch (action) {
        case 'pause':
          await pauseMutation.mutateAsync()
          break
        case 'resume':
          await resumeMutation.mutateAsync()
          break
        case 'cancel':
          if (window.confirm('Are you sure you want to cancel this execution?')) {
            await cancelMutation.mutateAsync()
          }
          break
        case 'retry':
          if (window.confirm('Are you sure you want to retry this execution?')) {
            await retryMutation.mutateAsync()
          }
          break
      }
    } catch (error) {
      console.error(`Failed to ${action} execution:`, error)
    }
  }, [pauseMutation, resumeMutation, cancelMutation, retryMutation])

  const handleExportData = useCallback(async () => {
    try {
      const blob = await workflowApi.exportExecutionData(workflowId!, executionId!)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `execution-${executionId}-data.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to export data:', error)
    }
  }, [workflowId, executionId])

  if (executionLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Activity className="w-8 h-8 animate-spin text-blue-500" />
        <span className="ml-2 text-lg">Loading execution...</span>
      </div>
    )
  }

  if (executionError) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <XCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Failed to load execution</h2>
          <p className="text-gray-600 mb-4">
            {executionError instanceof Error ? executionError.message : 'Unknown error occurred'}
          </p>
          <button
            onClick={handleRefresh}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          >
            <RefreshCw className="w-4 h-4 inline mr-2" />
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (!execution) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertTriangle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Execution not found</h2>
          <p className="text-gray-600">The requested execution could not be found.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <Link
                to={`/workflows/${workflowId}`}
                className="flex items-center text-gray-600 hover:text-gray-900"
              >
                <ArrowLeft className="w-5 h-5 mr-2" />
                Back to Workflow
              </Link>
              <div className="h-6 border-l border-gray-300" />
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  Execution Monitor
                </h1>
                <p className="text-sm text-gray-600">
                  {execution.workflow_id} • {executionId}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              {/* Auto-refresh toggle */}
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="rounded border-gray-300"
                />
                <span className="text-sm text-gray-600">Auto-refresh</span>
              </label>

              {/* WebSocket status indicator */}
              <div className="flex items-center space-x-2">
                <div
                  className={`w-2 h-2 rounded-full ${
                    websocket ? 'bg-green-500' : 'bg-red-500'
                  }`}
                />
                <span className="text-xs text-gray-500">
                  {websocket ? 'Connected' : 'Disconnected'}
                </span>
              </div>

              {/* Control buttons */}
              <div className="flex items-center space-x-2">
                {execution.status === 'running' && (
                  <button
                    onClick={() => handleControlAction('pause')}
                    disabled={pauseMutation.isPending}
                    className="px-3 py-1 bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:opacity-50"
                  >
                    <Pause className="w-4 h-4" />
                  </button>
                )}
                
                {execution.status === 'paused' && (
                  <button
                    onClick={() => handleControlAction('resume')}
                    disabled={resumeMutation.isPending}
                    className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
                  >
                    <Play className="w-4 h-4" />
                  </button>
                )}

                {['running', 'paused'].includes(execution.status) && (
                  <button
                    onClick={() => handleControlAction('cancel')}
                    disabled={cancelMutation.isPending}
                    className="px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50"
                  >
                    <Square className="w-4 h-4" />
                  </button>
                )}

                {['failed', 'cancelled'].includes(execution.status) && (
                  <button
                    onClick={() => handleControlAction('retry')}
                    disabled={retryMutation.isPending}
                    className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                )}

                <button
                  onClick={handleRefresh}
                  className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>

                <button
                  onClick={handleExportData}
                  className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Status bar */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                {getStatusIcon(execution.status)}
                <span className="font-medium capitalize">{execution.status}</span>
              </div>
              
              {metrics && (
                <>
                  <div className="text-sm text-gray-600">
                    Progress: {progressPercentage}% ({metrics.completedStates}/{metrics.totalStates} states)
                  </div>
                  
                  <div className="text-sm text-gray-600">
                    Duration: {formatDuration(metrics.totalExecutionTime)}
                  </div>
                  
                  <div className="text-sm text-gray-600">
                    Active: {metrics.activeStates}
                  </div>
                  
                  {metrics.failedStates > 0 && (
                    <div className="text-sm text-red-600">
                      Failed: {metrics.failedStates}
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Progress bar */}
            {metrics && (
              <div className="w-48">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progressPercentage}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Alerts */}
          {alerts && alerts.length > 0 && (
            <div className="mt-4 space-y-2">
              {alerts.slice(0, 3).map((alert) => (
                <div
                  key={alert.id}
                  className={`p-3 rounded-lg ${
                    alert.type === 'error'
                      ? 'bg-red-50 border border-red-200 text-red-800'
                      : alert.type === 'warning'
                      ? 'bg-yellow-50 border border-yellow-200 text-yellow-800'
                      : 'bg-blue-50 border border-blue-200 text-blue-800'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    {alert.type === 'error' ? (
                      <XCircle className="w-4 h-4" />
                    ) : alert.type === 'warning' ? (
                      <AlertTriangle className="w-4 h-4" />
                    ) : (
                      <Info className="w-4 h-4" />
                    )}
                    <span className="font-medium">{alert.message}</span>
                    <span className="text-xs opacity-75">
                      {formatTimestamp(alert.timestamp)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Tab navigation */}
        <div className="mb-6">
          <nav className="flex space-x-8">
            {[
              { key: 'overview', label: 'Overview', icon: Eye },
              { key: 'states', label: 'States', icon: Layers },
              { key: 'logs', label: 'Logs', icon: FileText },
              { key: 'events', label: 'Events', icon: Activity },
              { key: 'metrics', label: 'Metrics', icon: BarChart3 },
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key as any)}
                className={`flex items-center space-x-2 px-3 py-2 text-sm font-medium rounded-lg ${
                  activeTab === key
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Tab content */}
        <div className="bg-white rounded-lg shadow">
          {activeTab === 'overview' && (
            <OverviewTab
              execution={execution}
              metrics={metrics}
              states={states}
              alerts={alerts}
              onStateSelect={handleStateClick}
            />
          )}

          {activeTab === 'states' && (
            <StatesTab
              states={states}
              selectedState={selectedState}
              onStateSelect={handleStateClick}
              onRetryState={(stateName) => retryStateMutation.mutate(stateName)}
              isRetrying={retryStateMutation.isPending}
            />
          )}

          {activeTab === 'logs' && (
            <LogsTab
              logs={filteredLogs}
              filter={logFilter}
              onFilterChange={setLogFilter}
              searchTerm={searchTerm}
              onSearchChange={setSearchTerm}
            />
          )}

          {activeTab === 'events' && (
            <EventsTab
              events={filteredEvents}
              searchTerm={searchTerm}
              onSearchChange={setSearchTerm}
            />
          )}

          {activeTab === 'metrics' && (
            <MetricsTab
              metrics={metrics}
              states={states}
              execution={execution}
            />
          )}
        </div>
      </div>
    </div>
  )
}

// Sub-components
function OverviewTab({ execution, metrics, states, alerts, onStateSelect }) {
  return (
    <div className="p-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Execution Summary */}
        <div>
          <h3 className="text-lg font-medium text-gray-900 mb-4">Execution Summary</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">Status:</span>
              <div className="flex items-center space-x-2">
                {getStatusIcon(execution.status)}
                <span className="capitalize">{execution.status}</span>
              </div>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Started:</span>
              <span>{new Date(execution.started_at).toLocaleString()}</span>
            </div>
            {execution.completed_at && (
              <div className="flex justify-between">
                <span className="text-gray-600">Completed:</span>
                <span>{new Date(execution.completed_at).toLocaleString()}</span>
              </div>
            )}
            {metrics && (
              <div className="flex justify-between">
                <span className="text-gray-600">Duration:</span>
                <span>{formatDuration(metrics.totalExecutionTime)}</span>
              </div>
            )}
          </div>
        </div>

        {/* Metrics Overview */}
        {metrics && (
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-4">Metrics</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {metrics.completedStates}
                </div>
                <div className="text-sm text-gray-600">Completed</div>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-red-600">
                  {metrics.failedStates}
                </div>
                <div className="text-sm text-gray-600">Failed</div>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {metrics.activeStates}
                </div>
                <div className="text-sm text-gray-600">Active</div>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg">
                <div className="text-2xl font-bold text-gray-600">
                  {formatDuration(metrics.avgStateTime)}
                </div>
                <div className="text-sm text-gray-600">Avg Time</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* States Overview */}
      <div className="mt-8">
        <h3 className="text-lg font-medium text-gray-900 mb-4">States</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {states.map((state) => (
            <div
              key={state.name}
              onClick={() => onStateSelect(state.name)}
              className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${getStateStatusColor(
                state.status
              )} hover:shadow-md`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium">{state.name}</span>
                {getStatusIcon(state.status)}
              </div>
              <div className="text-sm text-gray-600">
                {state.duration && (
                  <span>Duration: {formatDuration(state.duration)}</span>
                )}
                {state.attempts > 1 && (
                  <span className="ml-2">({state.attempts} attempts)</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatesTab({ states, selectedState, onStateSelect, onRetryState, isRetrying }) {
  return (
    <div className="p-6">
      <div className="space-y-4">
        {states.map((state) => (
          <div
            key={state.name}
            className={`border rounded-lg transition-all ${
              selectedState === state.name ? 'ring-2 ring-blue-500' : ''
            } ${getStateStatusColor(state.status)}`}
          >
            <div
              onClick={() => onStateSelect(state.name)}
              className="p-4 cursor-pointer"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(state.status)}
                  <span className="font-medium">{state.name}</span>
                  {state.attempts > 1 && (
                    <span className="text-sm text-gray-500">
                      (Attempt {state.attempts})
                    </span>
                  )}
                </div>
                <div className="flex items-center space-x-4">
                  {state.duration && (
                    <span className="text-sm text-gray-600">
                      {formatDuration(state.duration)}
                    </span>
                  )}
                  {state.status === 'failed' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onRetryState(state.name)
                      }}
                      disabled={isRetrying}
                      className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
                    >
                      <RotateCcw className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            </div>

            {selectedState === state.name && (
              <div className="border-t bg-white p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium mb-2">Details</h4>
                    <div className="space-y-1 text-sm">
                      <div>Status: <span className="capitalize">{state.status}</span></div>
                      {state.startTime && (
                        <div>Started: {new Date(state.startTime).toLocaleString()}</div>
                      )}
                      {state.endTime && (
                        <div>Ended: {new Date(state.endTime).toLocaleString()}</div>
                      )}
                      {state.duration && (
                        <div>Duration: {formatDuration(state.duration)}</div>
                      )}
                    </div>
                  </div>
                  
                  {state.dependencies.length > 0 && (
                    <div>
                      <h4 className="font-medium mb-2">Dependencies</h4>
                      <div className="flex flex-wrap gap-1">
                        {state.dependencies.map((dep) => (
                          <span
                            key={dep}
                            className="px-2 py-1 text-xs bg-gray-100 rounded"
                          >
                            {dep}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {state.error && (
                  <div className="mt-4">
                    <h4 className="font-medium mb-2 text-red-600">Error</h4>
                    <pre className="bg-red-50 p-3 rounded text-sm text-red-800 overflow-x-auto">
                      {state.error}
                    </pre>
                  </div>
                )}

                {state.data && (
                  <div className="mt-4">
                    <h4 className="font-medium mb-2">Data</h4>
                    <pre className="bg-gray-50 p-3 rounded text-sm overflow-x-auto">
                      {JSON.stringify(state.data, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function LogsTab({ logs, filter, onFilterChange, searchTerm, onSearchChange }) {
  const getLevelColor = (level: string) => {
    switch (level) {
      case 'error': return 'text-red-600'
      case 'warning': return 'text-yellow-600'
      case 'info': return 'text-blue-600'
      case 'debug': return 'text-gray-600'
      default: return 'text-gray-600'
    }
  }

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <Filter className="w-4 h-4 text-gray-500" />
          <select
            value={filter}
            onChange={(e) => onFilterChange(e.target.value as any)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm"
          >
            <option value="all">All Levels</option>
            <option value="error">Error</option>
            <option value="warning">Warning</option>
            <option value="info">Info</option>
            <option value="debug">Debug</option>
          </select>
        </div>
        
        <div className="flex-1 max-w-md">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search logs..."
              value={searchTerm}
              onChange={(e) => onSearchChange(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm"
            />
          </div>
        </div>
      </div>

      <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-y-auto">
        <div className="space-y-1 font-mono text-sm">
          {logs.map((log) => (
            <div key={log.id} className="flex space-x-4">
              <span className="text-gray-400 whitespace-nowrap">
                {formatTimestamp(log.timestamp)}
              </span>
              <span className={`uppercase font-medium ${getLevelColor(log.level)}`}>
                {log.level}
              </span>
              {log.state && (
                <span className="text-blue-400">[{log.state}]</span>
              )}
              <span className="text-gray-100 flex-1">{log.message}</span>
            </div>
          ))}
          {logs.length === 0 && (
            <div className="text-gray-400 text-center py-8">
              No logs found matching the current filter
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function EventsTab({ events, searchTerm, onSearchChange }) {
  const getEventTypeColor = (type: string) => {
    switch (type) {
      case 'state_failed': return 'text-red-600'
      case 'alert': return 'text-orange-600'
      case 'resource_warning': return 'text-yellow-600'
      case 'state_completed': return 'text-green-600'
      case 'state_started': return 'text-blue-600'
      default: return 'text-gray-600'
    }
  }

  return (
    <div className="p-6">
      <div className="mb-4">
        <div className="relative max-w-md">
          <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search events..."
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm"
          />
        </div>
      </div>

      <div className="space-y-3">
        {events.map((event) => (
          <div
            key={event.id}
            className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-3">
                <span className={`font-medium ${getEventTypeColor(event.type)}`}>
                  {event.type.replace(/_/g, ' ').toUpperCase()}
                </span>
                {event.state && (
                  <span className="text-sm text-gray-500">[{event.state}]</span>
                )}
              </div>
              <span className="text-xs text-gray-500">
                {formatTimestamp(event.timestamp)}
              </span>
            </div>
            <p className="text-gray-900">{event.message}</p>
            {event.metadata && Object.keys(event.metadata).length > 0 && (
              <details className="mt-2">
                <summary className="text-sm text-gray-500 cursor-pointer">
                  View metadata
                </summary>
                <pre className="mt-2 text-xs bg-gray-100 p-2 rounded overflow-x-auto">
                  {JSON.stringify(event.metadata, null, 2)}
                </pre>
              </details>
            )}
          </div>
        ))}
        {events.length === 0 && (
          <div className="text-gray-500 text-center py-8">
            No events found matching the current search
          </div>
        )}
      </div>
    </div>
  )
}

function MetricsTab({ metrics, states, execution }) {
  if (!metrics) {
    return (
      <div className="p-6 text-center text-gray-500">
        No metrics available
      </div>
    )
  }

  const resourceData = [
    { name: 'CPU', value: metrics.resourceUtilization.cpu, color: 'bg-blue-500' },
    { name: 'Memory', value: metrics.resourceUtilization.memory, color: 'bg-green-500' },
    { name: 'Network', value: metrics.resourceUtilization.network, color: 'bg-purple-500' },
  ]

  return (
    <div className="p-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Key Metrics */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-medium text-gray-900 mb-3">Execution Stats</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">Total States:</span>
              <span className="font-medium">{metrics.totalStates}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Completed:</span>
              <span className="font-medium text-green-600">{metrics.completedStates}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Failed:</span>
              <span className="font-medium text-red-600">{metrics.failedStates}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Active:</span>
              <span className="font-medium text-blue-600">{metrics.activeStates}</span>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-medium text-gray-900 mb-3">Performance</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Time:</span>
              <span className="font-medium">{formatDuration(metrics.totalExecutionTime)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Avg State Time:</span>
              <span className="font-medium">{formatDuration(metrics.avgStateTime)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Throughput:</span>
              <span className="font-medium">{metrics.throughput.toFixed(2)} states/sec</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Error Rate:</span>
              <span className="font-medium text-red-600">{(metrics.errorRate * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Resource Utilization */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-medium text-gray-900 mb-3">Resource Usage</h3>
          <div className="space-y-3">
            {resourceData.map((resource) => (
              <div key={resource.name}>
                <div className="flex justify-between mb-1">
                  <span className="text-gray-600">{resource.name}:</span>
                  <span className="font-medium">{resource.value.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${resource.color}`}
                    style={{ width: `${resource.value}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* State Performance Chart */}
      <div className="mt-8">
        <h3 className="font-medium text-gray-900 mb-4">State Performance</h3>
        <div className="bg-gray-50 p-4 rounded-lg overflow-x-auto">
          <div className="min-w-full">
            <div className="flex items-end space-x-2 h-32">
              {states.map((state, index) => (
                <div
                  key={state.name}
                  className="flex-1 flex flex-col items-center"
                  title={`${state.name}: ${formatDuration(state.duration || 0)}`}
                >
                  <div
                    className={`w-full min-w-8 rounded-t ${
                      state.status === 'completed' ? 'bg-green-500' :
                      state.status === 'failed' ? 'bg-red-500' :
                      state.status === 'running' ? 'bg-blue-500' :
                      'bg-gray-300'
                    }`}
                    style={{
                      height: `${Math.max(
                        (state.duration || 0) / Math.max(...states.map(s => s.duration || 0)) * 100,
                        5
                      )}%`
                    }}
                  />
                  <span className="text-xs text-gray-600 mt-1 truncate max-w-16">
                    {state.name}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}