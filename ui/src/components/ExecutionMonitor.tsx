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

export default function ExecutionMonitor() {
  const { workflowId, executionId } = useParams()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'overview' | 'states' | 'logs' | 'metrics' | 'timeline'>('overview')
  const [selectedState, setSelectedState] = useState<string | null>(null)
  const [logFilter, setLogFilter] = useState<string>('')
  const [logLevel, setLogLevel] = useState<string>('all')
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null)

  // Real-time data fetching
  const { data: execution, isLoading } = useQuery({
    queryKey: ['execution', workflowId, executionId],
    queryFn: () => workflowApi.getExecution(workflowId!, executionId!),
    refetchInterval: 2000, // Fallback polling
  })

  const { data: states } = useQuery({
    queryKey: ['execution-states', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionStates(workflowId!, executionId!),
    refetchInterval: 1000,
  })

  const { data: metrics } = useQuery({
    queryKey: ['execution-metrics', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionMetrics(workflowId!, executionId!),
    refetchInterval: 2000,
  })

  const { data: events } = useQuery({
    queryKey: ['execution-events', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionEvents(workflowId!, executionId!, { limit: 100 }),
    refetchInterval: 1000,
  })

  const { data: logs } = useQuery({
    queryKey: ['execution-logs', workflowId, executionId, logLevel],
    queryFn: () => workflowApi.getExecutionLogs(workflowId!, executionId!, { 
      limit: 1000,
      level: logLevel !== 'all' ? logLevel as any : undefined
    }),
    refetchInterval: 1000,
  })

  // WebSocket for real-time updates
  useEffect(() => {
    if (!workflowId || !executionId) return

    const ws = workflowApi.createWebSocket(`/api/v1/workflows/${workflowId}/executions/${executionId}/ws`)
    
    ws.onopen = () => {
      console.log('📡 WebSocket connected for real-time updates')
      setWsConnection(ws)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('📨 Real-time update:', data)
        
        // Invalidate relevant queries to trigger refetch
        queryClient.invalidateQueries(['execution', workflowId, executionId])
        queryClient.invalidateQueries(['execution-states', workflowId, executionId])
        queryClient.invalidateQueries(['execution-metrics', workflowId, executionId])
        queryClient.invalidateQueries(['execution-events', workflowId, executionId])
        queryClient.invalidateQueries(['execution-logs', workflowId, executionId])
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }

    ws.onclose = () => {
      console.log('📡 WebSocket disconnected')
      setWsConnection(null)
    }

    ws.onerror = (error) => {
      console.error('📡 WebSocket error:', error)
    }

    return () => {
      ws.close()
    }
  }, [workflowId, executionId, queryClient])

  // Execution control mutations
  const pauseMutation = useMutation({
    mutationFn: () => workflowApi.pauseExecution(workflowId!, executionId!),
    onSuccess: () => {
      queryClient.invalidateQueries(['execution', workflowId, executionId])
    }
  })

  const resumeMutation = useMutation({
    mutationFn: () => workflowApi.resumeExecution(workflowId!, executionId!),
    onSuccess: () => {
      queryClient.invalidateQueries(['execution', workflowId, executionId])
    }
  })

  const cancelMutation = useMutation({
    mutationFn: () => workflowApi.cancelExecution(workflowId!, executionId!),
    onSuccess: () => {
      queryClient.invalidateQueries(['execution', workflowId, executionId])
    }
  })

  const retryMutation = useMutation({
    mutationFn: () => workflowApi.retryExecution(workflowId!, executionId!),
    onSuccess: () => {
      queryClient.invalidateQueries(['execution', workflowId, executionId])
    }
  })

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-500" />
      case 'running':
        return <Clock className="w-5 h-5 text-blue-500 animate-spin" />
      case 'paused':
        return <Pause className="w-5 h-5 text-yellow-500" />
      case 'pending':
        return <Clock className="w-5 h-5 text-gray-400" />
      default:
        return <Clock className="w-5 h-5 text-gray-400" />
    }
  }

  const getStateStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'running':
        return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'paused':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 'pending':
        return 'bg-gray-100 text-gray-800 border-gray-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const filteredLogs = useMemo(() => {
    if (!logs) return []
    return logs.filter(log => 
      log.message.toLowerCase().includes(logFilter.toLowerCase()) ||
      (log.state && log.state.toLowerCase().includes(logFilter.toLowerCase()))
    )
  }, [logs, logFilter])

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m ${seconds % 60}s`
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`
    } else {
      return `${seconds}s`
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (!execution) {
    return (
      <div className="text-center py-12">
        <XCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">Execution not found</h3>
        <Link to="/workflows" className="text-blue-600 hover:text-blue-800">
          Return to workflows
        </Link>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Link
                to="/workflows"
                className="mr-4 p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 flex items-center">
                  {getStatusIcon(execution.status)}
                  <span className="ml-3">Execution Monitor</span>
                  {wsConnection && (
                    <span className="ml-2 inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      <span className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></span>
                      Live
                    </span>
                  )}
                </h1>
                <p className="text-gray-600 mt-1">
                  Workflow: {execution.workflow_id} • Execution: {execution.execution_id}
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              {/* Execution Controls */}
              {execution.status === 'running' && (
                <>
                  <button
                    onClick={() => pauseMutation.mutate()}
                    disabled={pauseMutation.isLoading}
                    className="px-4 py-2 bg-yellow-100 hover:bg-yellow-200 text-yellow-700 rounded-lg disabled:opacity-50 flex items-center"
                  >
                    <Pause className="w-4 h-4 mr-2" />
                    {pauseMutation.isLoading ? 'Pausing...' : 'Pause'}
                  </button>
                  <button
                    onClick={() => cancelMutation.mutate()}
                    disabled={cancelMutation.isLoading}
                    className="px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg disabled:opacity-50 flex items-center"
                  >
                    <Square className="w-4 h-4 mr-2" />
                    {cancelMutation.isLoading ? 'Cancelling...' : 'Cancel'}
                  </button>
                </>
              )}
              
              {execution.status === 'paused' && (
                <button
                  onClick={() => resumeMutation.mutate()}
                  disabled={resumeMutation.isLoading}
                  className="px-4 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-lg disabled:opacity-50 flex items-center"
                >
                  <Play className="w-4 h-4 mr-2" />
                  {resumeMutation.isLoading ? 'Resuming...' : 'Resume'}
                </button>
              )}
              
              {(execution.status === 'failed' || execution.status === 'completed') && (
                <button
                  onClick={() => retryMutation.mutate()}
                  disabled={retryMutation.isLoading}
                  className="px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg disabled:opacity-50 flex items-center"
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  {retryMutation.isLoading ? 'Retrying...' : 'Retry'}
                </button>
              )}
              
              <button className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg flex items-center">
                <Download className="w-4 h-4 mr-2" />
                Export
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Activity className="w-8 h-8 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Status</p>
                <p className="text-2xl font-semibold text-gray-900 capitalize">{execution.status}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Timer className="w-8 h-8 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Duration</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {metrics?.totalExecutionTime ? formatDuration(metrics.totalExecutionTime) : '--'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <CheckCircle className="w-8 h-8 text-emerald-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Completed</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {metrics?.completedStates || 0}/{metrics?.totalStates || 0}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <TrendingUp className="w-8 h-8 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Throughput</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {metrics?.throughput?.toFixed(2) || '0.00'} states/min
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        {metrics && (
          <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Execution Progress</h3>
              <span className="text-sm text-gray-600">
                {Math.round((metrics.completedStates / metrics.totalStates) * 100)}% Complete
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div 
                className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                style={{ width: `${(metrics.completedStates / metrics.totalStates) * 100}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-sm text-gray-600 mt-2">
              <span>{metrics.completedStates} completed</span>
              <span>{metrics.failedStates} failed</span>
              <span>{metrics.activeStates} running</span>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
          <div className="border-b border-gray-200">
            <nav className="flex">
              {[
                { id: 'overview', label: 'Overview', icon: Eye },
                { id: 'states', label: 'States', icon: Layers },
                { id: 'timeline', label: 'Timeline', icon: GitBranch },
                { id: 'logs', label: 'Logs', icon: FileText },
                { id: 'metrics', label: 'Metrics', icon: BarChart3 },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 bg-blue-50'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <tab.icon className="w-4 h-4 mr-2" />
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          <div className="p-6">
            {/* Overview Tab */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Execution Summary */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Execution Summary</h3>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <dl className="grid grid-cols-2 gap-4">
                      <div>
                        <dt className="text-sm font-medium text-gray-600">Started At</dt>
                        <dd className="text-sm text-gray-900">{execution.started_at ? new Date(execution.started_at).toLocaleString() : '--'}</dd>
                      </div>
                      <div>
                        <dt className="text-sm font-medium text-gray-600">Current State</dt>
                        <dd className="text-sm text-gray-900">{execution.current_state || '--'}</dd>
                      </div>
                      <div>
                        <dt className="text-sm font-medium text-gray-600">Error Rate</dt>
                        <dd className="text-sm text-gray-900">{metrics?.errorRate?.toFixed(1) || '0.0'}%</dd>
                      </div>
                      <div>
                        <dt className="text-sm font-medium text-gray-600">Avg State Time</dt>
                        <dd className="text-sm text-gray-900">{metrics?.avgStateTime ? formatDuration(metrics.avgStateTime) : '--'}</dd>
                      </div>
                    </dl>
                  </div>
                </div>

                {/* Resource Utilization */}
                {metrics?.resourceUtilization && (
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Resource Utilization</h3>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center">
                          <Cpu className="w-5 h-5 text-blue-600 mr-2" />
                          <span className="font-medium">CPU</span>
                        </div>
                        <div className="mt-2">
                          <div className="flex justify-between text-sm">
                            <span>Usage</span>
                            <span>{metrics.resourceUtilization.cpu.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                            <div 
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${metrics.resourceUtilization.cpu}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center">
                          <MemoryStick className="w-5 h-5 text-green-600 mr-2" />
                          <span className="font-medium">Memory</span>
                        </div>
                        <div className="mt-2">
                          <div className="flex justify-between text-sm">
                            <span>Usage</span>
                            <span>{metrics.resourceUtilization.memory.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                            <div 
                              className="bg-green-600 h-2 rounded-full"
                              style={{ width: `${metrics.resourceUtilization.memory}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>

                      <div className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center">
                          <Network className="w-5 h-5 text-purple-600 mr-2" />
                          <span className="font-medium">Network</span>
                        </div>
                        <div className="mt-2">
                          <div className="flex justify-between text-sm">
                            <span>I/O</span>
                            <span>{metrics.resourceUtilization.network.toFixed(1)} MB/s</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                            <div 
                              className="bg-purple-600 h-2 rounded-full"
                              style={{ width: `${Math.min(metrics.resourceUtilization.network * 10, 100)}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Recent Events */}
                {events && events.length > 0 && (
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Events</h3>
                    <div className="space-y-2">
                      {events.slice(0, 5).map((event) => (
                        <div key={event.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center">
                            <span className={`w-2 h-2 rounded-full mr-3 ${
                              event.level === 'error' ? 'bg-red-500' :
                              event.level === 'warning' ? 'bg-yellow-500' :
                              event.level === 'success' ? 'bg-green-500' : 'bg-blue-500'
                            }`}></span>
                            <span className="text-sm text-gray-900">{event.message}</span>
                            {event.state && (
                              <span className="ml-2 px-2 py-1 text-xs bg-gray-200 text-gray-700 rounded">
                                {event.state}
                              </span>
                            )}
                          </div>
                          <span className="text-xs text-gray-500">
                            {new Date(event.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* States Tab */}
            {activeTab === 'states' && states && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Workflow States</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
                  {states.map((state) => (
                    <div 
                      key={state.name} 
                      className={`border rounded-lg p-4 cursor-pointer transition-all ${
                        selectedState === state.name ? 'ring-2 ring-blue-500 border-blue-300' : 'hover:shadow-md'
                      } ${getStateStatusColor(state.status)}`}
                      onClick={() => setSelectedState(selectedState === state.name ? null : state.name)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{state.name}</h4>
                        {getStatusIcon(state.status)}
                      </div>
                      
                      <div className="text-sm space-y-1">
                        <div className="flex justify-between">
                          <span>Duration:</span>
                          <span>{state.duration ? formatDuration(state.duration) : '--'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Attempts:</span>
                          <span>{state.attempts}</span>
                        </div>
                        {state.error && (
                          <div className="text-red-600 text-xs mt-2 p-2 bg-red-50 rounded">
                            {state.error}
                          </div>
                        )}
                      </div>

                      {/* Expanded details */}
                      {selectedState === state.name && (
                        <div className="mt-4 pt-4 border-t space-y-3">
                          {state.dependencies.length > 0 && (
                            <div>
                              <span className="text-xs font-medium text-gray-600">Dependencies:</span>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {state.dependencies.map((dep) => (
                                  <span key={dep} className="text-xs px-2 py-1 bg-gray-200 rounded">
                                    {dep}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {state.transitions.length > 0 && (
                            <div>
                              <span className="text-xs font-medium text-gray-600">Transitions:</span>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {state.transitions.map((transition) => (
                                  <span key={transition} className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded">
                                    {transition}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {state.metrics && (
                            <div>
                              <span className="text-xs font-medium text-gray-600">Metrics:</span>
                              <div className="grid grid-cols-2 gap-2 mt-1 text-xs">
                                <div>CPU: {state.metrics.cpuUsage.toFixed(1)}%</div>
                                <div>Memory: {state.metrics.memoryUsage.toFixed(1)}MB</div>
                                <div>Network: {state.metrics.networkIO.toFixed(1)}KB/s</div>
                                <div>Retries: {state.metrics.retryCount}</div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Timeline Tab */}
            {activeTab === 'timeline' && states && (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Execution Timeline</h3>
                <div className="relative">
                  {/* Timeline line */}
                  <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-200"></div>
                  
                  <div className="space-y-6">
                    {states
                      .filter(state => state.startTime)
                      .sort((a, b) => new Date(a.startTime!).getTime() - new Date(b.startTime!).getTime())
                      .map((state, index) => (
                        <div key={state.name} className="relative flex items-start">
                          {/* Timeline dot */}
                          <div className={`relative z-10 flex items-center justify-center w-6 h-6 rounded-full border-2 ${
                            state.status === 'completed' ? 'bg-green-100 border-green-500' :
                            state.status === 'failed' ? 'bg-red-100 border-red-500' :
                            state.status === 'running' ? 'bg-blue-100 border-blue-500' :
                            'bg-gray-100 border-gray-300'
                          }`}>
                            {state.status === 'completed' ? (
                              <CheckCircle className="w-3 h-3 text-green-600" />
                            ) : state.status === 'failed' ? (
                              <XCircle className="w-3 h-3 text-red-600" />
                            ) : state.status === 'running' ? (
                              <Clock className="w-3 h-3 text-blue-600" />
                            ) : (
                              <Clock className="w-3 h-3 text-gray-400" />
                            )}
                          </div>

                          {/* Timeline content */}
                          <div className="ml-6 min-w-0 flex-1">
                            <div className="flex items-center justify-between">
                              <h4 className="text-sm font-medium text-gray-900">{state.name}</h4>
                              <time className="text-xs text-gray-500">
                                {new Date(state.startTime!).toLocaleTimeString()}
                              </time>
                            </div>
                            <div className="mt-1 text-sm text-gray-600">
                              Duration: {state.duration ? formatDuration(state.duration) : 'In progress...'}
                              {state.attempts > 1 && (
                                <span className="ml-2 text-yellow-600">• {state.attempts} attempts</span>
                              )}
                            </div>
                            {state.error && (
                              <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
                                {state.error}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}

            {/* Logs Tab */}
            {activeTab === 'logs' && (
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">Execution Logs</h3>
                  <div className="flex space-x-3">
                    <div className="relative">
                      <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                      <input
                        type="text"
                        value={logFilter}
                        onChange={(e) => setLogFilter(e.target.value)}
                        placeholder="Filter logs..."
                        className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                      />
                    </div>
                    <select
                      value={logLevel}
                      onChange={(e) => setLogLevel(e.target.value)}
                      className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                    >
                      <option value="all">All Levels</option>
                      <option value="debug">Debug</option>
                      <option value="info">Info</option>
                      <option value="warning">Warning</option>
                      <option value="error">Error</option>
                    </select>
                  </div>
                </div>

                <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-y-auto font-mono text-sm">
                  {filteredLogs.length > 0 ? (
                    filteredLogs.map((log) => (
                      <div key={log.id} className="flex items-start space-x-3 py-1 hover:bg-gray-800 px-2 rounded">
                        <span className="text-gray-400 text-xs whitespace-nowrap">
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                        <span className={`text-xs uppercase font-medium w-16 ${
                          log.level === 'error' ? 'text-red-400' :
                          log.level === 'warning' ? 'text-yellow-400' :
                          log.level === 'info' ? 'text-blue-400' :
                          'text-gray-400'
                        }`}>
                          {log.level}
                        </span>
                        {log.state && (
                          <span className="text-purple-400 text-xs">[{log.state}]</span>
                        )}
                        <span className="text-gray-100 flex-1">{log.message}</span>
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-400 text-center py-8">
                      No logs found matching your filter
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Metrics Tab */}
            {activeTab === 'metrics' && metrics && (
              <div className="space-y-6">
                <h3 className="text-lg font-semibold text-gray-900">Performance Metrics</h3>
                
                {/* Metrics Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6">
                    <div className="flex items-center">
                      <Timer className="w-8 h-8 text-blue-600" />
                      <div className="ml-4">
                        <p className="text-sm font-medium text-blue-600">Total Execution Time</p>
                        <p className="text-2xl font-bold text-blue-900">
                          {formatDuration(metrics.totalExecutionTime)}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6">
                    <div className="flex items-center">
                      <Target className="w-8 h-8 text-green-600" />
                      <div className="ml-4">
                        <p className="text-sm font-medium text-green-600">Success Rate</p>
                        <p className="text-2xl font-bold text-green-900">
                          {((metrics.completedStates / (metrics.completedStates + metrics.failedStates)) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6">
                    <div className="flex items-center">
                      <Zap className="w-8 h-8 text-purple-600" />
                      <div className="ml-4">
                        <p className="text-sm font-medium text-purple-600">Throughput</p>
                        <p className="text-2xl font-bold text-purple-900">
                          {metrics.throughput.toFixed(2)} states/min
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 rounded-lg p-6">
                    <div className="flex items-center">
                      <Clock className="w-8 h-8 text-yellow-600" />
                      <div className="ml-4">
                        <p className="text-sm font-medium text-yellow-600">Avg State Time</p>
                        <p className="text-2xl font-bold text-yellow-900">
                          {formatDuration(metrics.avgStateTime)}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-6">
                    <div className="flex items-center">
                      <AlertTriangle className="w-8 h-8 text-red-600" />
                      <div className="ml-4">
                        <p className="text-sm font-medium text-red-600">Error Rate</p>
                        <p className="text-2xl font-bold text-red-900">
                          {metrics.errorRate.toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-lg p-6">
                    <div className="flex items-center">
                      <Activity className="w-8 h-8 text-indigo-600" />
                      <div className="ml-4">
                        <p className="text-sm font-medium text-indigo-600">Active States</p>
                        <p className="text-2xl font-bold text-indigo-900">
                          {metrics.activeStates}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Detailed Resource Usage */}
                <div className="bg-gray-50 rounded-lg p-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-4">Resource Usage Details</h4>
                  <div className="space-y-4">
                    {states && states.filter(s => s.metrics).map((state) => (
                      <div key={state.name} className="bg-white rounded-lg p-4">
                        <div className="flex justify-between items-center mb-2">
                          <h5 className="font-medium text-gray-900">{state.name}</h5>
                          <span className={`px-2 py-1 text-xs rounded ${getStateStatusColor(state.status)}`}>
                            {state.status}
                          </span>
                        </div>
                        {state.metrics && (
                          <div className="grid grid-cols-4 gap-4 text-sm">
                            <div>
                              <span className="text-gray-600">CPU:</span>
                              <span className="ml-1 font-medium">{state.metrics.cpuUsage.toFixed(1)}%</span>
                            </div>
                            <div>
                              <span className="text-gray-600">Memory:</span>
                              <span className="ml-1 font-medium">{state.metrics.memoryUsage.toFixed(1)}MB</span>
                            </div>
                            <div>
                              <span className="text-gray-600">Network:</span>
                              <span className="ml-1 font-medium">{state.metrics.networkIO.toFixed(1)}KB/s</span>
                            </div>
                            <div>
                              <span className="text-gray-600">Duration:</span>
                              <span className="ml-1 font-medium">{formatDuration(state.metrics.executionTime)}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}