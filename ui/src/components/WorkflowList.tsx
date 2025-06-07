import { useState, useMemo, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  Plus, Play, Trash2, Clock, CheckCircle, XCircle, Code, Edit,
  Search, BarChart3, Activity, TrendingUp, Zap, Settings, Monitor, AlertCircle,
  Pause, Square, RotateCcw, Eye, Filter, Grid, List, FileText, Download,
  ArrowUp, ArrowDown, Minus, Calendar, Users, Target, X
} from 'lucide-react'
// Navigation handled via onClick handlers
import WorkflowYamlCreator from './WorkflowYamlCreator'

// API client for actual backend calls
const API_BASE_URL = (() => {
  // Try different environment variable approaches
  if (typeof window !== 'undefined') {
    // Browser environment
    return window.location.origin.includes('localhost') 
      ? 'http://localhost:8000' 
      : window.location.origin
  }
  // Fallback
  return 'http://localhost:8000'
})()

const workflowApi = {
  listWorkflows: async (params = {}) => {
    const searchParams = new URLSearchParams(params)
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows?${searchParams}`)
    if (!response.ok) throw new Error('Failed to fetch workflows')
    return response.json()
  },
  
  deleteWorkflow: async (id) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${id}`, {
      method: 'DELETE'
    })
    if (!response.ok) throw new Error('Failed to delete workflow')
    return response.json()
  },
  
  executeWorkflow: async (id, params = {}) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${id}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
    if (!response.ok) throw new Error('Failed to execute workflow')
    return response.json()
  },
  
  scheduleWorkflow: async (id, schedule) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${id}/schedule`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(schedule)
    })
    if (!response.ok) throw new Error('Failed to schedule workflow')
    return response.json()
  },
  
  unscheduleWorkflow: async (id) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${id}/schedule`, {
      method: 'DELETE'
    })
    if (!response.ok) throw new Error('Failed to unschedule workflow')
    return response.json()
  },
  
  getWorkflowStatus: async (id) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${id}/status`)
    if (!response.ok) throw new Error('Failed to fetch workflow status')
    return response.json()
  },
  
  generateCodeFromWorkflow: async (id) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${id}/generate-code`, {
      method: 'POST'
    })
    if (!response.ok) throw new Error('Failed to generate code')
    return response.json()
  },
  
  getCurrentExecution: async (workflowId) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/current-execution`)
    if (!response.ok) throw new Error('Failed to fetch current execution')
    return response.json()
  },
  
  listExecutions: async (workflowId, params = {}) => {
    const searchParams = new URLSearchParams(params)
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/executions?${searchParams}`)
    if (!response.ok) throw new Error('Failed to fetch executions')
    return response.json()
  },
  
  getExecution: async (workflowId, executionId) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/executions/${executionId}`)
    if (!response.ok) throw new Error('Failed to fetch execution')
    return response.json()
  },
  
  getExecutionStates: async (workflowId, executionId) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/executions/${executionId}/states`)
    if (!response.ok) throw new Error('Failed to fetch execution states')
    return response.json()
  },
  
  getExecutionMetrics: async (workflowId, executionId) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/executions/${executionId}/metrics`)
    if (!response.ok) throw new Error('Failed to fetch execution metrics')
    return response.json()
  },
  
  getExecutionEvents: async (workflowId, executionId, params = {}) => {
    const searchParams = new URLSearchParams(params)
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/executions/${executionId}/events?${searchParams}`)
    if (!response.ok) throw new Error('Failed to fetch execution events')
    return response.json()
  },
  
  pauseExecution: async (workflowId, executionId) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/executions/${executionId}/pause`, {
      method: 'POST'
    })
    if (!response.ok) throw new Error('Failed to pause execution')
    return response.json()
  },
  
  resumeExecution: async (workflowId, executionId) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/executions/${executionId}/resume`, {
      method: 'POST'
    })
    if (!response.ok) throw new Error('Failed to resume execution')
    return response.json()
  },
  
  cancelExecution: async (workflowId, executionId) => {
    const response = await fetch(`${API_BASE_URL}/api/v1/workflows/${workflowId}/executions/${executionId}/cancel`, {
      method: 'POST'
    })
    if (!response.ok) throw new Error('Failed to cancel execution')
    return response.json()
  },
  
  testConnection: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/health`)
      return response.ok
    } catch {
      return false
    }
  }
}

// Schedule Modal Component
function ScheduleModal({ workflow, isOpen, onClose, onSchedule }) {
  const [scheduleType, setScheduleType] = useState('interval')
  const [intervalValue, setIntervalValue] = useState(5)
  const [intervalUnit, setIntervalUnit] = useState('minutes')
  const [cronExpression, setCronExpression] = useState('0 */1 * * *')
  const [startTime, setStartTime] = useState('')
  const [endTime, setEndTime] = useState('')
  const [isEnabled, setIsEnabled] = useState(true)

  if (!isOpen) return null

  const handleSubmit = (e) => {
    e.preventDefault()
    
    let schedule = {
      enabled: isEnabled,
      start_time: startTime || null,
      end_time: endTime || null
    }

    if (scheduleType === 'interval') {
      const intervalMs = {
        seconds: intervalValue * 1000,
        minutes: intervalValue * 60 * 1000,
        hours: intervalValue * 60 * 60 * 1000,
        days: intervalValue * 24 * 60 * 60 * 1000
      }[intervalUnit]
      
      schedule.type = 'interval'
      schedule.interval_ms = intervalMs
    } else {
      schedule.type = 'cron'
      schedule.cron_expression = cronExpression
    }

    onSchedule(schedule)
  }

  const getIntervalPreview = () => {
    if (scheduleType !== 'interval') return ''
    return `Runs every ${intervalValue} ${intervalUnit}`
  }

  const getCronPreview = () => {
    if (scheduleType !== 'cron') return ''
    const descriptions = {
      '0 */1 * * *': 'Every hour',
      '0 */6 * * *': 'Every 6 hours', 
      '0 0 * * *': 'Daily at midnight',
      '0 0 * * 0': 'Weekly on Sunday',
      '0 0 1 * *': 'Monthly on 1st'
    }
    return descriptions[cronExpression] || 'Custom schedule'
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-md w-full max-h-[90vh] overflow-hidden">
        <div className="bg-gray-50 px-6 py-4 border-b flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-900">Schedule Workflow</h2>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          <div>
            <h3 className="font-medium text-gray-900 mb-2">{workflow?.name}</h3>
            <p className="text-sm text-gray-600">{workflow?.description}</p>
          </div>

          <div>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={isEnabled}
                onChange={(e) => setIsEnabled(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm font-medium text-gray-700">Enable schedule</span>
            </label>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Schedule Type
            </label>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  value="interval"
                  checked={scheduleType === 'interval'}
                  onChange={(e) => setScheduleType(e.target.value)}
                  className="text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">Interval (every X time)</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  value="cron"
                  checked={scheduleType === 'cron'}
                  onChange={(e) => setScheduleType(e.target.value)}
                  className="text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">Cron expression</span>
              </label>
            </div>
          </div>

          {scheduleType === 'interval' && (
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Every
                </label>
                <input
                  type="number"
                  min="1"
                  value={intervalValue}
                  onChange={(e) => setIntervalValue(parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Unit
                </label>
                <select
                  value={intervalUnit}
                  onChange={(e) => setIntervalUnit(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="seconds">Seconds</option>
                  <option value="minutes">Minutes</option>
                  <option value="hours">Hours</option>
                  <option value="days">Days</option>
                </select>
              </div>
            </div>
          )}

          {scheduleType === 'cron' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Cron Expression
              </label>
              <select
                value={cronExpression}
                onChange={(e) => setCronExpression(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 mb-2"
              >
                <option value="0 */1 * * *">Every hour</option>
                <option value="0 */6 * * *">Every 6 hours</option>
                <option value="0 0 * * *">Daily at midnight</option>
                <option value="0 0 * * 0">Weekly on Sunday</option>
                <option value="0 0 1 * *">Monthly on 1st</option>
              </select>
              <input
                type="text"
                value={cronExpression}
                onChange={(e) => setCronExpression(e.target.value)}
                placeholder="Custom cron expression"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}

          <div className="bg-blue-50 p-3 rounded-lg">
            <p className="text-sm text-blue-800 font-medium">
              {scheduleType === 'interval' ? getIntervalPreview() : getCronPreview()}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Start Time (optional)
              </label>
              <input
                type="datetime-local"
                value={startTime}
                onChange={(e) => setStartTime(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                End Time (optional)
              </label>
              <input
                type="datetime-local"
                value={endTime}
                onChange={(e) => setEndTime(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="flex justify-end space-x-3 pt-4 border-t">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Schedule
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
function ExecutionMonitor({ workflowId, executionId, onClose }) {
  const [wsConnection, setWsConnection] = useState(null)
  
  const { data: execution, isLoading: executionLoading, refetch: refetchExecution } = useQuery({
    queryKey: ['execution', workflowId, executionId],
    queryFn: () => workflowApi.getExecution(workflowId, executionId),
    refetchInterval: 2000,
    enabled: !!(workflowId && executionId)
  })

  const { data: states, refetch: refetchStates } = useQuery({
    queryKey: ['execution-states', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionStates(workflowId, executionId),
    refetchInterval: 1000,
    enabled: !!(workflowId && executionId)
  })

  const { data: metrics } = useQuery({
    queryKey: ['execution-metrics', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionMetrics(workflowId, executionId),
    refetchInterval: 2000,
    enabled: !!(workflowId && executionId)
  })

  const { data: events } = useQuery({
    queryKey: ['execution-events', workflowId, executionId],
    queryFn: () => workflowApi.getExecutionEvents(workflowId, executionId, { limit: 50 }),
    refetchInterval: 1000,
    enabled: !!(workflowId && executionId)
  })

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!workflowId || !executionId) return

    const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/api/v1/workflows/${workflowId}/executions/${executionId}/ws`
    
    try {
      const ws = new WebSocket(wsUrl)
      
      ws.onopen = () => {
        console.log('📡 WebSocket connected for real-time updates')
        setWsConnection(ws)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          console.log('📨 Real-time update:', data)
          
          // Trigger refetch of data
          refetchExecution()
          refetchStates()
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
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
    }
  }, [workflowId, executionId, refetchExecution, refetchStates])

  const handleControlAction = async (action) => {
    try {
      switch (action) {
        case 'pause':
          await workflowApi.pauseExecution(workflowId, executionId)
          break
        case 'resume':
          await workflowApi.resumeExecution(workflowId, executionId)
          break
        case 'cancel':
          await workflowApi.cancelExecution(workflowId, executionId)
          break
      }
      refetchExecution()
    } catch (error) {
      console.error(`Failed to ${action} execution:`, error)
      alert(`Failed to ${action} execution: ${error.message}`)
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running':
        return <Activity className="w-4 h-4 text-blue-500 animate-pulse" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />
      case 'paused':
        return <Pause className="w-4 h-4 text-yellow-500" />
      case 'cancelled':
        return <Square className="w-4 h-4 text-gray-500" />
      default:
        return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const formatDuration = (ms) => {
    if (!ms) return '--'
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    
    if (hours > 0) return `${hours}h ${minutes % 60}m ${seconds % 60}s`
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`
    return `${seconds}s`
  }

  if (executionLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RotateCcw className="w-8 h-8 animate-spin text-blue-600" />
        <span className="ml-2 text-gray-600">Loading execution...</span>
      </div>
    )
  }

  if (!execution) {
    return (
      <div className="text-center py-12">
        <XCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">Execution not found</h3>
        <button onClick={onClose} className="text-blue-600 hover:text-blue-800">
          Close Monitor
        </button>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl shadow-lg max-w-6xl mx-auto max-h-[90vh] overflow-hidden flex flex-col">
      {/* Header */}
      <div className="bg-gray-50 px-6 py-4 border-b flex justify-between items-center">
        <div className="flex items-center">
          {getStatusIcon(execution.status)}
          <h2 className="text-xl font-bold text-gray-900 ml-3">
            Execution Monitor
          </h2>
          {wsConnection && (
            <span className="ml-3 inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              <span className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></span>
              Live
            </span>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          {/* Execution Controls */}
          {execution.status === 'running' && (
            <>
              <button
                onClick={() => handleControlAction('pause')}
                className="p-2 text-yellow-600 hover:bg-yellow-50 rounded-lg transition-colors"
                title="Pause Execution"
              >
                <Pause className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleControlAction('cancel')}
                className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                title="Cancel Execution"
              >
                <Square className="w-4 h-4" />
              </button>
            </>
          )}
          
          {execution.status === 'paused' && (
            <button
              onClick={() => handleControlAction('resume')}
              className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
              title="Resume Execution"
            >
              <Play className="w-4 h-4" />
            </button>
          )}
          
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
            title="Close Monitor"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Execution Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center">
              <Activity className="w-6 h-6 text-blue-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-blue-600">Status</p>
                <p className="text-lg font-bold text-blue-900 capitalize">{execution.status}</p>
              </div>
            </div>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <div className="flex items-center">
              <CheckCircle className="w-6 h-6 text-green-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-green-600">Completed</p>
                <p className="text-lg font-bold text-green-900">
                  {metrics?.completedStates || 0}/{metrics?.totalStates || 0}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-purple-50 rounded-lg p-4">
            <div className="flex items-center">
              <Clock className="w-6 h-6 text-purple-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-purple-600">Duration</p>
                <p className="text-lg font-bold text-purple-900">
                  {formatDuration(metrics?.totalExecutionTime)}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-orange-50 rounded-lg p-4">
            <div className="flex items-center">
              <TrendingUp className="w-6 h-6 text-orange-600" />
              <div className="ml-3">
                <p className="text-sm font-medium text-orange-600">Throughput</p>
                <p className="text-lg font-bold text-orange-900">
                  {metrics?.throughput?.toFixed(1) || '0.0'} states/min
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        {metrics && metrics.totalStates > 0 && (
          <div className="bg-gray-50 rounded-lg p-4 mb-6">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-sm font-medium text-gray-700">Execution Progress</h3>
              <span className="text-sm text-gray-600">
                {Math.round((metrics.completedStates / metrics.totalStates) * 100)}% Complete
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${(metrics.completedStates / metrics.totalStates) * 100}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>{metrics.completedStates} completed</span>
              <span>{metrics.failedStates || 0} failed</span>
              <span>{metrics.activeStates || 0} running</span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* States */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Workflow States</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {states && states.length > 0 ? (
                states.map((state) => (
                  <div key={state.name} className="bg-gray-50 rounded-lg p-3">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-gray-900">{state.name}</span>
                      {getStatusIcon(state.status)}
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      Duration: {formatDuration(state.duration)}
                      {state.attempts > 1 && (
                        <span className="ml-2 text-yellow-600">• {state.attempts} attempts</span>
                      )}
                    </div>
                    {state.error && (
                      <div className="text-xs text-red-600 mt-1 bg-red-50 p-2 rounded">
                        {state.error}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Activity className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                  <p>No states available</p>
                </div>
              )}
            </div>
          </div>

          {/* Recent Events */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Events</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {events && events.length > 0 ? (
                events.slice(0, 10).map((event, index) => (
                  <div key={event.id || index} className="bg-gray-50 rounded-lg p-3">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="text-sm text-gray-900">{event.message}</p>
                        {event.state && (
                          <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded mt-1 inline-block">
                            {event.state}
                          </span>
                        )}
                      </div>
                      <span className={`w-2 h-2 rounded-full ml-2 mt-1 ${
                        event.level === 'error' ? 'bg-red-500' :
                        event.level === 'warning' ? 'bg-yellow-500' :
                        event.level === 'success' ? 'bg-green-500' : 'bg-blue-500'
                      }`}></span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <FileText className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                  <p>No events available</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function WorkflowList() {
  const [showCreator, setShowCreator] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [viewMode, setViewMode] = useState('grid')
  const [sortBy, setSortBy] = useState('created_at')
  const [sortOrder, setSortOrder] = useState('desc')
  const [executingWorkflow, setExecutingWorkflow] = useState(null)
  const [monitoringExecution, setMonitoringExecution] = useState(null)
  const [scheduleModal, setScheduleModal] = useState({ isOpen: false, workflow: null })
  const [realTimeStatuses, setRealTimeStatuses] = useState({})

  // More frequent polling for real-time status updates
  const { data: workflowsData, isLoading, error, refetch } = useQuery({
    queryKey: ['workflows'],
    queryFn: () => workflowApi.listWorkflows(),
    refetchInterval: 2000 // Check every 2 seconds for real-time updates
  })

  const workflows = workflowsData?.workflows || []

  // Poll individual workflow statuses for active ones
  useEffect(() => {
    const activeWorkflows = workflows.filter(w => 
      w.status === 'running' || w.status === 'starting' || executingWorkflow === w.workflow_id
    )
    
    if (activeWorkflows.length === 0) return

    const pollStatuses = async () => {
      const statusPromises = activeWorkflows.map(async (workflow) => {
        try {
          const status = await workflowApi.getWorkflowStatus(workflow.workflow_id)
          return { id: workflow.workflow_id, status: status.status, lastUpdate: Date.now() }
        } catch (error) {
          console.error(`Failed to get status for workflow ${workflow.workflow_id}:`, error)
          return { id: workflow.workflow_id, status: workflow.status, lastUpdate: Date.now() }
        }
      })
      
      const statuses = await Promise.all(statusPromises)
      const statusMap = statuses.reduce((acc, { id, status, lastUpdate }) => {
        acc[id] = { status, lastUpdate }
        return acc
      }, {})
      
      setRealTimeStatuses(prev => ({ ...prev, ...statusMap }))
    }

    // Poll immediately, then every 1 second
    pollStatuses()
    const interval = setInterval(pollStatuses, 1000)
    
    return () => clearInterval(interval)
  }, [workflows, executingWorkflow])

  // Handle the case where the API returns a different structure
  const normalizedWorkflows = useMemo(() => {
    if (!Array.isArray(workflows)) {
      console.warn('Workflows data is not an array:', workflows)
      return []
    }
    
    return workflows.map(workflow => {
      // Use real-time status if available, otherwise use API status
      const realtimeStatus = realTimeStatuses[workflow.workflow_id]
      const currentStatus = realtimeStatus?.status || workflow.status || 'unknown'
      
      return {
        ...workflow,
        // Ensure required fields exist with defaults
        workflow_id: workflow.workflow_id || workflow.id,
        name: workflow.name || 'Unnamed Workflow',
        description: workflow.description || 'No description available',
        status: currentStatus,
        created_at: workflow.created_at || new Date().toISOString(),
        states_count: workflow.states_count || 0,
        // Add scheduling info
        is_scheduled: workflow.is_scheduled || false,
        schedule_info: workflow.schedule_info || null
      }
    })
  }, [workflows, realTimeStatuses])

  // Filter and sort workflows (API handles search and status, we handle sorting)
  const filteredWorkflows = useMemo(() => {
    let filtered = [...normalizedWorkflows]

    // Sort workflows
    filtered.sort((a, b) => {
      let aVal = a[sortBy]
      let bVal = b[sortBy]
      
      if (sortBy === 'created_at' || sortBy === 'updated_at' || sortBy === 'last_execution') {
        aVal = new Date(aVal || 0).getTime()
        bVal = new Date(bVal || 0).getTime()
      }
      
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        aVal = aVal.toLowerCase()
        bVal = bVal.toLowerCase()
      }
      
      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1
      } else {
        return aVal < bVal ? 1 : -1
      }
    })

    return filtered
  }, [normalizedWorkflows, sortBy, sortOrder])

  // Calculate global stats (handle missing fields from real DB)
  const stats = useMemo(() => {
    const total = normalizedWorkflows.length
    const running = normalizedWorkflows.filter(w => w.status === 'running').length
    const failed = normalizedWorkflows.filter(w => w.status === 'failed').length
    const paused = normalizedWorkflows.filter(w => w.status === 'paused').length
    
    // Handle missing executions_today field
    const totalExecutions = normalizedWorkflows.reduce((sum, w) => {
      return sum + (w.executions_today || 0)
    }, 0)
    
    // Handle missing success_rate field
    const workflowsWithSuccessRate = normalizedWorkflows.filter(w => typeof w.success_rate === 'number')
    const avgSuccessRate = workflowsWithSuccessRate.length > 0 
      ? workflowsWithSuccessRate.reduce((sum, w) => sum + w.success_rate, 0) / workflowsWithSuccessRate.length 
      : 0

    return { total, running, failed, paused, totalExecutions, avgSuccessRate }
  }, [normalizedWorkflows])

  const handleAction = async (action, workflowId, workflowName) => {
    try {
      switch (action) {
        case 'execute':
          setExecutingWorkflow(workflowId)
          // Update real-time status immediately for UI feedback
          setRealTimeStatuses(prev => ({ 
            ...prev, 
            [workflowId]: { status: 'starting', lastUpdate: Date.now() } 
          }))
          
          try {
            const result = await workflowApi.executeWorkflow(workflowId)
            console.log('✅ Workflow execution started:', result)
            
            // Update status to running
            setRealTimeStatuses(prev => ({ 
              ...prev, 
              [workflowId]: { status: 'running', lastUpdate: Date.now() } 
            }))
            
            // Automatically open monitoring for the new execution
            setMonitoringExecution({
              workflowId: workflowId,
              executionId: result.execution_id,
              workflowName: workflowName
            })
            
            refetch() // Refresh the list to show updated status
          } finally {
            setExecutingWorkflow(null)
          }
          break

        case 'pause':
          // For running workflows, we need to find the current execution and pause it
          try {
            const currentExecution = await workflowApi.getCurrentExecution(workflowId)
            if (currentExecution.execution_id) {
              await workflowApi.pauseExecution(workflowId, currentExecution.execution_id)
              setRealTimeStatuses(prev => ({ 
                ...prev, 
                [workflowId]: { status: 'paused', lastUpdate: Date.now() } 
              }))
              console.log('✅ Workflow paused successfully')
              refetch()
            }
          } catch (error) {
            console.error('Failed to pause workflow:', error)
            alert(`Failed to pause workflow: ${error.message}`)
          }
          break

        case 'resume':
          // For paused workflows, resume the current execution
          try {
            const currentExecution = await workflowApi.getCurrentExecution(workflowId)
            if (currentExecution.execution_id) {
              await workflowApi.resumeExecution(workflowId, currentExecution.execution_id)
              setRealTimeStatuses(prev => ({ 
                ...prev, 
                [workflowId]: { status: 'running', lastUpdate: Date.now() } 
              }))
              console.log('✅ Workflow resumed successfully')
              refetch()
            }
          } catch (error) {
            console.error('Failed to resume workflow:', error)
            alert(`Failed to resume workflow: ${error.message}`)
          }
          break

        case 'stop':
          // For running workflows, stop/cancel the current execution
          if (confirm(`Are you sure you want to stop the running workflow "${workflowName}"?`)) {
            try {
              const currentExecution = await workflowApi.getCurrentExecution(workflowId)
              if (currentExecution.execution_id) {
                await workflowApi.cancelExecution(workflowId, currentExecution.execution_id)
                setRealTimeStatuses(prev => ({ 
                  ...prev, 
                  [workflowId]: { status: 'cancelled', lastUpdate: Date.now() } 
                }))
                console.log('✅ Workflow stopped successfully')
                refetch()
              }
            } catch (error) {
              console.error('Failed to stop workflow:', error)
              alert(`Failed to stop workflow: ${error.message}`)
            }
          }
          break
          
        case 'schedule':
          // Open scheduling modal
          const targetWorkflow = normalizedWorkflows.find(w => w.workflow_id === workflowId)
          setScheduleModal({ 
            isOpen: true, 
            workflow: targetWorkflow 
          })
          break
          
        case 'unschedule':
          if (confirm(`Are you sure you want to unschedule workflow "${workflowName}"?`)) {
            try {
              await workflowApi.unscheduleWorkflow(workflowId)
              console.log('✅ Workflow unscheduled successfully')
              refetch()
            } catch (error) {
              console.error('Failed to unschedule workflow:', error)
              alert(`Failed to unschedule workflow: ${error.message}`)
            }
          }
          break
          
        case 'monitor':
          // Show execution history and monitoring for this workflow
          try {
            // First, try to get the most recent execution
            const executions = await workflowApi.listExecutions(workflowId, { limit: 1 })
            if (executions.length > 0) {
              // Open monitor for the most recent execution
              setMonitoringExecution({
                workflowId: workflowId,
                executionId: executions[0].execution_id,
                workflowName: workflowName
              })
            } else {
              // No executions found, start a new one to monitor
              setExecutingWorkflow(workflowId)
              try {
                const result = await workflowApi.executeWorkflow(workflowId)
                setMonitoringExecution({
                  workflowId: workflowId,
                  executionId: result.execution_id,
                  workflowName: workflowName
                })
              } finally {
                setExecutingWorkflow(null)
              }
            }
          } catch (error) {
            console.error('Failed to open monitor:', error)
            alert(`Failed to open monitor: ${error.message}`)
          }
          break
          
        case 'generateCode':
          const codeResult = await workflowApi.generateCodeFromWorkflow(workflowId)
          if (codeResult.success && codeResult.zip_content) {
            // Create download link for zip file
            const byteCharacters = atob(codeResult.zip_content)
            const byteNumbers = new Array(byteCharacters.length)
            for (let i = 0; i < byteCharacters.length; i++) {
              byteNumbers[i] = byteCharacters.charCodeAt(i)
            }
            const byteArray = new Uint8Array(byteNumbers)
            const blob = new Blob([byteArray], { type: 'application/zip' })
            
            const url = window.URL.createObjectURL(blob)
            const link = document.createElement('a')
            link.href = url
            link.download = `${workflowName}-generated.zip`
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            window.URL.revokeObjectURL(url)
            
            console.log('✅ Code generated and downloaded successfully')
          } else {
            alert(`Code generation failed: ${codeResult.message || 'Unknown error'}`)
          }
          break
          
        case 'delete':
          if (confirm(`Are you sure you want to delete workflow "${workflowName}"?\n\nThis action cannot be undone.`)) {
            await workflowApi.deleteWorkflow(workflowId)
            console.log('✅ Workflow deleted successfully')
            refetch()
          }
          break
      }
    } catch (error) {
      console.error(`Action ${action} failed:`, error)
      alert(`Failed to ${action} workflow: ${error.message}`)
      setExecutingWorkflow(null)
      // Reset status on error
      if (action === 'execute') {
        setRealTimeStatuses(prev => ({ 
          ...prev, 
          [workflowId]: { status: 'failed', lastUpdate: Date.now() } 
        }))
      }
    }
  }

  const handleScheduleWorkflow = async (schedule) => {
    try {
      await workflowApi.scheduleWorkflow(scheduleModal.workflow.workflow_id, schedule)
      console.log('✅ Workflow scheduled successfully')
      setScheduleModal({ isOpen: false, workflow: null })
      refetch()
    } catch (error) {
      console.error('Failed to schedule workflow:', error)
      alert(`Failed to schedule workflow: ${error.message}`)
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running':
        return <Activity className="w-3 h-3 text-blue-500 animate-pulse" />
      case 'starting':
        return <RotateCcw className="w-3 h-3 text-blue-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="w-3 h-3 text-green-500" />
      case 'failed':
        return <XCircle className="w-3 h-3 text-red-500" />
      case 'paused':
        return <Pause className="w-3 h-3 text-yellow-500" />
      case 'cancelled':
        return <Square className="w-3 h-3 text-gray-500" />
      case 'idle':
        return <Clock className="w-3 h-3 text-gray-400" />
      default:
        return <Clock className="w-3 h-3 text-gray-400" />
    }
  }

  const getStatusBadge = (status, isScheduled = false) => {
    const configs = {
      running: 'bg-blue-100 text-blue-800 border-blue-200',
      starting: 'bg-blue-100 text-blue-800 border-blue-200',
      completed: 'bg-green-100 text-green-800 border-green-200',
      failed: 'bg-red-100 text-red-800 border-red-200',
      paused: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      cancelled: 'bg-gray-100 text-gray-800 border-gray-200',
      idle: 'bg-gray-100 text-gray-800 border-gray-200'
    }
    
    return (
      <div className="flex items-center gap-2">
        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border ${configs[status] || configs.idle}`}>
          {getStatusIcon(status)}
          {status}
        </span>
        {isScheduled && (
          <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800 border border-purple-200">
            <Calendar className="w-3 h-3" />
            Scheduled
          </span>
        )}
      </div>
    )
  }

  const formatDuration = (seconds) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`
    return `${(seconds / 3600).toFixed(1)}h`
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-center">
          <XCircle className="w-6 h-6 text-red-400 mr-3" />
          <div>
            <h3 className="text-lg font-medium text-red-800">Failed to load workflows</h3>
            <p className="text-red-700 mt-1">
              {error?.message || 'Unable to connect to the workflow service'}
            </p>
          </div>
        </div>
        <div className="mt-4 space-x-3">
          <button
            onClick={() => refetch()}
            className="bg-red-100 hover:bg-red-200 text-red-800 px-4 py-2 rounded-lg transition-colors inline-flex items-center"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Retry
          </button>
          <button
            onClick={async () => {
              const isConnected = await workflowApi.testConnection()
              alert(isConnected ? 'API connection successful!' : 'API connection failed!')
            }}
            className="bg-blue-100 hover:bg-blue-200 text-blue-800 px-4 py-2 rounded-lg transition-colors inline-flex items-center"
          >
            Test Connection
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header with stats */}
      <div className="bg-white border-b shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-between items-start mb-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Workflows</h1>
              <p className="text-gray-600 mt-1">Design, monitor, and manage your automation workflows</p>
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={() => setShowCreator(true)}
                className="bg-white hover:bg-gray-50 text-gray-700 px-4 py-2 rounded-lg border border-gray-300 flex items-center transition-colors"
                title="Quick Create"
              >
                <Edit className="w-4 h-4 mr-2" />
                Quick Create
              </button>
              
              <button
                onClick={() => console.log('Navigate to workflow creator')}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center transition-colors"
                title="New Workflow"
              >
                <Plus className="w-4 h-4 mr-2" />
                New Workflow
              </button>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4">
              <div className="flex items-center">
                <Activity className="w-8 h-8 text-blue-600" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-blue-600">Total</p>
                  <p className="text-2xl font-bold text-blue-900">{stats.total}</p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4">
              <div className="flex items-center">
                <CheckCircle className="w-8 h-8 text-green-600" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-green-600">Running</p>
                  <p className="text-2xl font-bold text-green-900">{stats.running}</p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-4">
              <div className="flex items-center">
                <XCircle className="w-8 h-8 text-red-600" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-red-600">Failed</p>
                  <p className="text-2xl font-bold text-red-900">{stats.failed}</p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 rounded-lg p-4">
              <div className="flex items-center">
                <Pause className="w-8 h-8 text-yellow-600" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-yellow-600">Paused</p>
                  <p className="text-2xl font-bold text-yellow-900">{stats.paused}</p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4">
              <div className="flex items-center">
                <Zap className="w-8 h-8 text-purple-600" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-purple-600">Executions</p>
                  <p className="text-2xl font-bold text-purple-900">{stats.totalExecutions}</p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-lg p-4">
              <div className="flex items-center">
                <Target className="w-8 h-8 text-emerald-600" />
                <div className="ml-3">
                  <p className="text-sm font-medium text-emerald-600">Success</p>
                  <p className="text-2xl font-bold text-emerald-900">{stats.avgSuccessRate.toFixed(1)}%</p>
                </div>
              </div>
            </div>
          </div>

          {/* Search and filters */}
          <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
            <div className="flex flex-1 items-center space-x-4">
              <div className="relative flex-1 max-w-md">
                <Search className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search workflows..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
              >
                <option value="all">All Status</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="paused">Paused</option>
                <option value="idle">Idle</option>
              </select>

              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
              >
                <option value="created_at">Created Date</option>
                <option value="updated_at">Updated Date</option>
                <option value="name">Name</option>
                <option value="status">Status</option>
                <option value="states_count">States Count</option>
              </select>

              <button
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                title={`Sort ${sortOrder === 'asc' ? 'Descending' : 'Ascending'}`}
              >
                {sortOrder === 'asc' ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />}
              </button>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded-lg transition-colors ${viewMode === 'grid' ? 'bg-blue-100 text-blue-600' : 'text-gray-600 hover:bg-gray-100'}`}
                title="Grid View"
              >
                <Grid className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded-lg transition-colors ${viewMode === 'list' ? 'bg-blue-100 text-blue-600' : 'text-gray-600 hover:bg-gray-100'}`}
                title="List View"
              >
                <List className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Workflows Grid/List */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {filteredWorkflows.length === 0 ? (
          <div className="text-center py-16">
            <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-medium text-gray-900 mb-2">No workflows found</h3>
            <p className="text-gray-600 mb-6">
              {searchQuery || statusFilter !== 'all' 
                ? 'Try adjusting your search or filters' 
                : 'Get started by creating your first workflow'
              }
            </p>
            
            <div className="flex justify-center space-x-3">
              <button
                onClick={() => setShowCreator(true)}
                className="bg-white hover:bg-gray-50 text-gray-700 px-4 py-2 rounded-lg border border-gray-300 inline-flex items-center transition-colors"
              >
                <Edit className="w-4 h-4 mr-2" />
                Quick Create
              </button>
              
              <button
                onClick={() => console.log('Navigate to full creator')}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg inline-flex items-center transition-colors"
              >
                <Plus className="w-4 h-4 mr-2" />
                Full Creator
              </button>
            </div>
          </div>
        ) : (
          <div className={viewMode === 'grid' 
            ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6" 
            : "space-y-4"
          }>
            {filteredWorkflows.map((workflow) => (
              <div key={workflow.workflow_id} className={`bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200 ${
                viewMode === 'list' ? 'flex items-center p-4' : 'p-6'
              }`}>
                {viewMode === 'grid' ? (
                  <>
                    {/* Grid Card Layout */}
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1 min-w-0">
                        <h3 className="text-lg font-semibold text-gray-900 truncate mb-1">
                          {workflow.name}
                        </h3>
                        <p className="text-sm text-gray-600 line-clamp-2 mb-2">
                          {workflow.description}
                        </p>
                        <div className="flex items-center gap-2">
                          {getStatusBadge(workflow.status, workflow.is_scheduled)}
                          {workflow.status === 'running' && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 border border-blue-200">
                              <span className="w-2 h-2 bg-blue-400 rounded-full mr-1 animate-pulse"></span>
                              Live
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
                      <div>
                        <span className="text-gray-500">States:</span>
                        <span className="font-medium ml-1">{workflow.states_count || 0}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Success:</span>
                        <span className="font-medium ml-1">
                          {typeof workflow.success_rate === 'number' ? `${workflow.success_rate.toFixed(1)}%` : 'N/A'}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">Runs:</span>
                        <span className="font-medium ml-1">{workflow.executions_today || 0}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Duration:</span>
                        <span className="font-medium ml-1">
                          {workflow.avg_duration ? formatDuration(workflow.avg_duration) : 'N/A'}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="text-xs text-gray-500">
                        {workflow.author ? `by ${workflow.author}` : 'Unknown author'}
                      </div>
                      
                      <div className="flex space-x-1">
                        {/* Dynamic Action Button based on status */}
                        {workflow.status === 'running' ? (
                          <>
                            <button
                              onClick={() => handleAction('pause', workflow.workflow_id, workflow.name)}
                              disabled={executingWorkflow === workflow.workflow_id}
                              className="p-2 text-yellow-600 hover:bg-yellow-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                              title="Pause Workflow"
                            >
                              <Pause className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => handleAction('stop', workflow.workflow_id, workflow.name)}
                              disabled={executingWorkflow === workflow.workflow_id}
                              className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                              title="Stop Workflow"
                            >
                              <Square className="w-4 h-4" />
                            </button>
                          </>
                        ) : workflow.status === 'paused' ? (
                          <button
                            onClick={() => handleAction('resume', workflow.workflow_id, workflow.name)}
                            disabled={executingWorkflow === workflow.workflow_id}
                            className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Resume Workflow"
                          >
                            <Play className="w-4 h-4" />
                          </button>
                        ) : (
                          <button
                            onClick={() => handleAction('execute', workflow.workflow_id, workflow.name)}
                            disabled={executingWorkflow === workflow.workflow_id}
                            className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Execute Workflow"
                          >
                            {executingWorkflow === workflow.workflow_id ? (
                              <RotateCcw className="w-4 h-4 animate-spin" />
                            ) : (
                              <Play className="w-4 h-4" />
                            )}
                          </button>
                        )}
                        
                        <button
                          onClick={() => handleAction('monitor', workflow.workflow_id, workflow.name)}
                          className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                          title="View Execution History & Monitor"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        
                        {/* Scheduling Controls */}
                        {workflow.is_scheduled ? (
                          <button
                            onClick={() => handleAction('unschedule', workflow.workflow_id, workflow.name)}
                            className="p-2 text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
                            title="Unschedule Workflow"
                          >
                            <Calendar className="w-4 h-4" />
                          </button>
                        ) : (
                          <button
                            onClick={() => handleAction('schedule', workflow.workflow_id, workflow.name)}
                            className="p-2 text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
                            title="Schedule Workflow"
                          >
                            <Calendar className="w-4 h-4" />
                          </button>
                        )}
                        
                        <button
                          onClick={() => handleAction('generateCode', workflow.workflow_id, workflow.name)}
                          className="p-2 text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
                          title="Generate Code"
                        >
                          <Code className="w-4 h-4" />
                        </button>
                        
                        <button
                          onClick={() => console.log('Navigate to edit:', workflow.workflow_id)}
                          className="p-2 text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
                          title="Edit Workflow"
                        >
                          <Edit className="w-4 h-4" />
                        </button>
                        
                        <button
                          onClick={() => handleAction('delete', workflow.workflow_id, workflow.name)}
                          className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          title="Delete Workflow"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    {/* List Row Layout */}
                    <div className="flex-1 grid grid-cols-12 gap-4 items-center">
                      <div className="col-span-3">
                        <h3 className="font-semibold text-gray-900 truncate">{workflow.name}</h3>
                        <p className="text-sm text-gray-600 truncate">{workflow.description}</p>
                      </div>
                      
                      <div className="col-span-2">
                        <div className="flex items-center gap-2">
                          {getStatusBadge(workflow.status, workflow.is_scheduled)}
                          {workflow.status === 'running' && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 border border-blue-200">
                              <span className="w-2 h-2 bg-blue-400 rounded-full mr-1 animate-pulse"></span>
                              Live
                            </span>
                          )}
                        </div>
                      </div>
                      
                      <div className="col-span-1 text-center">
                        <span className="font-medium">{workflow.states_count || 0}</span>
                        <p className="text-xs text-gray-500">states</p>
                      </div>
                      
                      <div className="col-span-1 text-center">
                        <span className="font-medium">
                          {typeof workflow.success_rate === 'number' ? `${workflow.success_rate.toFixed(1)}%` : 'N/A'}
                        </span>
                        <p className="text-xs text-gray-500">success</p>
                      </div>
                      
                      <div className="col-span-1 text-center">
                        <span className="font-medium">{workflow.executions_today || 0}</span>
                        <p className="text-xs text-gray-500">today</p>
                      </div>
                      
                      <div className="col-span-1 text-center">
                        <span className="font-medium">
                          {workflow.avg_duration ? formatDuration(workflow.avg_duration) : 'N/A'}
                        </span>
                        <p className="text-xs text-gray-500">avg</p>
                      </div>
                      
                      <div className="col-span-1 text-center">
                        <span className="text-sm text-gray-600">{workflow.author || 'Unknown'}</span>
                      </div>
                      
                      <div className="col-span-2 flex justify-end space-x-1">
                        {/* Dynamic Action Button based on status */}
                        {workflow.status === 'running' ? (
                          <>
                            <button
                              onClick={() => handleAction('pause', workflow.workflow_id, workflow.name)}
                              disabled={executingWorkflow === workflow.workflow_id}
                              className="p-2 text-yellow-600 hover:bg-yellow-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                              title="Pause"
                            >
                              <Pause className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => handleAction('stop', workflow.workflow_id, workflow.name)}
                              disabled={executingWorkflow === workflow.workflow_id}
                              className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                              title="Stop"
                            >
                              <Square className="w-4 h-4" />
                            </button>
                          </>
                        ) : workflow.status === 'paused' ? (
                          <button
                            onClick={() => handleAction('resume', workflow.workflow_id, workflow.name)}
                            disabled={executingWorkflow === workflow.workflow_id}
                            className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Resume"
                          >
                            <Play className="w-4 h-4" />
                          </button>
                        ) : (
                          <button
                            onClick={() => handleAction('execute', workflow.workflow_id, workflow.name)}
                            disabled={executingWorkflow === workflow.workflow_id}
                            className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Execute"
                          >
                            {executingWorkflow === workflow.workflow_id ? (
                              <RotateCcw className="w-4 h-4 animate-spin" />
                            ) : (
                              <Play className="w-4 h-4" />
                            )}
                          </button>
                        )}
                        
                        <button
                          onClick={() => handleAction('monitor', workflow.workflow_id, workflow.name)}
                          className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                          title="Monitor"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        
                        {/* Scheduling Controls */}
                        {workflow.is_scheduled ? (
                          <button
                            onClick={() => handleAction('unschedule', workflow.workflow_id, workflow.name)}
                            className="p-2 text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
                            title="Unschedule"
                          >
                            <Calendar className="w-4 h-4" />
                          </button>
                        ) : (
                          <button
                            onClick={() => handleAction('schedule', workflow.workflow_id, workflow.name)}
                            className="p-2 text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
                            title="Schedule"
                          >
                            <Calendar className="w-4 h-4" />
                          </button>
                        )}
                        
                        <button
                          onClick={() => handleAction('generateCode', workflow.workflow_id, workflow.name)}
                          className="p-2 text-purple-600 hover:bg-purple-50 rounded-lg transition-colors"
                          title="Code"
                        >
                          <Code className="w-4 h-4" />
                        </button>
                        
                        <button
                          onClick={() => console.log('Navigate to edit:', workflow.workflow_id)}
                          className="p-2 text-gray-600 hover:bg-gray-50 rounded-lg transition-colors"
                          title="Edit"
                        >
                          <Edit className="w-4 h-4" />
                        </button>
                        
                        <button
                          onClick={() => handleAction('delete', workflow.workflow_id, workflow.name)}
                          className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Quick Create Modal */}
      {showCreator && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden">
            <WorkflowYamlCreator
              onSuccess={() => {
                setShowCreator(false)
                refetch()
              }}
              onCancel={() => setShowCreator(false)}
            />
          </div>
        </div>
      )}

      {/* Schedule Modal */}
      <ScheduleModal
        workflow={scheduleModal.workflow}
        isOpen={scheduleModal.isOpen}
        onClose={() => setScheduleModal({ isOpen: false, workflow: null })}
        onSchedule={handleScheduleWorkflow}
      />

      {/* Execution Monitor Modal */}
      {monitoringExecution && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <ExecutionMonitor
            workflowId={monitoringExecution.workflowId}
            executionId={monitoringExecution.executionId}
            onClose={() => setMonitoringExecution(null)}
          />
        </div>
      )}
    </div>
  )
}