import React, { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart3, TrendingUp, TrendingDown, Activity, Clock, CheckCircle,
  XCircle, AlertTriangle, Zap, Target, Users, Calendar, Filter,
  Download, RefreshCw, Eye, ArrowUp, ArrowDown, Minus
} from 'lucide-react'
import { workflowApi } from '../api/client'

interface DashboardMetrics {
  totalWorkflows: number
  activeWorkflows: number
  totalExecutions: number
  successfulExecutions: number
  failedExecutions: number
  avgExecutionTime: number
  systemHealth: number
  resourceUtilization: {
    cpu: number
    memory: number
    network: number
  }
}

interface WorkflowStats {
  id: string
  name: string
  totalExecutions: number
  successRate: number
  avgDuration: number
  lastExecuted: string
  status: 'active' | 'inactive' | 'error'
}

interface ExecutionTrend {
  date: string
  successful: number
  failed: number
  total: number
}

interface SystemAlert {
  id: string
  type: 'warning' | 'error' | 'info'
  message: string
  timestamp: string
  source: string
}

export default function Dashboard() {
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h')
  const [selectedMetric, setSelectedMetric] = useState<'executions' | 'performance' | 'resources'>('executions')

  // Fetch dashboard data
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['dashboard-metrics', timeRange],
    queryFn: () => workflowApi.getSystemMetrics({ 
      start_date: getStartDate(timeRange),
      end_date: new Date().toISOString(),
    }),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: workflows } = useQuery({
    queryKey: ['workflows-stats'],
    queryFn: () => workflowApi.listWorkflows(),
    refetchInterval: 60000, // Refresh every minute
  })

  const { data: executionTrends } = useQuery({
    queryKey: ['execution-trends', timeRange],
    queryFn: () => workflowApi.getSystemMetrics({
      start_date: getStartDate(timeRange),
      end_date: new Date().toISOString(),
      metrics: ['execution_trends']
    }),
    refetchInterval: 60000,
  })

  function getStartDate(range: string): string {
    const now = new Date()
    switch (range) {
      case '1h':
        return new Date(now.getTime() - 60 * 60 * 1000).toISOString()
      case '24h':
        return new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString()
      case '7d':
        return new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString()
      case '30d':
        return new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString()
      default:
        return new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString()
    }
  }

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`
    } else {
      return `${seconds}s`
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M'
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K'
    }
    return num.toString()
  }

  // Mock data for demonstration (replace with real API data)
  const mockMetrics: DashboardMetrics = {
    totalWorkflows: 42,
    activeWorkflows: 38,
    totalExecutions: 15420,
    successfulExecutions: 14281,
    failedExecutions: 1139,
    avgExecutionTime: 145000, // ms
    systemHealth: 94.2,
    resourceUtilization: {
      cpu: 34.5,
      memory: 67.2,
      network: 12.8
    }
  }

  const mockWorkflowStats: WorkflowStats[] = [
    {
      id: '1',
      name: 'Email Notification Pipeline',
      totalExecutions: 2840,
      successRate: 98.5,
      avgDuration: 12000,
      lastExecuted: '2025-01-20T10:30:00Z',
      status: 'active'
    },
    {
      id: '2', 
      name: 'Data Processing Workflow',
      totalExecutions: 1205,
      successRate: 94.2,
      avgDuration: 340000,
      lastExecuted: '2025-01-20T09:15:00Z',
      status: 'active'
    },
    {
      id: '3',
      name: 'API Health Monitor',
      totalExecutions: 8640,
      successRate: 99.8,
      avgDuration: 3000,
      lastExecuted: '2025-01-20T10:32:00Z',
      status: 'active'
    },
    {
      id: '4',
      name: 'File Backup Process',
      totalExecutions: 720,
      successRate: 89.4,
      avgDuration: 180000,
      lastExecuted: '2025-01-19T23:00:00Z',
      status: 'error'
    }
  ]

  const mockExecutionTrends: ExecutionTrend[] = Array.from({ length: 24 }, (_, i) => ({
    date: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
    successful: Math.floor(Math.random() * 100) + 50,
    failed: Math.floor(Math.random() * 20) + 2,
    total: 0
  })).map(trend => ({
    ...trend,
    total: trend.successful + trend.failed
  }))

  const mockAlerts: SystemAlert[] = [
    {
      id: '1',
      type: 'warning',
      message: 'High memory usage detected on worker node 2',
      timestamp: '2025-01-20T10:25:00Z',
      source: 'system'
    },
    {
      id: '2',
      type: 'error', 
      message: 'File Backup Process failed 3 times in the last hour',
      timestamp: '2025-01-20T10:15:00Z',
      source: 'workflow'
    },
    {
      id: '3',
      type: 'info',
      message: 'New plugin "PostgreSQL Connector" installed successfully',
      timestamp: '2025-01-20T09:45:00Z',
      source: 'system'
    }
  ]

  const successRate = ((mockMetrics.successfulExecutions / mockMetrics.totalExecutions) * 100)
  const errorRate = ((mockMetrics.failedExecutions / mockMetrics.totalExecutions) * 100)

  if (metricsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                <BarChart3 className="w-8 h-8 mr-3 text-blue-600" />
                Workflow Dashboard
              </h1>
              <p className="text-gray-600 mt-1">Monitor your workflow ecosystem performance and insights</p>
            </div>
            
            <div className="flex items-center space-x-3">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value as any)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="1h">Last Hour</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              
              <button className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg flex items-center">
                <Download className="w-4 h-4 mr-2" />
                Export
              </button>
              
              <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center">
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Activity className="w-8 h-8 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Workflows</p>
                <p className="text-2xl font-bold text-gray-900">{formatNumber(mockMetrics.totalWorkflows)}</p>
                <p className="text-sm text-green-600 flex items-center mt-1">
                  <ArrowUp className="w-3 h-3 mr-1" />
                  +3 this week
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Zap className="w-8 h-8 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Success Rate</p>
                <p className="text-2xl font-bold text-gray-900">{successRate.toFixed(1)}%</p>
                <p className="text-sm text-green-600 flex items-center mt-1">
                  <ArrowUp className="w-3 h-3 mr-1" />
                  +0.3% from yesterday
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Clock className="w-8 h-8 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Avg Execution Time</p>
                <p className="text-2xl font-bold text-gray-900">{formatDuration(mockMetrics.avgExecutionTime)}</p>
                <p className="text-sm text-red-600 flex items-center mt-1">
                  <ArrowUp className="w-3 h-3 mr-1" />
                  +12s from yesterday
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Target className="w-8 h-8 text-emerald-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">System Health</p>
                <p className="text-2xl font-bold text-gray-900">{mockMetrics.systemHealth.toFixed(1)}%</p>
                <p className="text-sm text-gray-600 flex items-center mt-1">
                  <Minus className="w-3 h-3 mr-1" />
                  Stable
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Charts and Analytics */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Execution Trends Chart */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border p-6">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-semibold text-gray-900">Execution Trends</h3>
              <div className="flex space-x-2">
                {['executions', 'performance', 'resources'].map((metric) => (
                  <button
                    key={metric}
                    onClick={() => setSelectedMetric(metric as any)}
                    className={`px-3 py-1 text-sm rounded-lg transition-colors ${
                      selectedMetric === metric
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    {metric.charAt(0).toUpperCase() + metric.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Simple bar chart representation */}
            <div className="space-y-4">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Executions over time ({timeRange})</span>
                <div className="flex space-x-4">
                  <span className="flex items-center">
                    <div className="w-3 h-3 bg-green-500 rounded mr-2"></div>
                    Successful
                  </span>
                  <span className="flex items-center">
                    <div className="w-3 h-3 bg-red-500 rounded mr-2"></div>
                    Failed
                  </span>
                </div>
              </div>
              
              <div className="grid grid-cols-12 gap-1 h-40">
                {mockExecutionTrends.slice(-12).map((trend, index) => {
                  const maxValue = Math.max(...mockExecutionTrends.map(t => t.total))
                  const successHeight = (trend.successful / maxValue) * 100
                  const failedHeight = (trend.failed / maxValue) * 100
                  
                  return (
                    <div key={index} className="flex flex-col justify-end h-full">
                      <div className="flex flex-col justify-end h-full bg-gray-100 rounded">
                        <div 
                          className="bg-red-500 rounded-t"
                          style={{ height: `${failedHeight}%` }}
                          title={`Failed: ${trend.failed}`}
                        ></div>
                        <div 
                          className="bg-green-500"
                          style={{ height: `${successHeight}%` }}
                          title={`Successful: ${trend.successful}`}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500 mt-1 text-center">
                        {new Date(trend.date).getHours().toString().padStart(2, '0')}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* System Health */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">System Health</h3>
            
            <div className="space-y-6">
              {/* Overall Health Score */}
              <div className="text-center">
                <div className="relative inline-flex items-center justify-center w-24 h-24">
                  <svg className="transform -rotate-90 w-24 h-24">
                    <circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="transparent"
                      className="text-gray-200"
                    />
                    <circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="transparent"
                      strokeDasharray={`${2 * Math.PI * 40}`}
                      strokeDashoffset={`${2 * Math.PI * 40 * (1 - mockMetrics.systemHealth / 100)}`}
                      className="text-green-500"
                    />
                  </svg>
                  <span className="absolute text-xl font-bold text-gray-900">
                    {mockMetrics.systemHealth.toFixed(0)}%
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-2">Overall Health</p>
              </div>

              {/* Resource Utilization */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">CPU Usage</span>
                    <span className="font-medium">{mockMetrics.resourceUtilization.cpu.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${mockMetrics.resourceUtilization.cpu}%` }}
                    ></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Memory Usage</span>
                    <span className="font-medium">{mockMetrics.resourceUtilization.memory.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full"
                      style={{ width: `${mockMetrics.resourceUtilization.memory}%` }}
                    ></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Network I/O</span>
                    <span className="font-medium">{mockMetrics.resourceUtilization.network.toFixed(1)} MB/s</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full"
                      style={{ width: `${Math.min(mockMetrics.resourceUtilization.network * 8, 100)}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Most Used Workflows */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">Most Used Workflows</h3>
            <div className="space-y-4">
              {mockWorkflowStats.map((workflow, index) => (
                <div key={workflow.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className="flex-shrink-0">
                      <span className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-600 text-sm font-medium">
                        {index + 1}
                      </span>
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900">{workflow.name}</h4>
                      <div className="flex items-center space-x-4 text-sm text-gray-600">
                        <span>{formatNumber(workflow.totalExecutions)} executions</span>
                        <span>•</span>
                        <span>{workflow.successRate.toFixed(1)}% success</span>
                        <span>•</span>
                        <span>{formatDuration(workflow.avgDuration)} avg</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`w-2 h-2 rounded-full ${
                      workflow.status === 'active' ? 'bg-green-500' :
                      workflow.status === 'error' ? 'bg-red-500' : 'bg-gray-400'
                    }`}></span>
                    <button className="text-gray-400 hover:text-gray-600">
                      <Eye className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Alerts */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">System Alerts</h3>
            <div className="space-y-4">
              {mockAlerts.map((alert) => (
                <div key={alert.id} className="flex items-start space-x-3 p-4 bg-gray-50 rounded-lg">
                  <div className="flex-shrink-0 mt-0.5">
                    {alert.type === 'error' ? (
                      <XCircle className="w-5 h-5 text-red-500" />
                    ) : alert.type === 'warning' ? (
                      <AlertTriangle className="w-5 h-5 text-yellow-500" />
                    ) : (
                      <CheckCircle className="w-5 h-5 text-blue-500" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900">{alert.message}</p>
                    <div className="flex items-center justify-between mt-1">
                      <p className="text-xs text-gray-500">{alert.source}</p>
                      <time className="text-xs text-gray-500">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </time>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-6">Quick Actions</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <button className="flex flex-col items-center p-4 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors">
              <Activity className="w-8 h-8 text-blue-600 mb-2" />
              <span className="text-sm font-medium text-blue-900">View All Workflows</span>
            </button>
            
            <button className="flex flex-col items-center p-4 bg-green-50 rounded-lg hover:bg-green-100 transition-colors">
              <TrendingUp className="w-8 h-8 text-green-600 mb-2" />
              <span className="text-sm font-medium text-green-900">Performance Report</span>
            </button>
            
            <button className="flex flex-col items-center p-4 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors">
              <Users className="w-8 h-8 text-purple-600 mb-2" />
              <span className="text-sm font-medium text-purple-900">User Management</span>
            </button>
            
            <button className="flex flex-col items-center p-4 bg-yellow-50 rounded-lg hover:bg-yellow-100 transition-colors">
              <Calendar className="w-8 h-8 text-yellow-600 mb-2" />
              <span className="text-sm font-medium text-yellow-900">Schedule Workflow</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}   