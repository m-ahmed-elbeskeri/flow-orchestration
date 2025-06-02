// ui/src/components/OverviewTab.tsx
import React from 'react'
import { Clock, Zap, Target, TrendingUp, AlertTriangle, CheckCircle, XCircle, Activity } from 'lucide-react'
import WorkflowFlowChart from './WorkflowFlowChart'

interface OverviewTabProps {
  execution: any
  metrics: ExecutionMetrics
  states: ExecutionState[]
  alerts: Alert[]
  onStateSelect: (stateName: string) => void
}

const OverviewTab: React.FC<OverviewTabProps> = ({ 
  execution, 
  metrics, 
  states, 
  alerts,
  onStateSelect 
}) => {
  const activeAlerts = alerts?.filter(a => !a.resolved) || []
  const criticalAlerts = activeAlerts.filter(a => a.severity === 'critical')

  return (
    <div className="space-y-8">
      {/* Critical Alerts Banner */}
      {criticalAlerts.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
            <h3 className="text-red-800 font-medium">Critical Issues Detected</h3>
          </div>
          <div className="mt-2 space-y-1">
            {criticalAlerts.slice(0, 3).map(alert => (
              <p key={alert.id} className="text-red-700 text-sm">{alert.message}</p>
            ))}
          </div>
        </div>
      )}

      {/* Key Metrics Grid */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <Clock className="w-8 h-8 text-blue-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Runtime</p>
                <p className="text-2xl font-bold text-gray-900">
                  {Math.round(metrics.totalExecutionTime / 1000)}s
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <Target className="w-8 h-8 text-green-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Success Rate</p>
                <p className="text-2xl font-bold text-gray-900">
                  {Math.round(((metrics.totalStates - metrics.failedStates) / metrics.totalStates) * 100)}%
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <Zap className="w-8 h-8 text-purple-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Throughput</p>
                <p className="text-2xl font-bold text-gray-900">
                  {metrics.throughput.toFixed(1)}/s
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <TrendingUp className="w-8 h-8 text-orange-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Avg State Time</p>
                <p className="text-2xl font-bold text-gray-900">
                  {Math.round(metrics.avgStateTime)}ms
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Workflow Visualization */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Workflow Progress</h3>
        </div>
        <div className="p-6">
          <WorkflowFlowChart 
            states={states} 
            onStateClick={onStateSelect}
            execution={execution}
          />
        </div>
      </div>

      {/* State Summary */}
      {states && (
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">State Summary</h3>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-2" />
                <p className="text-2xl font-bold text-gray-900">{metrics?.completedStates}</p>
                <p className="text-sm text-gray-600">Completed</p>
              </div>
              <div className="text-center">
                <Activity className="w-12 h-12 text-blue-500 mx-auto mb-2" />
                <p className="text-2xl font-bold text-gray-900">{metrics?.activeStates}</p>
                <p className="text-sm text-gray-600">Running</p>
              </div>
              <div className="text-center">
                <XCircle className="w-12 h-12 text-red-500 mx-auto mb-2" />
                <p className="text-2xl font-bold text-gray-900">{metrics?.failedStates}</p>
                <p className="text-sm text-gray-600">Failed</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Resource Utilization */}
      {metrics?.resourceUtilization && (
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Resource Utilization</h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>CPU Usage</span>
                  <span>{Math.round(metrics.resourceUtilization.cpu)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full" 
                    style={{ width: `${metrics.resourceUtilization.cpu}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>Memory Usage</span>
                  <span>{Math.round(metrics.resourceUtilization.memory)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full" 
                    style={{ width: `${metrics.resourceUtilization.memory}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>Network I/O</span>
                  <span>{Math.round(metrics.resourceUtilization.network)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-purple-500 h-2 rounded-full" 
                    style={{ width: `${metrics.resourceUtilization.network}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default OverviewTab