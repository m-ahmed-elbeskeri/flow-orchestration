// ui/src/components/WorkflowFlowChart.tsx
import React, { useMemo } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import { CheckCircle, XCircle, Clock, Activity, AlertTriangle } from 'lucide-react'

interface WorkflowFlowChartProps {
  states: ExecutionState[]
  onStateClick: (stateName: string) => void
  execution: any
}

const StateNode: React.FC<{ data: any }> = ({ data }) => {
  const getStatusColor = () => {
    switch (data.status) {
      case 'completed': return 'border-green-500 bg-green-50'
      case 'failed': return 'border-red-500 bg-red-50'
      case 'running': return 'border-blue-500 bg-blue-50'
      case 'pending': return 'border-gray-300 bg-gray-50'
      default: return 'border-gray-300 bg-white'
    }
  }

  const getStatusIcon = () => {
    switch (data.status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'failed': return <XCircle className="w-4 h-4 text-red-500" />
      case 'running': return <Activity className="w-4 h-4 text-blue-500 animate-pulse" />
      case 'retrying': return <AlertTriangle className="w-4 h-4 text-yellow-500" />
      default: return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  return (
    <div 
      className={`px-4 py-3 rounded-lg border-2 cursor-pointer transition-all hover:shadow-md ${getStatusColor()}`}
      onClick={() => data.onClick(data.name)}
    >
      <div className="flex items-center space-x-2">
        {getStatusIcon()}
        <span className="font-medium text-sm">{data.name}</span>
      </div>
      {data.duration && (
        <div className="text-xs text-gray-600 mt-1">
          {Math.round(data.duration)}ms
        </div>
      )}
      {data.attempts > 1 && (
        <div className="text-xs text-orange-600 mt-1">
          Attempt {data.attempts}
        </div>
      )}
    </div>
  )
}

const WorkflowFlowChart: React.FC<WorkflowFlowChartProps> = ({ states, onStateClick, execution }) => {
  const nodeTypes = {
    stateNode: StateNode
  }

  const { nodes, edges } = useMemo(() => {
    if (!states) return { nodes: [], edges: [] }

    const nodes: Node[] = states.map((state, index) => ({
      id: state.name,
      type: 'stateNode',
      position: { x: (index % 4) * 250, y: Math.floor(index / 4) * 100 },
      data: {
        name: state.name,
        status: state.status,
        duration: state.duration,
        attempts: state.attempts,
        onClick: onStateClick
      }
    }))

    const edges: Edge[] = []
    states.forEach(state => {
      state.transitions.forEach(target => {
        edges.push({
          id: `${state.name}-${target}`,
          source: state.name,
          target: target,
          animated: state.status === 'running',
          style: { 
            stroke: state.status === 'failed' ? '#ef4444' : 
                   state.status === 'completed' ? '#10b981' : '#6b7280' 
          }
        })
      })
    })

    return { nodes, edges }
  }, [states, onStateClick])

  return (
    <div style={{ height: '400px', width: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Controls />
        <Background />
      </ReactFlow>
    </div>
  )
}

export default WorkflowFlowChart