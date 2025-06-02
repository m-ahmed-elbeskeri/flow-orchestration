// src/types/workflow.ts
export interface Workflow {
  id: string
  name: string
  description: string
  version: string
  status: 'draft' | 'active' | 'paused'
  created_at: string
  updated_at: string
}

export interface FlowNode {
  id: string
  type: string
  position: { x: number; y: number }
  data: {
    label: string
    type: string
    config?: Record<string, any>
  }
}

export interface FlowEdge {
  id: string
  source: string
  target: string
  type?: string
  animated?: boolean
}

export interface WorkflowExecution {
  workflow_id: string
  execution_id: string
  status: 'running' | 'completed' | 'failed' | 'paused'
  started_at: string
  completed_at?: string
  current_state?: string
  error?: string
}