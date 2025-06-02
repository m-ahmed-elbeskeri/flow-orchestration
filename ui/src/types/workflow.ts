export interface Workflow {
  id: string
  name: string
  description: string
  version: string
  author: string
  status: 'created' | 'running' | 'completed' | 'failed' | 'paused'
  created_at: string
  updated_at: string
  yaml_content: string
}

export interface WorkflowCreateRequest {
  name: string
  yaml_content: string
  auto_start?: boolean
}

export interface WorkflowCreateResponse {
  workflow_id: string
  name: string
  status: string
  message?: string
}

export interface WorkflowListItem {
  workflow_id: string
  name: string
  description: string
  status: string
  created_at: string
  states_count?: number
}

export interface WorkflowExecution {
  workflow_id: string
  execution_id?: string
  status: 'running' | 'completed' | 'failed' | 'paused'
  started_at: string
  completed_at?: string
  current_state?: string
  error?: string
}