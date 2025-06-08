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
  metadata?: Record<string, any>
}

export interface WorkflowCreateRequest {
  name: string
  description?: string
  yaml_content: string
  auto_start?: boolean
  metadata?: Record<string, any>
}

export interface WorkflowCreateResponse {
  workflow_id: string
  name: string
  description: string
  status: string
  created_at: string
  updated_at: string
  states_count: number
  message?: string
}

export interface WorkflowListItem {
  workflow_id: string
  name: string
  description: string
  status: string
  created_at: string
  updated_at: string
  states_count?: number
  last_execution?: string
}

export interface WorkflowExecution {
  workflow_id: string
  execution_id: string
  status: 'running' | 'completed' | 'failed' | 'paused' | 'cancelled'
  started_at: string
  completed_at?: string
  current_state?: string
  error?: string
  parameters?: Record<string, any>
  metadata?: Record<string, any>
}

export interface ExecutionState {
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'retrying' | 'paused' | 'cancelled'
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

export interface StateMetrics {
  cpuUsage: number
  memoryUsage: number
  networkIO: number
  executionTime: number
  retryCount: number
  resourceUtilization?: {
    cpu: number
    memory: number
    network: number
  }
}

export interface ExecutionMetrics {
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
  timestamp: string
}

export interface LogEntry {
  id: string
  timestamp: string
  level: 'debug' | 'info' | 'warning' | 'error'
  message: string
  state?: string
  data?: any
  source?: string
}

export interface ExecutionEvent {
  id: string
  timestamp: string
  type: 'state_started' | 'state_completed' | 'state_failed' | 'workflow_paused' | 
        'workflow_resumed' | 'workflow_cancelled' | 'alert' | 'resource_warning' |
        'dependency_satisfied' | 'dependency_blocked'
  state?: string
  message: string
  level: 'info' | 'warning' | 'error' | 'success'
  metadata?: Record<string, any>
}

export interface Alert {
  id: string
  type: 'warning' | 'error' | 'info' | 'critical'
  message: string
  timestamp: string
  state?: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  rule_name?: string
  resolved?: boolean
  resolved_at?: string
}

export interface ValidationResult {
  is_valid: boolean
  errors: ValidationError[]
  warnings: ValidationWarning[]
  info: Record<string, any>
}

export interface ValidationError {
  message: string
  path: string
  code?: string
  line?: number
  column?: number
}

export interface ValidationWarning {
  message: string
  path: string
  code?: string
  line?: number
  column?: number
}

export interface CodeGenerationResult {
  success: boolean
  files: Record<string, string>
  zip_content?: string
  message?: string
  errors?: string[]
}

export interface WorkflowTemplate {
  id: string
  name: string
  description: string
  category: string
  yaml_content: string
  variables: TemplateVariable[]
  metadata?: Record<string, any>
}

export interface TemplateVariable {
  name: string
  type: 'string' | 'number' | 'boolean' | 'array' | 'object'
  description: string
  default?: any
  required?: boolean
  options?: any[]
}

export interface NodeType {
  id: string
  name: string
  category: 'builtin' | 'integration' | 'custom'
  description: string
  inputs: NodeInput[]
  outputs: NodeOutput[]
  icon?: string
  color?: string
  examples?: NodeExample[]
}

export interface NodeInput {
  name: string
  type: string
  required: boolean
  description: string
  default?: any
  options?: any[]
}

export interface NodeOutput {
  name: string
  type: string
  description: string
}

export interface NodeExample {
  name: string
  description: string
  config: Record<string, any>
}

export interface Plugin {
  name: string
  version: string
  description: string
  author: string
  status: 'loaded' | 'failed' | 'disabled'
  states: string[]
  dependencies: string[]
  manifest?: PluginManifest
}

export interface PluginManifest {
  name: string
  version: string
  description: string
  author: string
  states: Record<string, StateDefinition>
  resources?: ResourceRequirements
  dependencies?: string[]
}

export interface StateDefinition {
  description: string
  inputs: Record<string, InputDefinition>
  outputs: Record<string, OutputDefinition>
  examples?: Record<string, any>[]
}

export interface InputDefinition {
  type: string
  required: boolean
  description: string
  default?: any
  secret?: boolean
  options?: any[]
}

export interface OutputDefinition {
  type: string
  description: string
}

export interface ResourceRequirements {
  cpu_units?: number
  memory_mb?: number
  io_weight?: number
  network_weight?: number
  gpu_units?: number
  priority?: number
  timeout?: number
}

export interface Schedule {
  id: string
  cron?: string
  timezone: string
  enabled: boolean
  max_instances: number
  last_run?: string
  next_run?: string
}

export interface SystemInfo {
  version: string
  environment: string
  features: string[]
  uptime: number
  stats: {
    total_workflows: number
    active_workflows: number
    total_executions: number
    successful_executions: number
    failed_executions: number
  }
}

// API Response types
export interface ApiResponse<T = any> {
  data: T
  message?: string
  success: boolean
}

export interface PaginatedResponse<T = any> {
  items: T[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

export interface ApiError {
  message: string
  code?: string
  details?: any
  status?: number
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'execution_update' | 'state_update' | 'metrics_update' | 'event' | 'alert' | 'log'
  data: any
  timestamp: string
}

// Filter and search types
export interface ExecutionFilter {
  status?: string[]
  date_range?: {
    start: string
    end: string
  }
  workflow_id?: string
  search?: string
}

export interface LogFilter {
  level?: 'debug' | 'info' | 'warning' | 'error'
  state?: string
  search?: string
  limit?: number
  since?: string
}

export interface EventFilter {
  type?: string[]
  level?: string[]
  state?: string
  search?: string
  limit?: number
  since?: string
}

// Dashboard types
export interface DashboardMetrics {
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
  trends: {
    executions: TrendData[]
    success_rate: TrendData[]
    avg_duration: TrendData[]
  }
}

export interface TrendData {
  date: string
  value: number
}

export interface WorkflowStats {
  id: string
  name: string
  totalExecutions: number
  successRate: number
  avgDuration: number
  lastExecuted: string
  status: 'active' | 'inactive' | 'error'
}

// Environment and configuration types
export interface EnvironmentVariable {
  key: string
  value: string
  description?: string
  required?: boolean
}

export interface Secret {
  key: string
  description?: string
  created_at: string
  last_used?: string
}

export interface FileUpload {
  file_id: string
  filename: string
  size: number
  content_type: string
  uploaded_at: string
  workflow_id?: string
}

// Export types
export type ExecutionStatus = 'running' | 'completed' | 'failed' | 'paused' | 'cancelled'
export type StateStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'retrying' | 'paused' | 'cancelled'
export type LogLevel = 'debug' | 'info' | 'warning' | 'error'
export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical'
export type AlertType = 'warning' | 'error' | 'info' | 'critical'