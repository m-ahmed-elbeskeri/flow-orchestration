// ui/src/api/client.ts
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios'
import type {
  Workflow,
  WorkflowExecution,
  WorkflowCreateRequest,
  WorkflowCreateResponse,
  WorkflowListItem,
  ExecutionState,
  ExecutionMetrics,
  ExecutionEvent,
  Alert,
  ValidationResult,
  CodeGenerationResult,
  WorkflowTemplate,
  NodeType
} from '../types/workflow'

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const API_TIMEOUT = 30000 // 30 seconds

interface ApiError {
  message: string
  code?: string
  details?: any
}

interface PaginationParams {
  limit?: number
  offset?: number
}

interface FilterParams {
  status?: string
  author?: string
  created_after?: string
  created_before?: string
  search?: string
}

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token')
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }

        // Add request timestamp for debugging
        config.metadata = { startTime: new Date() }
        
        console.log(`🚀 ${config.method?.toUpperCase()} ${config.url}`, {
          params: config.params,
          data: config.data
        })
        
        return config
      },
      (error) => {
        console.error('❌ Request error:', error)
        return Promise.reject(error)
      }
    )

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        const duration = new Date().getTime() - response.config.metadata?.startTime?.getTime()
        console.log(`✅ ${response.config.method?.toUpperCase()} ${response.config.url} (${duration}ms)`, {
          status: response.status,
          data: response.data
        })
        return response
      },
      (error: AxiosError<ApiError>) => {
        const duration = error.config?.metadata ? 
          new Date().getTime() - error.config.metadata.startTime?.getTime() : 0
        
        console.error(`❌ ${error.config?.method?.toUpperCase()} ${error.config?.url} (${duration}ms)`, {
          status: error.response?.status,
          message: error.response?.data?.message || error.message,
          details: error.response?.data?.details
        })

        // Handle specific error cases
        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token')
          window.location.href = '/login'
        }

        // Transform error for consistent handling
        const apiError: ApiError = {
          message: error.response?.data?.message || error.message || 'An unexpected error occurred',
          code: error.response?.data?.code || error.code,
          details: error.response?.data?.details
        }

        return Promise.reject(apiError)
      }
    )
  }

  // Health and System
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await this.client.get('/api/v1/health')
    return response.data
  }

  async getSystemInfo(): Promise<any> {
    const response = await this.client.get('/api/v1/system/info')
    return response.data
  }

  // Authentication
  async login(credentials: { username: string; password: string }): Promise<{ token: string; user: any }> {
    const response = await this.client.post('/api/v1/auth/login', credentials)
    if (response.data.token) {
      localStorage.setItem('auth_token', response.data.token)
    }
    return response.data
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/api/v1/auth/logout')
    } finally {
      localStorage.removeItem('auth_token')
    }
  }

  async refreshToken(): Promise<{ token: string }> {
    const response = await this.client.post('/api/v1/auth/refresh')
    if (response.data.token) {
      localStorage.setItem('auth_token', response.data.token)
    }
    return response.data
  }

  // Workflows
  async listWorkflows(params: PaginationParams & FilterParams = {}): Promise<{
    workflows: WorkflowListItem[]
    total: number
    limit: number
    offset: number
  }> {
    const response = await this.client.get('/api/v1/workflows', { params })
    return response.data
  }

  async getWorkflow(workflowId: string): Promise<Workflow> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}`)
    return response.data
  }

  async createWorkflow(workflow: WorkflowCreateRequest): Promise<WorkflowCreateResponse> {
    const response = await this.client.post('/api/v1/workflows', workflow)
    return response.data
  }

  async createWorkflowFromYaml(data: {
    name: string
    yaml_content: string
    auto_start?: boolean
  }): Promise<WorkflowCreateResponse> {
    const response = await this.client.post('/api/v1/workflows/from-yaml', data)
    return response.data
  }

  async updateWorkflow(workflowId: string, updates: Partial<Workflow>): Promise<Workflow> {
    const response = await this.client.put(`/api/v1/workflows/${workflowId}`, updates)
    return response.data
  }

  async deleteWorkflow(workflowId: string): Promise<void> {
    await this.client.delete(`/api/v1/workflows/${workflowId}`)
  }

  async duplicateWorkflow(workflowId: string, name?: string): Promise<WorkflowCreateResponse> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/duplicate`, { name })
    return response.data
  }

  // Workflow Validation
  async validateWorkflow(yamlContent: string): Promise<ValidationResult> {
    const response = await this.client.post('/api/v1/workflows/validate-yaml', {
      yaml: yamlContent
    })
    return response.data
  }

  async validateWorkflowById(workflowId: string): Promise<ValidationResult> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/validate`)
    return response.data
  }

  // Code Generation
  async generateCodeFromWorkflow(workflowId: string): Promise<CodeGenerationResult> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/generate-code`)
    return response.data
  }

  async generateCodeFromYaml(yamlContent: string): Promise<CodeGenerationResult> {
    const response = await this.client.post('/api/v1/workflows/generate-code-from-yaml', {
      yaml_content: yamlContent
    })
    return response.data
  }

  // Workflow Execution
  async executeWorkflow(workflowId: string, params?: any): Promise<{
    execution_id: string
    status: string
    started_at: string
  }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/execute`, {
      parameters: params
    })
    return response.data
  }

  async getWorkflowExecutions(workflowId: string, params: PaginationParams = {}): Promise<{
    executions: WorkflowExecution[]
    total: number
  }> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions`, { params })
    return response.data
  }

  // Execution Monitoring
  async getExecution(workflowId: string, executionId: string): Promise<WorkflowExecution> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}`)
    return response.data
  }

  async getExecutionStates(workflowId: string, executionId: string): Promise<ExecutionState[]> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}/states`)
    return response.data
  }

  async getExecutionMetrics(workflowId: string, executionId: string): Promise<ExecutionMetrics> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}/metrics`)
    return response.data
  }

  async getExecutionEvents(
    workflowId: string, 
    executionId: string, 
    params: {
      limit?: number
      level?: string
      event_type?: string
      since?: string
    } = {}
  ): Promise<ExecutionEvent[]> {
    const response = await this.client.get(
      `/api/v1/workflows/${workflowId}/executions/${executionId}/events`,
      { params }
    )
    return response.data
  }

  async getExecutionAlerts(workflowId: string, executionId: string): Promise<Alert[]> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}/alerts`)
    return response.data
  }

  async getExecutionLogs(
    workflowId: string, 
    executionId: string,
    params: {
      limit?: number
      level?: 'debug' | 'info' | 'warning' | 'error'
      state?: string
      since?: string
    } = {}
  ): Promise<any[]> {
    const response = await this.client.get(
      `/api/v1/workflows/${workflowId}/executions/${executionId}/logs`,
      { params }
    )
    return response.data
  }

  // Execution Control
  async pauseExecution(workflowId: string, executionId: string): Promise<{
    status: string
    checkpoint?: any
  }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/pause`)
    return response.data
  }

  async resumeExecution(workflowId: string, executionId: string): Promise<{
    status: string
  }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/resume`)
    return response.data
  }

  async cancelExecution(workflowId: string, executionId: string): Promise<{
    status: string
  }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/cancel`)
    return response.data
  }

  async retryExecution(workflowId: string, executionId: string): Promise<{
    new_execution_id: string
    status: string
  }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/retry`)
    return response.data
  }

  async retryState(workflowId: string, executionId: string, stateName: string): Promise<{
    status: string
    state: string
  }> {
    const response = await this.client.post(
      `/api/v1/workflows/${workflowId}/executions/${executionId}/states/${stateName}/retry`
    )
    return response.data
  }

  async skipState(workflowId: string, executionId: string, stateName: string): Promise<{
    status: string
    state: string
  }> {
    const response = await this.client.post(
      `/api/v1/workflows/${workflowId}/executions/${executionId}/states/${stateName}/skip`
    )
    return response.data
  }

  // Templates
  async listTemplates(): Promise<WorkflowTemplate[]> {
    const response = await this.client.get('/api/v1/workflows/templates')
    return response.data
  }

  async getTemplate(templateId: string): Promise<WorkflowTemplate> {
    const response = await this.client.get(`/api/v1/workflows/templates/${templateId}`)
    return response.data
  }

  async createWorkflowFromTemplate(templateId: string, variables: Record<string, any>): Promise<WorkflowCreateResponse> {
    const response = await this.client.post('/api/v1/workflows/from-template', {
      template: templateId,
      variables
    })
    return response.data
  }

  // Examples and Samples
  async getWorkflowExamples(): Promise<any[]> {
    const response = await this.client.get('/api/v1/workflows/examples')
    return response.data
  }

  async importYamlWorkflow(yamlContent: string): Promise<{
    workflow: any
    validation: ValidationResult
  }> {
    const response = await this.client.post('/api/v1/workflows/import-yaml', {
      yaml: yamlContent
    })
    return response.data
  }

  // Node Types and Plugins
  async getAvailableNodes(): Promise<NodeType[]> {
    const response = await this.client.get('/api/v1/workflows/nodes')
    return response.data
  }

  async getNodeDocumentation(nodeType: string): Promise<any> {
    const response = await this.client.get(`/api/v1/workflows/nodes/${nodeType}/docs`)
    return response.data
  }

  async listPlugins(): Promise<any[]> {
    const response = await this.client.get('/api/v1/plugins')
    return response.data
  }

  async getPluginInfo(pluginName: string): Promise<any> {
    const response = await this.client.get(`/api/v1/plugins/${pluginName}`)
    return response.data
  }

  async installPlugin(pluginName: string): Promise<{ status: string; message: string }> {
    const response = await this.client.post(`/api/v1/plugins/${pluginName}/install`)
    return response.data
  }

  async uninstallPlugin(pluginName: string): Promise<{ status: string; message: string }> {
    const response = await this.client.post(`/api/v1/plugins/${pluginName}/uninstall`)
    return response.data
  }

  // Scheduling
  async scheduleWorkflow(workflowId: string, schedule: {
    cron?: string
    timezone?: string
    enabled?: boolean
    max_instances?: number
  }): Promise<{ schedule_id: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/schedule`, schedule)
    return response.data
  }

  async getWorkflowSchedule(workflowId: string): Promise<any> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/schedule`)
    return response.data
  }

  async updateWorkflowSchedule(workflowId: string, schedule: any): Promise<any> {
    const response = await this.client.put(`/api/v1/workflows/${workflowId}/schedule`, schedule)
    return response.data
  }

  async deleteWorkflowSchedule(workflowId: string): Promise<void> {
    await this.client.delete(`/api/v1/workflows/${workflowId}/schedule`)
  }

  // Analytics and Reports
  async getWorkflowAnalytics(workflowId: string, params: {
    start_date?: string
    end_date?: string
    granularity?: 'hour' | 'day' | 'week' | 'month'
  } = {}): Promise<any> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/analytics`, { params })
    return response.data
  }

  async getSystemMetrics(params: {
    start_date?: string
    end_date?: string
    metrics?: string[]
  } = {}): Promise<any> {
    const response = await this.client.get('/api/v1/system/metrics', { params })
    return response.data
  }

  async exportWorkflowData(workflowId: string, format: 'json' | 'csv' | 'xlsx' = 'json'): Promise<Blob> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/export`, {
      params: { format },
      responseType: 'blob'
    })
    return response.data
  }

  async exportExecutionData(workflowId: string, executionId: string, format: 'json' | 'csv' = 'json'): Promise<Blob> {
    const response = await this.client.get(
      `/api/v1/workflows/${workflowId}/executions/${executionId}/export`,
      {
        params: { format },
        responseType: 'blob'
      }
    )
    return response.data
  }

  // Environment and Configuration
  async getEnvironmentVariables(workflowId: string): Promise<Record<string, string>> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/environment`)
    return response.data
  }

  async updateEnvironmentVariables(workflowId: string, variables: Record<string, string>): Promise<void> {
    await this.client.put(`/api/v1/workflows/${workflowId}/environment`, { variables })
  }

  async getSecrets(workflowId: string): Promise<string[]> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/secrets`)
    return response.data
  }

  async setSecret(workflowId: string, key: string, value: string): Promise<void> {
    await this.client.post(`/api/v1/workflows/${workflowId}/secrets`, { key, value })
  }

  async deleteSecret(workflowId: string, key: string): Promise<void> {
    await this.client.delete(`/api/v1/workflows/${workflowId}/secrets/${key}`)
  }

  // File Operations
  async uploadFile(file: File, workflowId?: string): Promise<{
    file_id: string
    filename: string
    size: number
    content_type: string
  }> {
    const formData = new FormData()
    formData.append('file', file)
    if (workflowId) {
      formData.append('workflow_id', workflowId)
    }

    const response = await this.client.post('/api/v1/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  async downloadFile(fileId: string): Promise<Blob> {
    const response = await this.client.get(`/api/v1/files/${fileId}/download`, {
      responseType: 'blob'
    })
    return response.data
  }

  async deleteFile(fileId: string): Promise<void> {
    await this.client.delete(`/api/v1/files/${fileId}`)
  }

  // WebSocket connection helper
  createWebSocket(path: string, protocols?: string[]): WebSocket {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsHost = API_BASE_URL.replace(/^https?:\/\//, '')
    const token = localStorage.getItem('auth_token')
    
    const url = new URL(`${wsProtocol}//${wsHost}${path}`)
    if (token) {
      url.searchParams.set('token', token)
    }
    
    return new WebSocket(url.toString(), protocols)
  }

  // Utility methods
  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck()
      return true
    } catch {
      return false
    }
  }

  setAuthToken(token: string): void {
    localStorage.setItem('auth_token', token)
  }

  clearAuthToken(): void {
    localStorage.removeItem('auth_token')
  }

  getAuthToken(): string | null {
    return localStorage.getItem('auth_token')
  }

  isAuthenticated(): boolean {
    return !!this.getAuthToken()
  }
}

// Create and export singleton instance
const apiClient = new ApiClient()

// Export both the class and instance for flexibility
export { ApiClient }
export default apiClient

// Convenience export matching the pattern used in components
export const workflowApi = {
  // Health
  healthCheck: () => apiClient.healthCheck(),
  testConnection: () => apiClient.testConnection(),
  
  // Auth
  login: (credentials: { username: string; password: string }) => apiClient.login(credentials),
  logout: () => apiClient.logout(),
  
  // Workflows
  listWorkflows: (params?: PaginationParams & FilterParams) => apiClient.listWorkflows(params),
  getWorkflow: (id: string) => apiClient.getWorkflow(id),
  createWorkflow: (workflow: WorkflowCreateRequest) => apiClient.createWorkflow(workflow),
  createWorkflowFromYaml: (data: { name: string; yaml_content: string; auto_start?: boolean }) => 
    apiClient.createWorkflowFromYaml(data),
  updateWorkflow: (id: string, updates: Partial<Workflow>) => apiClient.updateWorkflow(id, updates),
  deleteWorkflow: (id: string) => apiClient.deleteWorkflow(id),
  duplicateWorkflow: (id: string, name?: string) => apiClient.duplicateWorkflow(id, name),
  
  // Validation & Code Generation
  validateWorkflow: (yaml: string) => apiClient.validateWorkflow(yaml),
  generateCodeFromWorkflow: (id: string) => apiClient.generateCodeFromWorkflow(id),
  generateCodeFromYaml: (yaml: string) => apiClient.generateCodeFromYaml(yaml),
  
  // Execution
  executeWorkflow: (id: string, params?: any) => apiClient.executeWorkflow(id, params),
  getWorkflowExecutions: (id: string, params?: PaginationParams) => apiClient.getWorkflowExecutions(id, params),
  
  // Execution Monitoring
  getExecution: (workflowId: string, executionId: string) => apiClient.getExecution(workflowId, executionId),
  getExecutionStates: (workflowId: string, executionId: string) => apiClient.getExecutionStates(workflowId, executionId),
  getExecutionMetrics: (workflowId: string, executionId: string) => apiClient.getExecutionMetrics(workflowId, executionId),
  getExecutionEvents: (workflowId: string, executionId: string, params?: any) => 
    apiClient.getExecutionEvents(workflowId, executionId, params),
  getExecutionAlerts: (workflowId: string, executionId: string) => apiClient.getExecutionAlerts(workflowId, executionId),
  getExecutionLogs: (workflowId: string, executionId: string, params?: any) => 
    apiClient.getExecutionLogs(workflowId, executionId, params),
  
  // Execution Control
  pauseExecution: (workflowId: string, executionId: string) => apiClient.pauseExecution(workflowId, executionId),
  resumeExecution: (workflowId: string, executionId: string) => apiClient.resumeExecution(workflowId, executionId),
  cancelExecution: (workflowId: string, executionId: string) => apiClient.cancelExecution(workflowId, executionId),
  retryExecution: (workflowId: string, executionId: string) => apiClient.retryExecution(workflowId, executionId),
  retryState: (workflowId: string, executionId: string, stateName: string) => 
    apiClient.retryState(workflowId, executionId, stateName),
  skipState: (workflowId: string, executionId: string, stateName: string) => 
    apiClient.skipState(workflowId, executionId, stateName),
  
  // Templates & Examples
  listTemplates: () => apiClient.listTemplates(),
  getTemplate: (id: string) => apiClient.getTemplate(id),
  createWorkflowFromTemplate: (templateId: string, variables: Record<string, any>) => 
    apiClient.createWorkflowFromTemplate(templateId, variables),
  getWorkflowExamples: () => apiClient.getWorkflowExamples(),
  
  // Plugins & Nodes
  listPlugins: () => apiClient.listPlugins(),
  getPluginInfo: (name: string) => apiClient.getPluginInfo(name),
  getAvailableNodes: () => apiClient.getAvailableNodes(),
  
  // Scheduling
  scheduleWorkflow: (id: string, schedule: any) => apiClient.scheduleWorkflow(id, schedule),
  getWorkflowSchedule: (id: string) => apiClient.getWorkflowSchedule(id),
  updateWorkflowSchedule: (id: string, schedule: any) => apiClient.updateWorkflowSchedule(id, schedule),
  deleteWorkflowSchedule: (id: string) => apiClient.deleteWorkflowSchedule(id),
  
  // Analytics
  getWorkflowAnalytics: (id: string, params?: any) => apiClient.getWorkflowAnalytics(id, params),
  getSystemMetrics: (params?: any) => apiClient.getSystemMetrics(params),
  
  // Export
  exportWorkflowData: (id: string, format?: 'json' | 'csv' | 'xlsx') => apiClient.exportWorkflowData(id, format),
  exportExecutionData: (workflowId: string, executionId: string, format?: 'json' | 'csv') => 
    apiClient.exportExecutionData(workflowId, executionId, format),
  
  // Files
  uploadFile: (file: File, workflowId?: string) => apiClient.uploadFile(file, workflowId),
  downloadFile: (fileId: string) => apiClient.downloadFile(fileId),
  deleteFile: (fileId: string) => apiClient.deleteFile(fileId),
  
  // WebSocket
  createWebSocket: (path: string, protocols?: string[]) => apiClient.createWebSocket(path, protocols),
  
  // Environment
  getEnvironmentVariables: (id: string) => apiClient.getEnvironmentVariables(id),
  updateEnvironmentVariables: (id: string, variables: Record<string, string>) => 
    apiClient.updateEnvironmentVariables(id, variables),
  getSecrets: (id: string) => apiClient.getSecrets(id),
  setSecret: (id: string, key: string, value: string) => apiClient.setSecret(id, key, value),
  deleteSecret: (id: string, key: string) => apiClient.deleteSecret(id, key),
}