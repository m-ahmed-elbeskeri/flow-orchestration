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
  NodeType,
  NodeAppearance,
  LogEntry,
  PaginatedResponse,
  ApiError,
  SystemInfo
} from '../types/workflow'

interface ApiResponse<T = any> {
  data?: T
  message?: string
  success?: boolean
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
  private instance: AxiosInstance
  private authToken: string | null = null

  constructor() {
    this.instance = axios.create({
      baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  private setupInterceptors(): void {
    // Request interceptor for auth
    this.instance.interceptors.request.use(
      (config) => {
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`
        }
        return config
      },
      (error) => Promise.reject(error)
    )

    // Response interceptor for error handling
    this.instance.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const apiError = this.handleApiError(error)
        return Promise.reject(apiError)
      }
    )
  }

  private handleApiError(error: AxiosError): ApiError {
    const status = error.response?.status
    const data = error.response?.data as any

    return {
      message: data?.message || error.message || 'An unexpected error occurred',
      code: data?.code || 'UNKNOWN_ERROR',
      details: data?.details || null,
      status: status || 0,
    }
  }

  // Auth methods
  setAuthToken(token: string): void {
    this.authToken = token
    localStorage.setItem('auth_token', token)
  }

  clearAuthToken(): void {
    this.authToken = null
    localStorage.removeItem('auth_token')
  }

  getAuthToken(): string | null {
    if (!this.authToken) {
      this.authToken = localStorage.getItem('auth_token')
    }
    return this.authToken
  }

  isAuthenticated(): boolean {
    return !!this.getAuthToken()
  }

  // Health and system info
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await this.instance.get('/health')
    return response.data
  }

  async getSystemInfo(): Promise<SystemInfo> {
    const response = await this.instance.get('/system/info')
    return response.data
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck()
      return true
    } catch {
      return false
    }
  }

  // Authentication
  async login(credentials: { username: string; password: string }): Promise<{ token: string; user: any }> {
    const response = await this.instance.post('/auth/login', credentials)
    const { token } = response.data
    this.setAuthToken(token)
    return response.data
  }

  async logout(): Promise<void> {
    try {
      await this.instance.post('/auth/logout')
    } finally {
      this.clearAuthToken()
    }
  }

  async refreshToken(): Promise<{ token: string }> {
    const response = await this.instance.post('/auth/refresh')
    const { token } = response.data
    this.setAuthToken(token)
    return response.data
  }

  // Workflow management
  async listWorkflows(params: PaginationParams & FilterParams = {}): Promise<PaginatedResponse<WorkflowListItem>> {
    const response = await this.instance.get('/workflows', { params })
    return response.data
  }

  async getWorkflow(workflowId: string): Promise<Workflow> {
    const response = await this.instance.get(`/workflows/${workflowId}`)
    return response.data
  }

  async createWorkflow(workflow: WorkflowCreateRequest): Promise<WorkflowCreateResponse> {
    const response = await this.instance.post('/workflows', workflow)
    return response.data
  }

  async createWorkflowFromYaml(data: {
    name: string
    yaml_content: string
    auto_start?: boolean
  }): Promise<WorkflowCreateResponse> {
    const response = await this.instance.post('/workflows/from-yaml', data)
    return response.data
  }

  async updateWorkflow(workflowId: string, updates: Partial<Workflow>): Promise<Workflow> {
    const response = await this.instance.put(`/workflows/${workflowId}`, updates)
    return response.data
  }

  async deleteWorkflow(workflowId: string): Promise<void> {
    await this.instance.delete(`/workflows/${workflowId}`)
  }

  async duplicateWorkflow(workflowId: string, name?: string): Promise<WorkflowCreateResponse> {
    const response = await this.instance.post(`/workflows/${workflowId}/duplicate`, { name })
    return response.data
  }

  // Workflow validation and code generation
  async validateWorkflow(yamlContent: string): Promise<ValidationResult> {
    const response = await this.instance.post('/workflows/validate-yaml', { yaml: yamlContent })
    return response.data
  }

  async validateWorkflowById(workflowId: string): Promise<ValidationResult> {
    const response = await this.instance.post(`/workflows/${workflowId}/validate`)
    return response.data
  }

  async generateCodeFromWorkflow(workflowId: string): Promise<CodeGenerationResult> {
    const response = await this.instance.post(`/workflows/${workflowId}/generate-code`)
    return response.data
  }

  async generateCodeFromYaml(yamlContent: string): Promise<CodeGenerationResult> {
    const response = await this.instance.post('/workflows/generate-code', { yaml_content: yamlContent })
    return response.data
  }

  // Workflow execution
  async executeWorkflow(workflowId: string, params?: any): Promise<{ execution_id: string }> {
    const response = await this.instance.post(`/workflows/${workflowId}/execute`, {
      parameters: params || {},
      priority: 1
    })
    return response.data
  }

  async getWorkflowExecutions(
    workflowId: string,
    params: PaginationParams = {}
  ): Promise<PaginatedResponse<WorkflowExecution>> {
    const response = await this.instance.get(`/workflows/${workflowId}/executions`, { params })
    return response.data
  }

  async getExecution(workflowId: string, executionId: string): Promise<WorkflowExecution> {
    const response = await this.instance.get(`/workflows/${workflowId}/executions/${executionId}`)
    return response.data
  }

  // Execution monitoring
  async getExecutionStates(workflowId: string, executionId: string): Promise<ExecutionState[]> {
    const response = await this.instance.get(`/workflows/${workflowId}/executions/${executionId}/states`)
    return response.data
  }

  async getExecutionMetrics(workflowId: string, executionId: string): Promise<ExecutionMetrics> {
    const response = await this.instance.get(`/workflows/${workflowId}/executions/${executionId}/metrics`)
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
    const response = await this.instance.get(`/workflows/${workflowId}/executions/${executionId}/events`, { params })
    return response.data
  }

  async getExecutionAlerts(workflowId: string, executionId: string): Promise<Alert[]> {
    const response = await this.instance.get(`/workflows/${workflowId}/executions/${executionId}/alerts`)
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
  ): Promise<LogEntry[]> {
    const response = await this.instance.get(`/workflows/${workflowId}/executions/${executionId}/logs`, { params })
    return response.data
  }

  // Execution control
  async pauseExecution(workflowId: string, executionId: string): Promise<{ success: boolean; checkpoint_id?: string }> {
    const response = await this.instance.post(`/workflows/${workflowId}/executions/${executionId}/pause`)
    return response.data
  }

  async resumeExecution(workflowId: string, executionId: string): Promise<{ success: boolean }> {
    const response = await this.instance.post(`/workflows/${workflowId}/executions/${executionId}/resume`)
    return response.data
  }

  async cancelExecution(workflowId: string, executionId: string): Promise<{ success: boolean }> {
    const response = await this.instance.post(`/workflows/${workflowId}/executions/${executionId}/cancel`)
    return response.data
  }

  async retryExecution(workflowId: string, executionId: string): Promise<{ execution_id: string }> {
    const response = await this.instance.post(`/workflows/${workflowId}/executions/${executionId}/retry`)
    return response.data
  }

  async retryState(workflowId: string, executionId: string, stateName: string): Promise<{ success: boolean }> {
    const response = await this.instance.post(`/workflows/${workflowId}/executions/${executionId}/states/${stateName}/retry`)
    return response.data
  }

  async skipState(workflowId: string, executionId: string, stateName: string): Promise<{ success: boolean }> {
    const response = await this.instance.post(`/workflows/${workflowId}/executions/${executionId}/states/${stateName}/skip`)
    return response.data
  }

  // Workflow status and control
  async getWorkflowStatus(workflowId: string): Promise<any> {
    const response = await this.instance.get(`/workflows/${workflowId}/status`)
    return response.data
  }

  async pauseWorkflow(workflowId: string): Promise<{ paused_executions: string[] }> {
    const response = await this.instance.post(`/workflows/${workflowId}/pause`)
    return response.data
  }

  async resumeWorkflow(workflowId: string): Promise<{ resumed_executions: string[] }> {
    const response = await this.instance.post(`/workflows/${workflowId}/resume`)
    return response.data
  }

  async stopWorkflow(workflowId: string): Promise<{ stopped_executions: string[] }> {
    const response = await this.instance.post(`/workflows/${workflowId}/stop`)
    return response.data
  }

  // Templates
  async listTemplates(): Promise<WorkflowTemplate[]> {
    const response = await this.instance.get('/workflows/templates')
    return response.data
  }

  async getTemplate(templateId: string): Promise<WorkflowTemplate> {
    const response = await this.instance.get(`/workflows/templates/${templateId}`)
    return response.data
  }

  async createWorkflowFromTemplate(
    templateId: string,
    variables: Record<string, any>
  ): Promise<WorkflowCreateResponse> {
    const response = await this.instance.post(`/workflows/templates/${templateId}/create`, { variables })
    return response.data
  }

  async getWorkflowExamples(): Promise<any[]> {
    const response = await this.instance.get('/workflows/examples')
    return response.data
  }

  // Import/Export
  async importYamlWorkflow(yamlContent: string): Promise<{ workflow_id: string; visual_data: any }> {
    const response = await this.instance.post('/workflows/import-yaml', { yaml: yamlContent })
    return response.data
  }

  async exportWorkflowData(workflowId: string, format: 'json' | 'csv' | 'xlsx' = 'json'): Promise<Blob> {
    const response = await this.instance.get(`/workflows/${workflowId}/export`, {
      params: { format },
      responseType: 'blob'
    })
    return response.data
  }

  async exportExecutionData(workflowId: string, executionId: string, format: 'json' | 'csv' = 'json'): Promise<Blob> {
    const response = await this.instance.get(`/workflows/${workflowId}/executions/${executionId}/export`, {
      params: { format },
      responseType: 'blob'
    })
    return response.data
  }

  // Node registry (NEW)
  async getAvailableNodes(): Promise<Record<string, NodeAppearance>> {
    const response = await this.instance.get('/nodes/available')
    return response.data.data
  }

  async getNodeCategories(): Promise<Record<string, string[]>> {
    const response = await this.instance.get('/nodes/categories')
    return response.data.data
  }

  async getNodeDetails(nodeType: string): Promise<NodeAppearance> {
    const response = await this.instance.get(`/nodes/${nodeType}`)
    return response.data.data
  }

  async refreshNodeRegistry(): Promise<void> {
    await this.instance.post('/nodes/refresh')
  }

  async getNodeDocumentation(nodeType: string): Promise<any> {
    const response = await this.instance.get(`/nodes/${nodeType}/docs`)
    return response.data
  }

  // Plugin management
  async listPlugins(): Promise<any[]> {
    const response = await this.instance.get('/plugins')
    return response.data
  }

  async getPluginInfo(pluginName: string): Promise<any> {
    const response = await this.instance.get(`/plugins/${pluginName}`)
    return response.data
  }

  async installPlugin(pluginName: string): Promise<{ success: boolean; message: string }> {
    const response = await this.instance.post(`/plugins/${pluginName}/install`)
    return response.data
  }

  async uninstallPlugin(pluginName: string): Promise<{ success: boolean; message: string }> {
    const response = await this.instance.post(`/plugins/${pluginName}/uninstall`)
    return response.data
  }

  // Scheduling
  async scheduleWorkflow(workflowId: string, schedule: {
    cron?: string
    timezone?: string
    enabled?: boolean
    max_instances?: number
  }): Promise<{ schedule_id: string }> {
    const response = await this.instance.post(`/workflows/${workflowId}/schedule`, schedule)
    return response.data
  }

  async getWorkflowSchedule(workflowId: string): Promise<any> {
    const response = await this.instance.get(`/workflows/${workflowId}/schedule`)
    return response.data
  }

  async updateWorkflowSchedule(workflowId: string, schedule: any): Promise<any> {
    const response = await this.instance.put(`/workflows/${workflowId}/schedule`, schedule)
    return response.data
  }

  async deleteWorkflowSchedule(workflowId: string): Promise<void> {
    await this.instance.delete(`/workflows/${workflowId}/schedule`)
  }

  // Analytics
  async getWorkflowAnalytics(workflowId: string, params: {
    start_date?: string
    end_date?: string
    granularity?: 'hour' | 'day' | 'week' | 'month'
  } = {}): Promise<any> {
    const response = await this.instance.get(`/workflows/${workflowId}/analytics`, { params })
    return response.data
  }

  async getSystemMetrics(params: {
    start_date?: string
    end_date?: string
    metrics?: string[]
  } = {}): Promise<any> {
    const response = await this.instance.get('/system/metrics', { params })
    return response.data
  }

  // Environment and secrets
  async getEnvironmentVariables(workflowId: string): Promise<Record<string, string>> {
    const response = await this.instance.get(`/workflows/${workflowId}/environment`)
    return response.data
  }

  async updateEnvironmentVariables(workflowId: string, variables: Record<string, string>): Promise<void> {
    await this.instance.put(`/workflows/${workflowId}/environment`, { variables })
  }

  async getSecrets(workflowId: string): Promise<string[]> {
    const response = await this.instance.get(`/workflows/${workflowId}/secrets`)
    return response.data
  }

  async setSecret(workflowId: string, key: string, value: string): Promise<void> {
    await this.instance.post(`/workflows/${workflowId}/secrets`, { key, value })
  }

  async deleteSecret(workflowId: string, key: string): Promise<void> {
    await this.instance.delete(`/workflows/${workflowId}/secrets/${key}`)
  }

  // File operations
  async uploadFile(file: File, workflowId?: string): Promise<{ file_id: string; url: string }> {
    const formData = new FormData()
    formData.append('file', file)
    if (workflowId) {
      formData.append('workflow_id', workflowId)
    }

    const response = await this.instance.post('/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  async downloadFile(fileId: string): Promise<Blob> {
    const response = await this.instance.get(`/files/${fileId}`, {
      responseType: 'blob'
    })
    return response.data
  }

  async deleteFile(fileId: string): Promise<void> {
    await this.instance.delete(`/files/${fileId}`)
  }

  // WebSocket connections
  createWebSocket(path: string, protocols?: string[]): WebSocket {
    const wsUrl = this.instance.defaults.baseURL?.replace('http', 'ws') + path
    const ws = new WebSocket(wsUrl, protocols)
    
    // Add auth token to connection if available
    if (this.authToken) {
      ws.addEventListener('open', () => {
        ws.send(JSON.stringify({
          type: 'auth',
          token: this.authToken
        }))
      })
    }
    
    return ws
  }
}

// Create and export a singleton instance
export const workflowApi = new ApiClient()
export default workflowApi