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
  private client: AxiosInstance
  private baseURL: string

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000'
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        const token = this.getAuthToken()
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => Promise.reject(error)
    )

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          this.clearAuthToken()
          window.location.href = '/login'
        }
        return Promise.reject(this.handleApiError(error))
      }
    )
  }

  private handleApiError(error: AxiosError): ApiError {
    if (error.response) {
      return {
        message: error.response.data?.message || error.message,
        code: error.response.data?.code,
        status: error.response.status,
        details: error.response.data?.details
      }
    } else if (error.request) {
      return {
        message: 'Network error - please check your connection',
        code: 'NETWORK_ERROR'
      }
    } else {
      return {
        message: error.message || 'Unknown error occurred',
        code: 'UNKNOWN_ERROR'
      }
    }
  }

  // Authentication methods
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

  // System methods
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await this.client.get('/api/v1/health')
    return response.data
  }

  async getSystemInfo(): Promise<SystemInfo> {
    const response = await this.client.get('/api/v1/system/info')
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
    const response = await this.client.post('/api/v1/auth/login', credentials)
    return response.data
  }

  async logout(): Promise<void> {
    await this.client.post('/api/v1/auth/logout')
    this.clearAuthToken()
  }

  async refreshToken(): Promise<{ token: string }> {
    const response = await this.client.post('/api/v1/auth/refresh')
    return response.data
  }

  // Workflow management
  async listWorkflows(params: PaginationParams & FilterParams = {}): Promise<PaginatedResponse<WorkflowListItem>> {
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

  // Workflow validation and code generation
  async validateWorkflow(yamlContent: string): Promise<ValidationResult> {
    const response = await this.client.post('/api/v1/workflows/validate-yaml', { yaml: yamlContent })
    return response.data
  }

  async validateWorkflowById(workflowId: string): Promise<ValidationResult> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/validate`)
    return response.data
  }

  async generateCodeFromWorkflow(workflowId: string): Promise<CodeGenerationResult> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/generate-code`)
    return response.data
  }

  async generateCodeFromYaml(yamlContent: string): Promise<CodeGenerationResult> {
    const response = await this.client.post('/api/v1/workflows/generate-code-from-yaml', { yaml: yamlContent })
    return response.data
  }

  // Workflow execution
  async executeWorkflow(workflowId: string, params?: any): Promise<{ execution_id: string; status: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/execute`, {
      parameters: params,
      priority: 1
    })
    return response.data
  }

  async getWorkflowExecutions(
    workflowId: string, 
    params: PaginationParams = {}
  ): Promise<PaginatedResponse<WorkflowExecution>> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions`, { params })
    return response.data
  }

  // Execution monitoring - These are the key methods for ExecutionMonitor
  async getExecution(workflowId: string, executionId: string): Promise<WorkflowExecution> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}`)
    return response.data
  }

  async getExecutionStates(workflowId: string, executionId: string): Promise<ExecutionState[]> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}/states`)
    return response.data || []
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
    return response.data || []
  }

  async getExecutionAlerts(workflowId: string, executionId: string): Promise<Alert[]> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}/alerts`)
    return response.data || []
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
    const response = await this.client.get(
      `/api/v1/workflows/${workflowId}/executions/${executionId}/logs`,
      { params }
    )
    return response.data || []
  }

  // Execution control
  async pauseExecution(workflowId: string, executionId: string): Promise<{ status: string; message: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/pause`)
    return response.data
  }

  async resumeExecution(workflowId: string, executionId: string): Promise<{ status: string; message: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/resume`)
    return response.data
  }

  async cancelExecution(workflowId: string, executionId: string): Promise<{ status: string; message: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/cancel`)
    return response.data
  }

  async retryExecution(workflowId: string, executionId: string): Promise<{ execution_id: string; status: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/retry`)
    return response.data
  }

  async retryState(workflowId: string, executionId: string, stateName: string): Promise<{ status: string }> {
    const response = await this.client.post(
      `/api/v1/workflows/${workflowId}/executions/${executionId}/states/${stateName}/retry`
    )
    return response.data
  }

  async skipState(workflowId: string, executionId: string, stateName: string): Promise<{ status: string }> {
    const response = await this.client.post(
      `/api/v1/workflows/${workflowId}/executions/${executionId}/states/${stateName}/skip`
    )
    return response.data
  }

  // Templates and examples
  async listTemplates(): Promise<WorkflowTemplate[]> {
    const response = await this.client.get('/api/v1/workflows/templates')
    return response.data || []
  }

  async getTemplate(templateId: string): Promise<WorkflowTemplate> {
    const response = await this.client.get(`/api/v1/workflows/templates/${templateId}`)
    return response.data
  }

  async createWorkflowFromTemplate(
    templateId: string, 
    variables: Record<string, any>
  ): Promise<WorkflowCreateResponse> {
    const response = await this.client.post('/api/v1/workflows/from-template', {
      template: templateId,
      variables
    })
    return response.data
  }

  async getWorkflowExamples(): Promise<any[]> {
    const response = await this.client.get('/api/v1/workflows/examples')
    return response.data || []
  }

  // Import/Export
  async importYamlWorkflow(yamlContent: string): Promise<{ workflow: Workflow }> {
    const response = await this.client.post('/api/v1/workflows/import-yaml', { yaml: yamlContent })
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

  // Nodes and plugins
  async getAvailableNodes(): Promise<NodeType[]> {
    const response = await this.client.get('/api/v1/workflows/nodes')
    return response.data || []
  }

  async getNodeDocumentation(nodeType: string): Promise<any> {
    const response = await this.client.get(`/api/v1/workflows/nodes/${nodeType}/docs`)
    return response.data
  }

  async listPlugins(): Promise<any[]> {
    const response = await this.client.get('/api/v1/plugins')
    return response.data || []
  }

  async getPluginInfo(pluginName: string): Promise<any> {
    const response = await this.client.get(`/api/v1/plugins/${pluginName}`)
    return response.data
  }

  async installPlugin(pluginName: string): Promise<{ success: boolean; message: string }> {
    const response = await this.client.post(`/api/v1/plugins/${pluginName}/install`)
    return response.data
  }

  async uninstallPlugin(pluginName: string): Promise<{ success: boolean; message: string }> {
    const response = await this.client.delete(`/api/v1/plugins/${pluginName}`)
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

  // Analytics and metrics
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

  // Environment and secrets
  async getEnvironmentVariables(workflowId: string): Promise<Record<string, string>> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/environment`)
    return response.data || {}
  }

  async updateEnvironmentVariables(workflowId: string, variables: Record<string, string>): Promise<void> {
    await this.client.put(`/api/v1/workflows/${workflowId}/environment`, { variables })
  }

  async getSecrets(workflowId: string): Promise<string[]> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/secrets`)
    return response.data || []
  }

  async setSecret(workflowId: string, key: string, value: string): Promise<void> {
    await this.client.post(`/api/v1/workflows/${workflowId}/secrets`, { key, value })
  }

  async deleteSecret(workflowId: string, key: string): Promise<void> {
    await this.client.delete(`/api/v1/workflows/${workflowId}/secrets/${key}`)
  }

  // File management
  async uploadFile(file: File, workflowId?: string): Promise<{ file_id: string; url: string }> {
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

  // WebSocket connection
  createWebSocket(path: string, protocols?: string[]): WebSocket {
    const wsUrl = this.baseURL.replace(/^http/, 'ws') + path
    return new WebSocket(wsUrl, protocols)
  }
}

// Export a singleton instance
export const workflowApi = new ApiClient()
export default workflowApi