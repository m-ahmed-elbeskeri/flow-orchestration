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
  private authToken: string | null = null

  constructor() {
    this.client = axios.create({
      baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
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
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`
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
          // Optionally redirect to login
        }
        return Promise.reject(error)
      }
    )
  }

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await this.client.get('/api/v1/health')
    return response.data
  }

  async getSystemInfo(): Promise<any> {
    const response = await this.client.get('/api/v1/system/info')
    return response.data
  }

  async login(credentials: { username: string; password: string }): Promise<{ token: string; user: any }> {
    const response = await this.client.post('/api/v1/auth/login', credentials)
    const { token } = response.data
    this.setAuthToken(token)
    return response.data
  }

  async logout(): Promise<void> {
    await this.client.post('/api/v1/auth/logout')
    this.clearAuthToken()
  }

  async refreshToken(): Promise<{ token: string }> {
    const response = await this.client.post('/api/v1/auth/refresh')
    const { token } = response.data
    this.setAuthToken(token)
    return response.data
  }

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
    const response = await this.client.post('/api/v1/workflows/generate-code-from-yaml', { yaml_content: yamlContent })
    return response.data
  }

  async executeWorkflow(workflowId: string, params?: any): Promise<{ execution_id: string; status: string; message?: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/execute`, { parameters: params })
    return response.data
  }

  async getWorkflowExecutions(workflowId: string, params: PaginationParams = {}): Promise<{
    executions: WorkflowExecution[]
    total: number
    limit: number
    offset: number
  }> {
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions`, { params })
    return response.data
  }

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
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}/events`, { params })
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
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}/logs`, { params })
    return response.data
  }

  async pauseExecution(workflowId: string, executionId: string): Promise<{ success: boolean; message?: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/pause`)
    return response.data
  }

  async resumeExecution(workflowId: string, executionId: string): Promise<{ success: boolean; message?: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/resume`)
    return response.data
  }

  async cancelExecution(workflowId: string, executionId: string): Promise<{ success: boolean; message?: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/cancel`)
    return response.data
  }

  async retryExecution(workflowId: string, executionId: string): Promise<{ execution_id: string; status: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/retry`)
    return response.data
  }

  async retryState(workflowId: string, executionId: string, stateName: string): Promise<{ success: boolean; message?: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/states/${stateName}/retry`)
    return response.data
  }

  async skipState(workflowId: string, executionId: string, stateName: string): Promise<{ success: boolean; message?: string }> {
    const response = await this.client.post(`/api/v1/workflows/${workflowId}/executions/${executionId}/states/${stateName}/skip`)
    return response.data
  }

  async listTemplates(): Promise<WorkflowTemplate[]> {
    const response = await this.client.get('/api/v1/workflows/templates')
    return response.data
  }

  async getTemplate(templateId: string): Promise<WorkflowTemplate> {
    const response = await this.client.get(`/api/v1/workflows/templates/${templateId}`)
    return response.data
  }

  async createWorkflowFromTemplate(templateId: string, variables: Record<string, any>): Promise<WorkflowCreateResponse> {
    const response = await this.client.post('/api/v1/workflows/from-template', { template: templateId, variables })
    return response.data
  }

  async getWorkflowExamples(): Promise<any[]> {
    const response = await this.client.get('/api/v1/workflows/examples')
    return response.data
  }

  async importYamlWorkflow(yamlContent: string): Promise<{ workflow_id: string; status: string }> {
    const response = await this.client.post('/api/v1/workflows/import-yaml', { yaml: yamlContent })
    return response.data
  }

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

  async installPlugin(pluginName: string): Promise<{ success: boolean; message?: string }> {
    const response = await this.client.post(`/api/v1/plugins/${pluginName}/install`)
    return response.data
  }

  async uninstallPlugin(pluginName: string): Promise<{ success: boolean; message?: string }> {
    const response = await this.client.delete(`/api/v1/plugins/${pluginName}`)
    return response.data
  }

  async scheduleWorkflow(workflowId: string, schedule: {
    cron?: string
    timezone?: string
    enabled?: boolean
    max_instances?: number
  }): Promise<{ schedule_id: string; status: string }> {
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
    const response = await this.client.get(`/api/v1/workflows/${workflowId}/executions/${executionId}/export`, {
      params: { format },
      responseType: 'blob'
    })
    return response.data
  }

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

  createWebSocket(path: string, protocols?: string[]): WebSocket {
    const baseUrl = this.client.defaults.baseURL?.replace(/^http/, 'ws') || 'ws://localhost:8000'
    return new WebSocket(`${baseUrl}${path}`, protocols)
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.healthCheck()
      return true
    } catch (error) {
      console.error('API connection test failed:', error)
      return false
    }
  }

  setAuthToken(token: string): void {
    this.authToken = token
    localStorage.setItem('auth_token', token)
  }

  clearAuthToken(): void {
    this.authToken = null
    localStorage.removeItem('auth_token')
  }

  getAuthToken(): string | null {
    return this.authToken || localStorage.getItem('auth_token')
  }

  isAuthenticated(): boolean {
    return !!this.getAuthToken()
  }

  // Method aliases for shorter names (optional)
  list = this.listWorkflows
  create = this.createWorkflow
  delete = this.deleteWorkflow
  update = this.updateWorkflow
  execute = this.executeWorkflow
}

// Create and export a singleton instance
export const workflowApi = new ApiClient()

// Also export the class for testing or custom instances
export { ApiClient }
export default workflowApi