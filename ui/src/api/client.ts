import axios from 'axios'
import type { 
  Workflow, 
  WorkflowExecution, 
  WorkflowCreateRequest, 
  WorkflowCreateResponse,
  WorkflowListItem 
} from '../types/workflow'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

console.log('🔧 API Base URL:', API_BASE_URL)

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 second timeout for code generation
})

// Add request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log('📤 API Request:', {
      method: config.method?.toUpperCase(),
      url: config.url,
      baseURL: config.baseURL,
      data: config.data ? 'Present' : 'None',
      headers: config.headers
    })
    return config
  },
  (error) => {
    console.error('📤 Request Error:', error)
    return Promise.reject(error)
  }
)

// Add response interceptor for debugging
api.interceptors.response.use(
  (response) => {
    console.log('📥 API Response:', {
      status: response.status,
      url: response.config.url,
      dataSize: response.data ? Object.keys(response.data).length : 0
    })
    return response
  },
  (error) => {
    console.error('📥 API Error:', {
      status: error.response?.status,
      statusText: error.response?.statusText,
      url: error.config?.url,
      data: error.response?.data,
      message: error.message
    })
    return Promise.reject(error)
  }
)

export const workflowApi = {
  // Health and testing
  health: () => {
    console.log('🩺 Checking API health...')
    return api.get('/health')
  },

  test: () => {
    console.log('🧪 Testing API connection...')
    return api.post('/test-create')
  },

  // Basic workflow operations
  list: () => {
    console.log('📋 Fetching workflows list...')
    return api.get<WorkflowListItem[]>('/workflows')
  },
  
  get: (id: string) => {
    console.log('📋 Fetching workflow:', id)
    return api.get<Workflow>(`/workflows/${id}`)
  },
  
  // YAML workflow operations
  createFromYaml: (data: WorkflowCreateRequest) => {
    console.log('📋 Creating workflow from YAML:', { name: data.name, yaml_length: data.yaml_content.length })
    return api.post<WorkflowCreateResponse>('/workflows/from-yaml', data)
  },
  
  // Legacy workflow creation
  create: (data: any) => {
    console.log('📋 Creating legacy workflow:', data)
    return api.post<WorkflowCreateResponse>('/workflows', data)
  },
  
  update: (id: string, data: Partial<Workflow>) => {
    console.log('📋 Updating workflow:', id, data)
    return api.put<Workflow>(`/workflows/${id}`, data)
  },
    
  delete: (id: string) => {
    console.log('📋 Deleting workflow:', id)
    return api.delete(`/workflows/${id}`)
  },

  // Execution operations
  execute: (id: string) => {
    console.log('▶️ Executing workflow:', id)
    return api.post<WorkflowExecution>(`/workflows/${id}/execute`)
  },
  
  getStatus: (id: string) => {
    console.log('📊 Getting workflow status:', id)
    return api.get(`/workflows/${id}/status`)
  },
  
  pause: (id: string) => {
    console.log('⏸️ Pausing workflow:', id)
    return api.post(`/workflows/${id}/pause`)
  },
  
  resume: (id: string) => {
    console.log('▶️ Resuming workflow:', id)
    return api.post(`/workflows/${id}/resume`)
  },

  // YAML operations
  validate: (yaml: string) => {
    console.log('✅ Validating YAML (length:', yaml.length, 'chars)')
    return api.post('/workflows/validate-yaml', { yaml })
  },

  // Code generation operations
  generateCode: (workflowId: string) => {
    console.log('🔧 Generating code for workflow:', workflowId)
    return api.post(`/workflows/${workflowId}/generate-code`)
  },
  
  generateCodeFromYaml: (yamlContent: string) => {
    console.log('🔧 Generating code from YAML (length:', yamlContent.length, 'chars)')
    return api.post('/workflows/generate-code-from-yaml', { yaml_content: yamlContent })
  },
  
  // Templates and examples
  getExamples: () => {
    console.log('📚 Fetching workflow examples...')
    return api.get('/workflows/examples')
  },

  listTemplates: () => {
    console.log('📋 Fetching templates...')
    return api.get('/workflows/templates')
  },
  
  createFromTemplate: (template: string, variables: Record<string, any>) => {
    console.log('📋 Creating from template:', template, variables)
    return api.post('/workflows/from-template', { template, variables })
  },

  // Visual editor endpoints (for future use)
  getVisual: (id: string) => {
    console.log('🎨 Getting visual workflow:', id)
    return api.get(`/workflows/${id}/visual`)
  },
  
  saveVisual: (id: string, data: any) => {
    console.log('🎨 Saving visual workflow:', id)
    return api.put(`/workflows/${id}/visual`, data)
  },
  
  // Generator endpoints (advanced)
  generateCodeAdvanced: (workflow: any) => {
    console.log('🔧 Generating advanced code for workflow')
    return api.post('/workflows/generate-code', { workflow })
  },
  
  validateWorkflow: (workflow: any) => {
    console.log('✅ Validating workflow object')
    return api.post('/workflows/validate', { workflow })
  },
  
  importYaml: (yaml: string) => {
    console.log('📥 Importing YAML workflow')
    return api.post('/workflows/import-yaml', { yaml })
  },

  // Debug endpoints
  debug: (data: any) => {
    console.log('🐛 Debug endpoint call')
    return api.post('/debug-workflows', data)
  },

  // Simple workflow creation for testing
  createSimple: (name: string = "Test Workflow", agentName: string = "test_agent") => {
    console.log('📋 Creating simple workflow:', name, agentName)
    return api.post('/workflows/simple', null, { params: { name, agent_name: agentName } })
  }
}

export default api