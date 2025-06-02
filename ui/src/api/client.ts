// src/api/client.ts
import axios from 'axios'
import type { Workflow, WorkflowExecution } from '../types/workflow'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const workflowApi = {
  // Workflow CRUD
  list: () => api.get<Workflow[]>('/workflows'),
  get: (id: string) => api.get<Workflow>(`/workflows/${id}`),
  create: (data: Partial<Workflow>) => api.post<Workflow>('/workflows', data),
  update: (id: string, data: Partial<Workflow>) => api.put<Workflow>(`/workflows/${id}`, data),
  delete: (id: string) => api.delete(`/workflows/${id}`),
  
  // Execution
  execute: (id: string) => api.post<WorkflowExecution>(`/workflows/${id}/execute`),
  getExecution: (workflowId: string, executionId: string) => 
    api.get<WorkflowExecution>(`/workflows/${workflowId}/executions/${executionId}`),
  listExecutions: (workflowId: string) => 
    api.get<WorkflowExecution[]>(`/workflows/${workflowId}/executions`),
    
  // Visual editor
  getVisual: (id: string) => api.get(`/workflows/${id}/visual`),
  saveVisual: (id: string, data: any) => api.put(`/workflows/${id}/visual`, data),
}