// ui/src/store/workflow.ts - NEW
import { create } from 'zustand'

interface WorkflowStore {
    workflows: Workflow[]
    currentWorkflow: Workflow | null
    executions: Record<string, WorkflowExecution[]>
    isLoading: boolean
    error: string | null
    
    // Actions
    fetchWorkflows: () => Promise<void>
    createWorkflow: (workflow: WorkflowCreateRequest) => Promise<void>
    executeWorkflow: (workflowId: string) => Promise<void>
    selectWorkflow: (workflow: Workflow) => void
    clearError: () => void
}

export const useWorkflowStore = create<WorkflowStore>((set, get) => ({
    workflows: [],
    currentWorkflow: null,
    executions: {},
    isLoading: false,
    error: null,
    
    fetchWorkflows: async () => {
        set({ isLoading: true, error: null })
        try {
            const workflows = await workflowApi.listWorkflows()
            set({ workflows: workflows.items, isLoading: false })
        } catch (error) {
            set({ error: error.message, isLoading: false })
        }
    },
    
    createWorkflow: async (workflowData) => {
        set({ isLoading: true, error: null })
        try {
            const workflow = await workflowApi.createWorkflow(workflowData)
            const workflows = get().workflows
            set({ 
                workflows: [...workflows, workflow], 
                isLoading: false 
            })
        } catch (error) {
            set({ error: error.message, isLoading: false })
        }
    },
    
    // Add other actions...
}))