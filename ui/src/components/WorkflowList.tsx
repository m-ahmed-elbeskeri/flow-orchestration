import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Plus, Play, Trash2, Clock, CheckCircle, XCircle, FileText, Download, Code } from 'lucide-react'
import { workflowApi } from '../api/client'
import WorkflowYamlCreator from './WorkflowYamlCreator'
import { downloadFile, downloadBase64File } from '../utils/download'
import type { WorkflowListItem } from '../types/workflow'

export default function WorkflowList() {
  const [showCreator, setShowCreator] = useState(false)
  const [generatingCode, setGeneratingCode] = useState<string | null>(null)

  const { data: workflows, isLoading, error, refetch } = useQuery({
    queryKey: ['workflows'],
    queryFn: async () => {
      console.log('🔄 Fetching workflows...')
      try {
        const response = await workflowApi.list()
        console.log('✅ Workflows fetched successfully:', response.data)
        return response.data
      } catch (error) {
        console.error('❌ Failed to fetch workflows:', error)
        throw error
      }
    },
    retry: 1,
    retryDelay: 1000,
  })

  const handleGenerateCode = async (workflowId: string, workflowName: string) => {
    try {
      setGeneratingCode(workflowId)
      console.log('🔧 Generating code for workflow:', workflowId)
      
      const response = await workflowApi.generateCode(workflowId)
      console.log('✅ Code generated:', response.data)
      
      if (response.data.success) {
        if (response.data.content) {
          // Download as zip file
          downloadBase64File(
            response.data.content,
            response.data.file_name,
            'application/zip'
          )
          alert(`Code package downloaded: ${response.data.file_name}`)
        } else if (response.data.python_code) {
          // Download as Python file
          downloadFile(
            response.data.python_code,
            response.data.file_name,
            'text/x-python'
          )
          alert(`Python code downloaded: ${response.data.file_name}`)
        }
      } else {
        alert('Code generation failed: ' + response.data.message)
      }
      
    } catch (error: any) {
      console.error('❌ Code generation failed:', error)
      
      let errorMessage = 'Unknown error'
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
      } else if (error.message) {
        errorMessage = error.message
      }
      
      alert(`Failed to generate code: ${errorMessage}`)
    } finally {
      setGeneratingCode(null)
    }
  }

  const handleDelete = async (id: string) => {
    if (confirm('Are you sure you want to delete this workflow?')) {
      try {
        await workflowApi.delete(id)
        refetch()
        alert('Workflow deleted successfully')
      } catch (error) {
        console.error('Failed to delete workflow:', error)
        alert('Failed to delete workflow')
      }
    }
  }

  const handleExecute = async (id: string) => {
    try {
      await workflowApi.execute(id)
      alert('Workflow execution started')
      refetch()
    } catch (error) {
      console.error('Failed to execute workflow:', error)
      alert('Failed to execute workflow')
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />
      case 'running':
        return <Clock className="h-5 w-5 text-blue-500 animate-spin" />
      default:
        return <Clock className="h-5 w-5 text-gray-500" />
    }
  }

  // Test API function (keep existing)
  const testApi = async () => {
    try {
      console.log('🧪 Testing API connection...')
      const response = await fetch('http://localhost:8000/api/v1/health')
      const data = await response.json()
      console.log('✅ Health check result:', data)
      alert(`API Health: ${data.status}`)
    } catch (error) {
      console.error('❌ Health check failed:', error)
      alert('API connection failed - check console')
    }
  }

  if (showCreator) {
    return (
      <WorkflowYamlCreator
        onSuccess={() => {
          setShowCreator(false)
          refetch()
        }}
        onCancel={() => setShowCreator(false)}
      />
    )
  }

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <div>Loading workflows...</div>
          <div className="mt-4">
            <button
              onClick={testApi}
              className="bg-gray-600 text-white px-4 py-2 rounded text-sm"
            >
              🧪 Test API Connection
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-semibold mb-2">Error loading workflows</h3>
          <p className="text-red-600 mb-4">{error.message}</p>
          <div className="flex gap-2">
            <button
              onClick={() => refetch()}
              className="bg-red-600 text-white px-4 py-2 rounded text-sm"
            >
              Retry
            </button>
            <button
              onClick={testApi}
              className="bg-gray-600 text-white px-4 py-2 rounded text-sm"
            >
              Test API
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold">Workflows</h1>
          <p className="text-gray-600">Manage and execute your YAML-based workflows</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={testApi}
            className="bg-gray-600 text-white px-3 py-2 rounded text-sm"
          >
            🧪 Test API
          </button>
          <button
            onClick={() => setShowCreator(true)}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            <Plus className="h-4 w-4" />
            New Workflow
          </button>
        </div>
      </div>

      <div className="grid gap-4">
        {!workflows || workflows.length === 0 ? (
          <div className="text-center py-12 bg-gray-50 rounded-lg">
            <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No workflows found</h3>
            <p className="text-gray-600 mb-4">Create your first YAML workflow to get started.</p>
            <button
              onClick={() => setShowCreator(true)}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
            >
              Create Workflow
            </button>
          </div>
        ) : (
          workflows.map((workflow: WorkflowListItem) => (
            <div key={workflow.workflow_id} className="border rounded-lg p-4 bg-white shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex items-start gap-4">
                  {getStatusIcon(workflow.status)}
                  <div>
                    <h3 className="font-semibold text-lg">{workflow.name}</h3>
                    <p className="text-gray-600 text-sm">{workflow.description}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                      <span>ID: {workflow.workflow_id.substring(0, 8)}...</span>
                      <span>Status: {workflow.status}</span>
                      {workflow.states_count && <span>States: {workflow.states_count}</span>}
                      <span>Created: {new Date(workflow.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleGenerateCode(workflow.workflow_id, workflow.name)}
                    disabled={generatingCode === workflow.workflow_id}
                    className="p-2 text-purple-600 hover:bg-purple-50 rounded-lg transition-colors disabled:opacity-50"
                    title="Generate and download code"
                  >
                    {generatingCode === workflow.workflow_id ? (
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-purple-600 border-t-transparent"></div>
                    ) : (
                      <Code className="h-4 w-4" />
                    )}
                  </button>
                  
                  <button
                    onClick={() => handleExecute(workflow.workflow_id)}
                    className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                    title="Execute workflow"
                  >
                    <Play className="h-4 w-4" />
                  </button>
                  
                  <button
                    onClick={() => handleDelete(workflow.workflow_id)}
                    className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    title="Delete workflow"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}