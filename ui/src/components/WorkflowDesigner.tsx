// ui/src/components/WorkflowCreator.tsx
import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { 
  Code, Plus, Save, Play, FileText, Copy, Download, Wand2, ArrowLeft, 
  Settings, Eye, CheckCircle, AlertCircle, Palette, Edit3 
} from 'lucide-react'
import { workflowApi } from '../api/client'
import { downloadFile } from '../utils/download'
import type { WorkflowCreateRequest } from '../types/workflow'
import Editor from '@monaco-editor/react'
import VisualWorkflowDesigner from './VisualWorkflowDesigner'

type EditorMode = 'yaml' | 'visual'

export default function WorkflowCreator() {
  const navigate = useNavigate()
  const [mode, setMode] = useState<EditorMode>('yaml')
  const [yamlContent, setYamlContent] = useState('')
  const [workflowName, setWorkflowName] = useState('')
  const [description, setDescription] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [validationResult, setValidationResult] = useState<any>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [isGeneratingCode, setIsGeneratingCode] = useState(false)
  const editorRef = useRef<any>(null)

  const defaultTemplate = `name: my_workflow
version: "1.0.0"
description: "Describe what your workflow does"
author: "Your Name"

config:
  timeout: 300
  max_concurrent: 5

environment:
  variables:
    LOG_LEVEL: INFO
  secrets: []

states:
  - name: start
    type: builtin.start
    description: "Workflow starting point"
    transitions:
      - on_success: end

  - name: end
    type: builtin.end
    description: "Workflow completion"`

  const emailWorkflowTemplate = `name: email_processor
version: "1.0.0"
description: "Process urgent emails and send notifications"
author: "Team Workflow"

config:
  timeout: 600
  max_concurrent: 3

environment:
  variables:
    LOG_LEVEL: INFO
    EMAIL_CHECK_QUERY: "is:unread label:urgent"
  secrets:
    - GMAIL_CREDENTIALS
    - SLACK_TOKEN

states:
  - name: start
    type: builtin.start
    transitions:
      - on_success: fetch_emails

  - name: fetch_emails
    type: gmail.read_emails
    config:
      query: "is:unread label:urgent"
      max_results: 10
      mark_as_read: false
    dependencies:
      - name: start
        type: required
    transitions:
      - on_success: check_emails
      - on_failure: error_handler

  - name: check_emails
    type: builtin.conditional
    config:
      condition: len(context.get_state("fetched_emails", [])) > 0
    dependencies:
      - name: fetch_emails
    transitions:
      - on_true: send_notification
      - on_false: end

  - name: send_notification
    type: slack.send_message
    config:
      channel: "#alerts"
      message: "Found {{ fetched_emails|length }} urgent emails"
    dependencies:
      - name: check_emails
    transitions:
      - on_success: end
      - on_failure: error_handler

  - name: error_handler
    type: builtin.error_handler
    config:
      log_level: ERROR
      notify: true
    transitions:
      - on_complete: end

  - name: end
    type: builtin.end`

  const dataProcessingTemplate = `name: data_processing_pipeline
version: "1.0.0"
description: "Process and transform data with validation"
author: "Data Team"

config:
  timeout: 1200
  max_concurrent: 2

environment:
  variables:
    LOG_LEVEL: INFO
    BATCH_SIZE: "100"
  secrets:
    - DATABASE_URL
    - API_KEY

states:
  - name: start
    type: builtin.start
    transitions:
      - on_success: fetch_data

  - name: fetch_data
    type: webhook.http_request
    config:
      url: "https://api.example.com/data"
      method: "GET"
      headers:
        Authorization: "Bearer {{ api_key }}"
    dependencies:
      - name: start
        type: required
    transitions:
      - on_success: validate_data
      - on_failure: error_handler

  - name: validate_data
    type: builtin.conditional
    config:
      condition: len(context.get_state("response_data", [])) > 0
    dependencies:
      - name: fetch_data
    transitions:
      - on_true: transform_data
      - on_false: end

  - name: transform_data
    type: builtin.transform
    config:
      operation: "normalize"
      fields: ["name", "email", "created_at"]
    dependencies:
      - name: validate_data
    transitions:
      - on_success: store_data
      - on_failure: error_handler

  - name: store_data
    type: webhook.http_request
    config:
      url: "https://api.example.com/store"
      method: "POST"
      body: "{{ transformed_data }}"
    dependencies:
      - name: transform_data
    transitions:
      - on_success: end
      - on_failure: error_handler

  - name: error_handler
    type: builtin.error_handler
    config:
      log_level: ERROR
      retry: true
      max_retries: 3
    transitions:
      - on_complete: end

  - name: end
    type: builtin.end`

  const templates = {
    blank: { name: 'Blank Workflow', content: defaultTemplate },
    email: { name: 'Email Processing', content: emailWorkflowTemplate },
    data: { name: 'Data Processing', content: dataProcessingTemplate },
  }

  const handleEditorDidMount = (editor: any, monaco: any) => {
    editorRef.current = editor
    
    // Configure YAML language support
    monaco.languages.yaml?.configure({
      validate: true,
      schemas: [{
        uri: 'workflow-schema',
        fileMatch: ['*'],
        schema: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            version: { type: 'string' },
            description: { type: 'string' },
            author: { type: 'string' },
            config: { type: 'object' },
            environment: { type: 'object' },
            states: { type: 'array' }
          },
          required: ['name', 'version', 'states']
        }
      }]
    })

    if (!yamlContent) {
      setYamlContent(defaultTemplate)
    }
  }

  const handleTemplateSelect = (templateKey: string) => {
    const template = templates[templateKey as keyof typeof templates]
    if (template) {
      setYamlContent(template.content)
      // Extract name from template for auto-naming
      const nameMatch = template.content.match(/name:\s*(.+)/)
      if (nameMatch) {
        setWorkflowName(nameMatch[1].trim())
      }
    }
  }

  const extractNameFromYaml = (yaml: string): string => {
    const nameMatch = yaml.match(/name:\s*(.+)/)
    return nameMatch ? nameMatch[1].trim() : 'Untitled Workflow'
  }

  const validateWorkflow = async () => {
    if (!yamlContent.trim()) {
      setValidationResult({
        is_valid: false,
        errors: [{ message: 'Workflow content is empty', path: 'root' }],
        warnings: []
      })
      return
    }

    setIsValidating(true)
    try {
      const result = await workflowApi.validateWorkflow(yamlContent)
      setValidationResult(result)
    } catch (error: any) {
      setValidationResult({
        is_valid: false,
        errors: [{ message: error.message || 'Validation failed', path: 'root' }],
        warnings: []
      })
    } finally {
      setIsValidating(false)
    }
  }

  const handleGenerateCode = async () => {
    if (!yamlContent.trim()) return
    
    setIsGeneratingCode(true)
    try {
      const result = await workflowApi.generateCodeFromYaml(yamlContent)
      if (result.success && result.zip_content) {
        // Convert base64 to blob and download
        const binaryString = atob(result.zip_content)
        const bytes = new Uint8Array(binaryString.length)
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i)
        }
        const blob = new Blob([bytes], { type: 'application/zip' })
        downloadFile(blob, `${workflowName || 'workflow'}_generated.zip`, 'application/zip')
      }
    } catch (error) {
      console.error('Code generation failed:', error)
    } finally {
      setIsGeneratingCode(false)
    }
  }

  const handleCreate = async () => {
    const currentYaml = yamlContent.trim()
    if (!currentYaml) return

    // Auto-validate if not already validated
    if (!validationResult) {
      await validateWorkflow()
      return // Let user see validation results before proceeding
    }

    if (!validationResult.is_valid) {
      return // Don't create if validation failed
    }

    const finalName = workflowName || extractNameFromYaml(currentYaml)
    
    setIsCreating(true)
    try {
      const workflow: WorkflowCreateRequest = {
        name: finalName,
        description: description || 'Created with YAML editor',
        yaml_content: currentYaml,
        auto_start: false
      }

      const result = await workflowApi.createWorkflow(workflow)
      navigate('/workflows')
    } catch (error: any) {
      console.error('Failed to create workflow:', error)
    } finally {
      setIsCreating(false)
    }
  }

  const copyToClipboard = () => {
    navigator.clipboard.writeText(yamlContent)
  }

  const clearEditor = () => {
    setYamlContent('')
    setValidationResult(null)
    setWorkflowName('')
    setDescription('')
  }

  const getValidationStatus = () => {
    if (!validationResult) return null
    if (validationResult.is_valid) {
      return (
        <div className="flex items-center gap-2 text-green-600">
          <CheckCircle className="w-4 h-4" />
          <span className="text-sm">Valid workflow</span>
        </div>
      )
    } else {
      return (
        <div className="flex items-center gap-2 text-red-600">
          <AlertCircle className="w-4 h-4" />
          <span className="text-sm">{validationResult.errors?.length || 0} error(s)</span>
        </div>
      )
    }
  }

  const handleYamlChange = (value: string | undefined) => {
    setYamlContent(value || '')
    // Reset validation when content changes
    setValidationResult(null)
  }

  const handleVisualDesignerChange = (yaml: string) => {
    setYamlContent(yaml)
    setValidationResult(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/workflows')}
              className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:text-gray-900"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Workflows
            </button>
            <div className="h-6 w-px bg-gray-300" />
            <h1 className="text-xl font-semibold">Create New Workflow</h1>
          </div>
          
          <div className="flex items-center gap-3">
            {getValidationStatus()}
            <button
              onClick={validateWorkflow}
              disabled={isValidating || !yamlContent.trim()}
              className="flex items-center gap-2 px-3 py-2 border rounded-lg hover:bg-gray-50 disabled:opacity-50"
            >
              <Eye className="w-4 h-4" />
              {isValidating ? 'Validating...' : 'Validate'}
            </button>
            <button
              onClick={handleGenerateCode}
              disabled={isGeneratingCode || !yamlContent.trim()}
              className="flex items-center gap-2 px-3 py-2 border rounded-lg hover:bg-gray-50 disabled:opacity-50"
            >
              <Wand2 className="w-4 h-4" />
              {isGeneratingCode ? 'Generating...' : 'Generate Code'}
            </button>
            <button
              onClick={handleCreate}
              disabled={isCreating || !yamlContent.trim()}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              <Save className="w-4 h-4" />
              {isCreating ? 'Creating...' : 'Create Workflow'}
            </button>
          </div>
        </div>

        {/* Mode Tabs */}
        <div className="flex items-center gap-1 mt-4">
          <button
            onClick={() => setMode('yaml')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
              mode === 'yaml'
                ? 'bg-blue-100 text-blue-700 border border-blue-200'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            <Code className="w-4 h-4" />
            YAML Editor
          </button>
          <button
            onClick={() => setMode('visual')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
              mode === 'visual'
                ? 'bg-blue-100 text-blue-700 border border-blue-200'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            <Palette className="w-4 h-4" />
            Visual Designer
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex h-[calc(100vh-140px)]">
        {/* Left Sidebar - Templates (only for YAML mode) */}
        {mode === 'yaml' && (
          <div className="w-80 bg-white border-r p-6 overflow-y-auto">
            <div className="mb-6">
              <label className="block text-sm font-medium mb-2">Workflow Name</label>
              <input
                type="text"
                value={workflowName}
                onChange={(e) => setWorkflowName(e.target.value)}
                placeholder="Enter workflow name"
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium mb-2">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe what this workflow does"
                rows={3}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="mb-6">
              <h3 className="font-medium mb-3">Templates</h3>
              <div className="space-y-2">
                {Object.entries(templates).map(([key, template]) => (
                  <button
                    key={key}
                    onClick={() => handleTemplateSelect(key)}
                    className="w-full text-left p-3 border rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors"
                  >
                    <div className="font-medium text-sm">{template.name}</div>
                    <div className="text-xs text-gray-500 mt-1">
                      {key === 'blank' && 'Start with a basic workflow structure'}
                      {key === 'email' && 'Process emails with Gmail and Slack'}
                      {key === 'data' && 'HTTP data processing pipeline'}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <button
                onClick={copyToClipboard}
                disabled={!yamlContent.trim()}
                className="w-full flex items-center gap-2 px-3 py-2 border rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                <Copy className="w-4 h-4" />
                Copy YAML
              </button>
              
              <button
                onClick={clearEditor}
                className="w-full flex items-center gap-2 px-3 py-2 border border-red-200 text-red-600 rounded-lg hover:bg-red-50"
              >
                <FileText className="w-4 h-4" />
                Clear Editor
              </button>
            </div>

            {/* Validation Results */}
            {validationResult && (
              <div className="mt-6 p-4 border rounded-lg">
                <h4 className="font-medium mb-2">Validation Results</h4>
                {validationResult.is_valid ? (
                  <div className="text-green-600 text-sm">✓ Workflow is valid</div>
                ) : (
                  <div className="space-y-2">
                    {validationResult.errors?.map((error: any, index: number) => (
                      <div key={index} className="text-red-600 text-sm">
                        • {error.message}
                      </div>
                    ))}
                    {validationResult.warnings?.map((warning: any, index: number) => (
                      <div key={index} className="text-yellow-600 text-sm">
                        ⚠ {warning.message}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Main Editor Area */}
        <div className="flex-1 flex flex-col">
          {mode === 'yaml' ? (
            <div className="flex-1">
              <Editor
                height="100%"
                defaultLanguage="yaml"
                value={yamlContent}
                onChange={handleYamlChange}
                onMount={handleEditorDidMount}
                theme="vs-light"
                options={{
                  fontSize: 14,
                  lineNumbers: 'on',
                  wordWrap: 'on',
                  minimap: { enabled: false },
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  tabSize: 2,
                  insertSpaces: true,
                  folding: true,
                  bracketPairColorization: { enabled: true },
                }}
              />
            </div>
          ) : (
            <VisualWorkflowDesigner
              initialYaml={yamlContent}
              onChange={handleVisualDesignerChange}
            />
          )}
        </div>
      </div>
    </div>
  )
}