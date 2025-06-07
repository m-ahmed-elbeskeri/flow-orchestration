import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Code, Plus, Save, Play, FileText, Copy, Download, Wand2, ArrowLeft, Settings, Eye, CheckCircle, AlertCircle } from 'lucide-react'
import { workflowApi } from '../api/client'
import { downloadFile } from '../utils/download'
import type { WorkflowCreateRequest } from '../types/workflow'

// Monaco Editor
import Editor from '@monaco-editor/react'

const WORKFLOW_TEMPLATES = {
  basic_email: {
    name: 'Basic Email Notification',
    description: 'Simple workflow that sends an email notification',
    category: 'Communication',
    yaml: `name: email_notification_workflow
version: "1.0.0"
description: "Send email notifications based on triggers"
author: "Workflow Designer"

config:
  timeout: 300
  max_concurrent: 5
  retry_policy:
    max_retries: 3
    initial_delay: 1.0
    exponential_base: 2.0

environment:
  variables:
    LOG_LEVEL: INFO
    EMAIL_SUBJECT: "Workflow Notification"
  secrets:
    - GMAIL_CREDENTIALS
    - SMTP_PASSWORD

states:
  - name: start
    type: builtin.start
    description: "Initialize the workflow"
    transitions:
      - on_success: validate_input

  - name: validate_input
    type: builtin.conditional
    description: "Validate input parameters"
    config:
      condition: "context.get_variable('recipient_email') is not None"
    dependencies:
      - name: start
        type: required
    transitions:
      - on_true: send_notification
      - on_false: handle_error

  - name: send_notification
    type: gmail.send_email
    description: "Send email notification"
    config:
      to: 
        - "{{ recipient_email }}"
      subject: "{{ EMAIL_SUBJECT }}"
      body: |
        Hello,
        
        This is an automated notification from your workflow system.
        
        Workflow: {{ workflow_name }}
        Timestamp: {{ current_timestamp }}
        
        Best regards,
        Workflow System
    resources:
      cpu_units: 0.1
      memory_mb: 50
      network_weight: 1.0
    dependencies:
      - name: validate_input
        type: required
    transitions:
      - on_success: log_success
      - on_failure: handle_error

  - name: log_success
    type: builtin.transform
    description: "Log successful completion"
    dependencies:
      - name: send_notification
        type: required
    transitions:
      - on_success: end

  - name: handle_error
    type: builtin.error_handler
    description: "Handle workflow errors"
    config:
      log_level: ERROR
      notify: true
      retry_count: 2
    transitions:
      - on_complete: end

  - name: end
    type: builtin.end
    description: "Workflow completion"

schedule:
  enabled: false
  cron: "0 9 * * MON-FRI"
  timezone: "UTC"
  max_instances: 1`
  },
  
  data_processing: {
    name: 'Data Processing Pipeline',
    description: 'ETL workflow for data processing and analysis',
    category: 'Data & Analytics',
    yaml: `name: data_processing_pipeline
version: "1.0.0"
description: "Extract, transform, and load data with notifications"
author: "Data Engineering Team"

config:
  timeout: 1800
  max_concurrent: 3
  retry_policy:
    max_retries: 5
    initial_delay: 2.0
    exponential_base: 2.0
    jitter: true

environment:
  variables:
    LOG_LEVEL: INFO
    DATA_SOURCE_URL: "https://api.example.com/data"
    BATCH_SIZE: "1000"
    OUTPUT_FORMAT: "parquet"
  secrets:
    - DATABASE_CREDENTIALS
    - API_KEY
    - SLACK_WEBHOOK_URL

states:
  - name: start
    type: builtin.start
    description: "Initialize data pipeline"
    transitions:
      - on_success: fetch_data

  - name: fetch_data
    type: webhook.http_request
    description: "Fetch data from external API"
    config:
      url: "{{ DATA_SOURCE_URL }}"
      method: "GET"
      headers:
        Authorization: "Bearer {{ API_KEY }}"
        Content-Type: "application/json"
      timeout: 120
    resources:
      cpu_units: 0.5
      memory_mb: 200
      network_weight: 2.0
    dependencies:
      - name: start
        type: required
    transitions:
      - on_success: validate_data
      - on_failure: notify_failure

  - name: validate_data
    type: builtin.transform
    description: "Validate and clean incoming data"
    resources:
      cpu_units: 1.0
      memory_mb: 512
    dependencies:
      - name: fetch_data
        type: required
    transitions:
      - on_success: transform_data
      - on_failure: notify_failure

  - name: transform_data
    type: builtin.transform
    description: "Apply business logic transformations"
    resources:
      cpu_units: 2.0
      memory_mb: 1024
    dependencies:
      - name: validate_data
        type: required
    transitions:
      - on_success: store_results
      - on_failure: notify_failure

  - name: store_results
    type: builtin.transform
    description: "Store processed data"
    resources:
      cpu_units: 0.5
      memory_mb: 256
      network_weight: 1.0
    dependencies:
      - name: transform_data
        type: required
    transitions:
      - on_success: notify_completion
      - on_failure: notify_failure

  - name: notify_completion
    type: slack.send_message
    description: "Notify successful completion"
    config:
      channel: "#data-pipeline"
      message: "✅ Data Pipeline Completed Successfully"
    dependencies:
      - name: store_results
        type: required
    transitions:
      - on_success: end

  - name: notify_failure
    type: slack.send_message
    description: "Notify pipeline failure"
    config:
      channel: "#data-pipeline-alerts"
      message: "❌ Data Pipeline Failed"
    transitions:
      - on_complete: end

  - name: end
    type: builtin.end
    description: "Pipeline completion"

schedule:
  enabled: true
  cron: "0 */6 * * *"
  timezone: "UTC"
  max_instances: 1`
  },

  api_monitoring: {
    name: 'API Health Monitoring',
    description: 'Monitor API endpoints and send alerts',
    category: 'Monitoring & Alerts',
    yaml: `name: api_health_monitor
version: "1.0.0"
description: "Monitor API health and send alerts on failures"
author: "DevOps Team"

config:
  timeout: 300
  max_concurrent: 10
  retry_policy:
    max_retries: 3
    initial_delay: 5.0

environment:
  variables:
    LOG_LEVEL: INFO
    HEALTH_CHECK_INTERVAL: "300"
    ALERT_THRESHOLD: "3"
  secrets:
    - MONITORING_API_KEY
    - PAGERDUTY_TOKEN
    - SLACK_WEBHOOK

states:
  - name: start
    type: builtin.start
    transitions:
      - on_success: check_api_health

  - name: check_api_health
    type: webhook.http_request
    description: "Check API health status"
    config:
      url: "https://api.example.com/health"
      method: "GET"
      timeout: 30
      expected_status: 200
    dependencies:
      - name: start
        type: required
    transitions:
      - on_success: log_healthy_status
      - on_failure: handle_api_failure

  - name: log_healthy_status
    type: builtin.transform
    description: "Log healthy API status"
    dependencies:
      - name: check_api_health
        type: required
    transitions:
      - on_success: end

  - name: handle_api_failure
    type: builtin.conditional
    description: "Check if this is a recurring failure"
    config:
      condition: "context.get_variable('consecutive_failures', 0) >= int(context.get_variable('ALERT_THRESHOLD', 3))"
    dependencies:
      - name: check_api_health
        type: required
    transitions:
      - on_true: send_critical_alert
      - on_false: log_failure

  - name: log_failure
    type: builtin.transform
    description: "Log API failure"
    dependencies:
      - name: handle_api_failure
        type: required
    transitions:
      - on_success: end

  - name: send_critical_alert
    type: slack.send_message
    description: "Send critical alert for API failure"
    config:
      channel: "#critical-alerts"
      message: "🚨 **CRITICAL: API Health Check Failed**"
    dependencies:
      - name: handle_api_failure
        type: required
    transitions:
      - on_success: create_incident
      - on_failure: end

  - name: create_incident
    type: webhook.http_request
    description: "Create PagerDuty incident"
    config:
      url: "https://events.pagerduty.com/v2/enqueue"
      method: "POST"
    dependencies:
      - name: send_critical_alert
        type: required
    transitions:
      - on_success: end
      - on_failure: end

  - name: end
    type: builtin.end

schedule:
  enabled: true
  cron: "*/5 * * * *"
  timezone: "UTC"
  max_instances: 1`
  },

  blank: {
    name: 'Blank Workflow',
    description: 'Start with a minimal workflow template',
    category: 'Getting Started',
    yaml: `name: my_workflow
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
  }
}

export default function WorkflowCreator() {
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    yamlContent: ''
  })
  const [selectedTemplate, setSelectedTemplate] = useState<string>('')
  const [validationResult, setValidationResult] = useState<any>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isGeneratingCode, setIsGeneratingCode] = useState(false)
  const [activeTab, setActiveTab] = useState<'editor' | 'preview'>('editor')
  const editorRef = useRef<any>(null)

  const handleEditorDidMount = (editor: any, monaco: any) => {
    editorRef.current = editor

    // Configure YAML language support
    monaco.languages.setMonarchTokensProvider('yaml', {
      tokenizer: {
        root: [
          [/^\s*-/, 'delimiter'],
          [/^\s*[\w-]+:/, 'key'],
          [/#.*$/, 'comment'],
          [/".*?"/, 'string'],
          [/'.*?'/, 'string'],
          [/\btrue\b|\bfalse\b/, 'boolean'],
          [/\d+/, 'number'],
        ]
      }
    })

    // Set editor theme
    monaco.editor.defineTheme('workflow-theme', {
      base: 'vs',
      inherit: true,
      rules: [
        { token: 'key', foreground: '0066cc', fontStyle: 'bold' },
        { token: 'string', foreground: '009900' },
        { token: 'comment', foreground: '999999', fontStyle: 'italic' },
        { token: 'delimiter', foreground: 'ff6600' },
        { token: 'boolean', foreground: '9900cc' },
        { token: 'number', foreground: 'cc6600' },
      ],
      colors: {
        'editor.background': '#fafafa'
      }
    })
    monaco.editor.setTheme('workflow-theme')
  }

  const handleTemplateSelect = (templateKey: string) => {
    const template = WORKFLOW_TEMPLATES[templateKey as keyof typeof WORKFLOW_TEMPLATES]
    if (template) {
      setSelectedTemplate(templateKey)
      setFormData({
        name: template.name,
        description: template.description,
        yamlContent: template.yaml
      })
      // Reset validation when template changes
      setValidationResult(null)
    }
  }

  const validateWorkflow = async () => {
    if (!formData.yamlContent.trim()) {
      setValidationResult({ is_valid: false, errors: [{ message: 'YAML content is required' }] })
      return
    }

    setIsValidating(true)
    setValidationResult(null)

    try {
      console.log('🔍 Validating YAML...')
      const result = await workflowApi.validateWorkflow(formData.yamlContent)
      console.log('✅ Validation result:', result)
      setValidationResult(result)
    } catch (error: any) {
      console.error('❌ Validation error:', error)
      setValidationResult({
        is_valid: false,
        errors: [{ message: error.message || 'Validation failed' }]
      })
    } finally {
      setIsValidating(false)
    }
  }

  const saveWorkflow = async () => {
    if (!formData.name.trim()) {
      alert('Please enter a workflow name')
      return
    }
    
    if (!formData.yamlContent.trim()) {
      alert('Please enter YAML content')
      return
    }

    if (!validationResult || !validationResult.is_valid) {
      alert('Please validate the workflow first and fix any errors')
      return
    }

    setIsSaving(true)
    try {
      console.log('💾 Saving workflow...')
      const workflow = {
        name: formData.name,
        yaml_content: formData.yamlContent,
        auto_start: false
      }

      const result = await workflowApi.createWorkflowFromYaml(workflow)
      console.log('✅ Workflow saved:', result)
      
      alert(`Workflow "${formData.name}" saved successfully!`)
      navigate('/workflows')
    } catch (error: any) {
      console.error('❌ Failed to save workflow:', error)
      alert(`Failed to save workflow: ${error.message || 'Unknown error'}`)
    } finally {
      setIsSaving(false)
    }
  }

  const handleGenerateCode = async () => {
    if (!formData.yamlContent.trim()) {
      alert('Please enter YAML content first')
      return
    }

    setIsGeneratingCode(true)
    try {
      console.log('🔧 Generating code from YAML...')
      const result = await workflowApi.generateCodeFromYaml(formData.yamlContent)
      console.log('✅ Code generated:', result)
      
      if (result && result.success) {
        if (result.zip_content) {
          const workflowName = formData.name || 'workflow'
          const base64Data = result.zip_content
          const byteCharacters = atob(base64Data)
          const byteNumbers = new Array(byteCharacters.length)
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i)
          }
          const byteArray = new Uint8Array(byteNumbers)
          const blob = new Blob([byteArray], { type: 'application/zip' })
          
          const url = window.URL.createObjectURL(blob)
          const link = document.createElement('a')
          link.href = url
          link.download = `${workflowName}-generated.zip`
          document.body.appendChild(link)
          link.click()
          document.body.removeChild(link)
          window.URL.revokeObjectURL(url)
          
          console.log('✅ Code downloaded successfully')
        }
      } else {
        console.error('❌ Code generation failed:', result?.message)
        alert(`Code generation failed: ${result?.message || 'Unknown error'}`)
      }
    } catch (error: any) {
      console.error('❌ Code generation failed:', error)
      alert(`Code generation failed: ${error.message || 'Unknown error'}`)
    } finally {
      setIsGeneratingCode(false)
    }
  }

  const copyToClipboard = () => {
    navigator.clipboard.writeText(formData.yamlContent)
    alert('YAML content copied to clipboard!')
  }

  const clearEditor = () => {
    setFormData({ name: '', description: '', yamlContent: '' })
    setSelectedTemplate('')
    setValidationResult(null)
  }

  // Group templates by category
  const templatesByCategory = Object.entries(WORKFLOW_TEMPLATES).reduce((acc, [key, template]) => {
    const category = template.category || 'Other'
    if (!acc[category]) acc[category] = []
    acc[category].push({ key, ...template })
    return acc
  }, {} as Record<string, any[]>)

  // Validation status helper
  const getValidationStatus = () => {
    if (!validationResult) return null
    return validationResult.is_valid ? 'valid' : 'invalid'
  }

  const isValidated = validationResult && validationResult.is_valid

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header - Fixed height */}
      <div className="bg-white border-b shadow-sm flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <button
                onClick={() => navigate('/workflows')}
                className="mr-4 p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                  <Plus className="w-8 h-8 mr-3 text-blue-600" />
                  Create New Workflow
                </h1>
                <p className="text-gray-600 mt-1">Design and configure your automation workflow</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={copyToClipboard}
                disabled={!formData.yamlContent.trim()}
                className="px-4 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                <Copy className="w-4 h-4 mr-2" />
                Copy YAML
              </button>
              
              <button
                onClick={handleGenerateCode}
                disabled={isGeneratingCode || !formData.yamlContent.trim()}
                className="px-4 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
              >
                <Code className="w-4 h-4 mr-2" />
                {isGeneratingCode ? 'Generating...' : 'Generate Code'}
              </button>
              
              <button
                onClick={validateWorkflow}
                disabled={isValidating || !formData.yamlContent.trim()}
                className="px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
              >
                <CheckCircle className="w-4 h-4 mr-2" />
                {isValidating ? 'Validating...' : 'Validate Workflow'}
              </button>
              
              <button
                onClick={saveWorkflow}
                disabled={isSaving || !isValidated || !formData.name.trim() || !formData.yamlContent.trim()}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
              >
                <Save className="w-4 h-4 mr-2" />
                {isSaving ? 'Saving...' : 'Save Workflow'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - Flex grow with proper constraints */}
      <div className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 min-h-0">
        <div className="grid grid-cols-12 gap-8 h-full">
          {/* Left Sidebar - Fixed width, scrollable */}
          <div className="col-span-3">
            <div className="bg-white rounded-lg shadow-sm border h-full overflow-hidden flex flex-col">
              <div className="p-6 border-b flex-shrink-0">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                  <Settings className="w-5 h-5 mr-2 text-gray-600" />
                  Configuration
                </h2>
              </div>
              
              <div className="flex-1 overflow-y-auto p-6">
                {/* Workflow Info */}
                <div className="mb-8">
                  <h3 className="text-sm font-semibold text-gray-900 mb-4">Basic Information</h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Workflow Name *
                      </label>
                      <input
                        type="text"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        placeholder="Enter workflow name..."
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Description
                      </label>
                      <textarea
                        value={formData.description}
                        onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                        placeholder="Describe what this workflow does..."
                        rows={3}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                      />
                    </div>
                  </div>
                </div>

                {/* Validation Status */}
                {validationResult && (
                  <div className="mb-8">
                    <h3 className="text-sm font-semibold text-gray-900 mb-4">Validation Status</h3>
                    <div className={`p-3 rounded-lg border ${
                      validationResult.is_valid 
                        ? 'bg-green-50 border-green-200' 
                        : 'bg-red-50 border-red-200'
                    }`}>
                      <div className="flex items-center">
                        {validationResult.is_valid ? (
                          <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-red-600 mr-2" />
                        )}
                        <span className={`text-sm font-medium ${
                          validationResult.is_valid ? 'text-green-800' : 'text-red-800'
                        }`}>
                          {validationResult.is_valid ? 'Valid Configuration' : 'Invalid Configuration'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Templates */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 mb-4 flex items-center">
                    <Wand2 className="w-4 h-4 mr-2 text-purple-600" />
                    Templates
                  </h3>
                  
                  <div className="space-y-6">
                    {Object.entries(templatesByCategory).map(([category, templates]) => (
                      <div key={category}>
                        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
                          {category}
                        </h4>
                        <div className="space-y-2">
                          {templates.map((template) => (
                            <button
                              key={template.key}
                              onClick={() => handleTemplateSelect(template.key)}
                              className={`w-full text-left p-3 rounded-lg border transition-all ${
                                selectedTemplate === template.key
                                  ? 'border-blue-500 bg-blue-50 shadow-sm'
                                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                              }`}
                            >
                              <div className="font-medium text-gray-900 text-sm">{template.name}</div>
                              <div className="text-xs text-gray-600 mt-1 line-clamp-2">{template.description}</div>
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Quick Actions */}
                <div className="mt-8 pt-6 border-t space-y-3">
                  <button
                    onClick={clearEditor}
                    className="w-full px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                  >
                    Clear All
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Center - Editor - Fixed height with overflow */}
          <div className="col-span-6">
            <div className="bg-white rounded-lg shadow-sm border h-full overflow-hidden flex flex-col">
              <div className="flex justify-between items-center px-6 py-4 border-b flex-shrink-0">
                <div className="flex space-x-1">
                  <button
                    onClick={() => setActiveTab('editor')}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      activeTab === 'editor'
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
                    }`}
                  >
                    <FileText className="w-4 h-4 inline mr-2" />
                    YAML Editor
                  </button>
                  <button
                    onClick={() => setActiveTab('preview')}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      activeTab === 'preview'
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
                    }`}
                  >
                    <Eye className="w-4 h-4 inline mr-2" />
                    Preview
                  </button>
                </div>
              </div>
              
              <div className="flex-1 min-h-0">
                {activeTab === 'editor' ? (
                  <Editor
                    height="100%"
                    defaultLanguage="yaml"
                    value={formData.yamlContent}
                    onChange={(value) => {
                      setFormData({ ...formData, yamlContent: value || '' })
                      // Reset validation when content changes
                      setValidationResult(null)
                    }}
                    onMount={handleEditorDidMount}
                    options={{
                      minimap: { enabled: true },
                      lineNumbers: 'on',
                      roundedSelection: false,
                      scrollBeyondLastLine: false,
                      automaticLayout: true,
                      fontSize: 14,
                      tabSize: 2,
                      insertSpaces: true,
                      wordWrap: 'on',
                      folding: true,
                      showFoldingControls: 'always',
                      bracketPairColorization: { enabled: true },
                    }}
                    theme="workflow-theme"
                  />
                ) : (
                  <div className="h-full p-6 overflow-y-auto bg-gray-50">
                    <div className="prose max-w-none">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">Workflow Preview</h3>
                      {formData.yamlContent ? (
                        <pre className="bg-white p-4 rounded-lg border text-sm overflow-x-auto whitespace-pre-wrap">
                          {formData.yamlContent}
                        </pre>
                      ) : (
                        <div className="text-center py-12 text-gray-500">
                          <FileText className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                          <p>Select a template or start writing YAML to see the preview</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Panel - Validation - Fixed height with overflow */}
          <div className="col-span-3">
            <div className="bg-white rounded-lg shadow-sm border h-full overflow-hidden flex flex-col">
              <div className="p-6 border-b flex-shrink-0">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                  <Play className="w-5 h-5 mr-2 text-green-600" />
                  Validation
                </h2>
              </div>
              
              <div className="flex-1 overflow-y-auto p-6">
                {!validationResult && (
                  <div className="text-center py-12 text-gray-500">
                    <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                      <Play className="w-8 h-8 text-gray-400" />
                    </div>
                    <p className="text-sm">Click "Validate Workflow" to check your configuration</p>
                  </div>
                )}

                {validationResult && (
                  <div className="space-y-6">
                    {/* Errors */}
                    {validationResult.errors && validationResult.errors.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-red-800 mb-3 flex items-center">
                          <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                          Errors ({validationResult.errors.length})
                        </h4>
                        <div className="space-y-2">
                          {validationResult.errors.map((error: any, index: number) => (
                            <div key={index} className="text-red-700 text-sm bg-red-50 p-3 rounded-lg border border-red-200">
                              {error.message || error}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Warnings */}
                    {validationResult.warnings && validationResult.warnings.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-yellow-800 mb-3 flex items-center">
                          <span className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
                          Warnings ({validationResult.warnings.length})
                        </h4>
                        <div className="space-y-2">
                          {validationResult.warnings.map((warning: any, index: number) => (
                            <div key={index} className="text-yellow-700 text-sm bg-yellow-50 p-3 rounded-lg border border-yellow-200">
                              {warning.message || warning}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Info */}
                    {validationResult.info && Object.keys(validationResult.info).length > 0 && (
                      <div>
                        <h4 className="font-semibold text-blue-800 mb-3 flex items-center">
                          <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                          Workflow Info
                        </h4>
                        <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                          <pre className="text-blue-700 text-xs overflow-auto whitespace-pre-wrap">
                            {JSON.stringify(validationResult.info, null, 2)}
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}