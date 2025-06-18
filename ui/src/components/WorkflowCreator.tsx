import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Code, Plus, Save, Play, FileText, Copy, Download, Wand2, ArrowLeft, Settings, Eye, CheckCircle, AlertCircle, Workflow, Activity, Zap } from 'lucide-react'
import { workflowApi } from '../api/client'
import { downloadFile } from '../utils/download'
import type { WorkflowCreateRequest } from '../types/workflow'

// Monaco Editor
import Editor from '@monaco-editor/react'

// Import the Visual Workflow Designer
import VisualWorkflowDesigner from './VisualWorkflowDesigner'

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
  const [activeTab, setActiveTab] = useState<'visual' | 'editor' | 'preview'>('visual')
  const editorRef = useRef<any>(null)

  const handleEditorDidMount = (editor: any, monaco: any) => {
    editorRef.current = editor

    // Configure YAML language support with dark theme
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

    // Set dark theme
    monaco.editor.defineTheme('workflow-dark-theme', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'key', foreground: '00d4ff', fontStyle: 'bold' },
        { token: 'string', foreground: '10b981' },
        { token: 'comment', foreground: '6b7280', fontStyle: 'italic' },
        { token: 'delimiter', foreground: 'f59e0b' },
        { token: 'boolean', foreground: '8b5cf6' },
        { token: 'number', foreground: 'f97316' },
      ],
      colors: {
        'editor.background': '#1f2937',
        'editor.foreground': '#f9fafb',
        'editorLineNumber.foreground': '#6b7280',
        'editor.selectionBackground': '#374151',
        'editor.lineHighlightBackground': '#374151',
      }
    })
    monaco.editor.setTheme('workflow-dark-theme')
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
      setValidationResult(null)
    }
  }

  const handleVisualDesignerChange = (yaml: string) => {
    setFormData({ ...formData, yamlContent: yaml })
    setValidationResult(null)
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

  const getValidationStatus = () => {
    if (!validationResult) return null
    return validationResult.is_valid ? 'valid' : 'invalid'
  }

  const isValidated = validationResult && validationResult.is_valid

  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex flex-col overflow-hidden">
      {/* Enhanced Header */}
      <div className="bg-gray-800/95 border-b border-gray-700 shadow-2xl flex-shrink-0 backdrop-blur-sm">
        <div className="max-w-full mx-auto px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <button
                onClick={() => navigate('/workflows')}
                className="mr-4 p-2.5 text-gray-400 hover:text-blue-400 hover:bg-gray-700/50 rounded-xl transition-all duration-200"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <div className="flex items-center">
                <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl mr-4 shadow-lg">
                  <Plus className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white flex items-center">
                    Create New Workflow
                    <div className="ml-3 flex items-center space-x-2 bg-blue-500/20 px-3 py-1 rounded-full border border-blue-400/30">
                      <Activity className="w-3 h-3 text-blue-400 animate-pulse" />
                      <span className="text-xs text-blue-300 font-medium">AI Powered</span>
                    </div>
                  </h1>
                  <p className="text-gray-400 mt-1">Design and configure your automation workflow with AI assistance</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={copyToClipboard}
                disabled={!formData.yamlContent.trim()}
                className="px-4 py-2 text-gray-300 hover:text-white hover:bg-gray-700/50 rounded-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center backdrop-blur-sm"
              >
                <Copy className="w-4 h-4 mr-2" />
                Copy YAML
              </button>
              
              <button
                onClick={handleGenerateCode}
                disabled={isGeneratingCode || !formData.yamlContent.trim()}
                className="px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center shadow-lg"
              >
                <Code className="w-4 h-4 mr-2" />
                {isGeneratingCode ? (
                  <>
                    <Activity className="w-4 h-4 mr-2 animate-spin" />
                    Generating...
                  </>
                ) : (
                  'Generate Code'
                )}
              </button>
              
              <button
                onClick={validateWorkflow}
                disabled={isValidating || !formData.yamlContent.trim()}
                className="px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center shadow-lg"
              >
                {isValidating ? (
                  <>
                    <Activity className="w-4 h-4 mr-2 animate-spin" />
                    Validating...
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Validate
                  </>
                )}
              </button>
              
              <button
                onClick={saveWorkflow}
                disabled={isSaving || !isValidated || !formData.name.trim() || !formData.yamlContent.trim()}
                className="px-6 py-2 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center shadow-lg"
              >
                {isSaving ? (
                  <>
                    <Activity className="w-4 h-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    Save Workflow
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - Full height layout */}
      <div className="flex-1 flex min-h-0">
        {/* Left Sidebar - Enhanced Dark Theme */}
        <div className="w-80 bg-gray-900/95 border-r border-gray-700 shadow-2xl overflow-hidden flex flex-col backdrop-blur-sm">
          <div className="p-6 border-b border-gray-700 flex-shrink-0 bg-gradient-to-r from-gray-800/50 to-gray-900/50">
            <h2 className="text-lg font-semibold text-white flex items-center">
              <Settings className="w-5 h-5 mr-3 text-blue-400" />
              Configuration
            </h2>
          </div>
          
          <div className="flex-1 overflow-y-auto p-6">
            {/* Workflow Info */}
            <div className="mb-8">
              <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center">
                <div className="w-1 h-4 bg-gradient-to-b from-blue-400 to-purple-500 rounded-full mr-3"></div>
                Basic Information
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Workflow Name *
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="Enter workflow name..."
                    className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-700/50 text-white backdrop-blur-sm"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Description
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder="Describe what this workflow does..."
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-gray-700/50 text-white backdrop-blur-sm"
                  />
                </div>
              </div>
            </div>

            {/* Validation Status */}
            {validationResult && (
              <div className="mb-8">
                <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center">
                  <div className="w-1 h-4 bg-gradient-to-b from-green-400 to-blue-500 rounded-full mr-3"></div>
                  Validation Status
                </h3>
                <div className={`p-4 rounded-xl border backdrop-blur-sm ${
                  validationResult.is_valid 
                    ? 'bg-green-500/20 border-green-400/30' 
                    : 'bg-red-500/20 border-red-400/30'
                }`}>
                  <div className="flex items-center">
                    {validationResult.is_valid ? (
                      <CheckCircle className="w-5 h-5 text-green-400 mr-3" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-red-400 mr-3" />
                    )}
                    <span className={`text-sm font-medium ${
                      validationResult.is_valid ? 'text-green-300' : 'text-red-300'
                    }`}>
                      {validationResult.is_valid ? 'Valid Configuration' : 'Invalid Configuration'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Templates */}
            <div>
              <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center">
                <div className="w-1 h-4 bg-gradient-to-b from-purple-400 to-pink-500 rounded-full mr-3"></div>
                <Wand2 className="w-4 h-4 mr-2 text-purple-400" />
                Templates
              </h3>
              
              <div className="space-y-6">
                {Object.entries(templatesByCategory).map(([category, templates]) => (
                  <div key={category}>
                    <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                      {category}
                    </h4>
                    <div className="space-y-2">
                      {templates.map((template) => (
                        <button
                          key={template.key}
                          onClick={() => handleTemplateSelect(template.key)}
                          className={`w-full text-left p-4 rounded-xl border transition-all duration-200 backdrop-blur-sm ${
                            selectedTemplate === template.key
                              ? 'border-blue-500/50 bg-blue-500/20 shadow-lg shadow-blue-500/20'
                              : 'border-gray-600/50 hover:border-gray-500 hover:bg-gray-700/30'
                          }`}
                        >
                          <div className="font-medium text-white text-sm">{template.name}</div>
                          <div className="text-xs text-gray-400 mt-1 line-clamp-2">{template.description}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="mt-8 pt-6 border-t border-gray-700 space-y-3">
              <button
                onClick={clearEditor}
                className="w-full px-4 py-2 bg-gray-700/50 hover:bg-gray-600/50 text-gray-300 rounded-xl transition-all duration-200 backdrop-blur-sm"
              >
                Clear All
              </button>
            </div>
          </div>
        </div>

        {/* Center - Designer/Editor */}
        <div className="flex-1 bg-gray-800/50 overflow-hidden flex flex-col backdrop-blur-sm">
          <div className="flex justify-between items-center px-6 py-4 border-b border-gray-700 flex-shrink-0 bg-gray-800/50">
            <div className="flex space-x-1">
              <button
                onClick={() => setActiveTab('visual')}
                className={`px-4 py-2 text-sm font-medium rounded-xl transition-all duration-200 ${
                  activeTab === 'visual'
                    ? 'bg-blue-500/20 text-blue-300 border border-blue-400/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'
                }`}
              >
                <Workflow className="w-4 h-4 inline mr-2" />
                Visual Designer
              </button>
              <button
                onClick={() => setActiveTab('editor')}
                className={`px-4 py-2 text-sm font-medium rounded-xl transition-all duration-200 ${
                  activeTab === 'editor'
                    ? 'bg-blue-500/20 text-blue-300 border border-blue-400/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'
                }`}
              >
                <FileText className="w-4 h-4 inline mr-2" />
                YAML Editor
              </button>
              <button
                onClick={() => setActiveTab('preview')}
                className={`px-4 py-2 text-sm font-medium rounded-xl transition-all duration-200 ${
                  activeTab === 'preview'
                    ? 'bg-blue-500/20 text-blue-300 border border-blue-400/30'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'
                }`}
              >
                <Eye className="w-4 h-4 inline mr-2" />
                Preview
              </button>
            </div>
          </div>
          
          <div className="flex-1 min-h-0">
            {activeTab === 'visual' ? (
              <div className="h-full">
                <VisualWorkflowDesigner
                  initialYaml={formData.yamlContent}
                  onChange={handleVisualDesignerChange}
                  onSave={handleVisualDesignerChange}
                />
              </div>
            ) : activeTab === 'editor' ? (
              <Editor
                height="100%"
                defaultLanguage="yaml"
                value={formData.yamlContent}
                onChange={(value) => {
                  setFormData({ ...formData, yamlContent: value || '' })
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
                theme="workflow-dark-theme"
              />
            ) : (
              <div className="h-full p-6 overflow-y-auto bg-gray-800/30">
                <div className="max-w-none">
                  <h3 className="text-lg font-semibold text-white mb-4">Workflow Preview</h3>
                  {formData.yamlContent ? (
                    <pre className="bg-gray-800/80 p-6 rounded-xl border border-gray-600 text-sm overflow-x-auto whitespace-pre-wrap text-gray-200 backdrop-blur-sm">
                      {formData.yamlContent}
                    </pre>
                  ) : (
                    <div className="text-center py-16 text-gray-400">
                      <div className="w-20 h-20 mx-auto mb-6 bg-gray-700/50 rounded-full flex items-center justify-center backdrop-blur-sm">
                        <FileText className="w-10 h-10 text-gray-500" />
                      </div>
                      <p className="text-lg">Select a template or start designing your workflow to see the preview</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Validation */}
        <div className="w-80 bg-gray-900/95 border-l border-gray-700 shadow-2xl overflow-hidden flex flex-col backdrop-blur-sm">
          <div className="p-6 border-b border-gray-700 flex-shrink-0 bg-gradient-to-r from-gray-800/50 to-gray-900/50">
            <h2 className="text-lg font-semibold text-white flex items-center">
              <Play className="w-5 h-5 mr-3 text-green-400" />
              Validation
            </h2>
          </div>
          
          <div className="flex-1 overflow-y-auto p-6">
            {!validationResult && (
              <div className="text-center py-16 text-gray-400">
                <div className="w-20 h-20 mx-auto mb-6 bg-gray-700/50 rounded-full flex items-center justify-center backdrop-blur-sm">
                  <Play className="w-10 h-10 text-gray-500" />
                </div>
                <p className="text-sm">Click "Validate Workflow" to check your configuration</p>
              </div>
            )}

            {validationResult && (
              <div className="space-y-6">
                {/* Errors */}
                {validationResult.errors && validationResult.errors.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-red-300 mb-3 flex items-center">
                      <span className="w-2 h-2 bg-red-500 rounded-full mr-3"></span>
                      Errors ({validationResult.errors.length})
                    </h4>
                    <div className="space-y-2">
                      {validationResult.errors.map((error: any, index: number) => (
                        <div key={index} className="text-red-300 text-sm bg-red-500/20 p-4 rounded-xl border border-red-400/30 backdrop-blur-sm">
                          {error.message || error}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Warnings */}
                {validationResult.warnings && validationResult.warnings.length > 0 && (
                  <div>
                    <h4 className="font-semibold text-yellow-300 mb-3 flex items-center">
                      <span className="w-2 h-2 bg-yellow-500 rounded-full mr-3"></span>
                      Warnings ({validationResult.warnings.length})
                    </h4>
                    <div className="space-y-2">
                      {validationResult.warnings.map((warning: any, index: number) => (
                        <div key={index} className="text-yellow-300 text-sm bg-yellow-500/20 p-4 rounded-xl border border-yellow-400/30 backdrop-blur-sm">
                          {warning.message || warning}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Info */}
                {validationResult.info && Object.keys(validationResult.info).length > 0 && (
                  <div>
                    <h4 className="font-semibold text-blue-300 mb-3 flex items-center">
                      <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
                      Workflow Info
                    </h4>
                    <div className="bg-blue-500/20 p-4 rounded-xl border border-blue-400/30 backdrop-blur-sm">
                      <pre className="text-blue-300 text-xs overflow-auto whitespace-pre-wrap">
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
  )
}