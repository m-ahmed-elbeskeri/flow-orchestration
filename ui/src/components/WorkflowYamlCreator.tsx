import { useState, useRef } from 'react'
import { Code, Plus, Save, Play, FileText, Copy, Download, Wand2 } from 'lucide-react'
import { workflowApi } from '../api/client'
import { downloadFile } from '../utils/download'
import type { WorkflowCreateRequest } from '../types/workflow'

// Monaco Editor (install with: npm install @monaco-editor/react)
import Editor from '@monaco-editor/react'

interface Props {
  onSuccess?: () => void
  onCancel?: () => void
}

const WORKFLOW_TEMPLATES = {
  basic_email: {
    name: 'Basic Email Notification',
    description: 'Simple workflow that sends an email notification',
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
    config:
      function: |
        async def process(context):
            import datetime
            timestamp = datetime.datetime.now().isoformat()
            context.set_output("completion_time", timestamp)
            context.set_output("status", "success")
            return "completed"
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
    config:
      function: |
        async def process(context):
            import json
            
            # Get the fetched data
            response = context.get_state("http_response")
            data = response.get("body", {})
            
            # Validation logic
            if not isinstance(data, dict) or "records" not in data:
                raise ValueError("Invalid data format")
            
            records = data["records"]
            valid_records = []
            
            for record in records:
                if all(key in record for key in ["id", "timestamp", "value"]):
                    # Clean and normalize the record
                    cleaned_record = {
                        "id": str(record["id"]),
                        "timestamp": record["timestamp"],
                        "value": float(record["value"]),
                        "processed_at": context.get_variable("current_timestamp")
                    }
                    valid_records.append(cleaned_record)
            
            context.set_state("validated_data", valid_records)
            context.set_state("validation_summary", {
                "total_records": len(records),
                "valid_records": len(valid_records),
                "invalid_records": len(records) - len(valid_records)
            })
            
            return len(valid_records)
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
    config:
      function: |
        async def process(context):
            import statistics
            
            validated_data = context.get_state("validated_data", [])
            batch_size = int(context.get_variable("BATCH_SIZE", 1000))
            
            # Process in batches
            processed_batches = []
            for i in range(0, len(validated_data), batch_size):
                batch = validated_data[i:i + batch_size]
                
                # Calculate batch statistics
                values = [record["value"] for record in batch]
                batch_stats = {
                    "count": len(batch),
                    "mean": statistics.mean(values) if values else 0,
                    "median": statistics.median(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }
                
                # Transform records
                transformed_batch = []
                for record in batch:
                    transformed_record = {
                        **record,
                        "normalized_value": (record["value"] - batch_stats["mean"]) / (batch_stats["max"] - batch_stats["min"]) if batch_stats["max"] != batch_stats["min"] else 0,
                        "batch_id": f"batch_{i // batch_size + 1}",
                        "batch_stats": batch_stats
                    }
                    transformed_batch.append(transformed_record)
                
                processed_batches.append(transformed_batch)
            
            # Flatten all batches
            final_data = [record for batch in processed_batches for record in batch]
            
            context.set_state("transformed_data", final_data)
            context.set_output("processing_summary", {
                "total_batches": len(processed_batches),
                "total_records_processed": len(final_data),
                "output_format": context.get_variable("OUTPUT_FORMAT")
            })
            
            return len(final_data)
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
    config:
      function: |
        async def process(context):
            import json
            import datetime
            
            transformed_data = context.get_state("transformed_data", [])
            output_format = context.get_variable("OUTPUT_FORMAT", "json")
            
            # Simulate storing data (replace with actual storage logic)
            storage_result = {
                "stored_at": datetime.datetime.now().isoformat(),
                "record_count": len(transformed_data),
                "format": output_format,
                "location": f"s3://data-bucket/processed/{datetime.date.today()}/data.{output_format}",
                "size_mb": len(json.dumps(transformed_data)) / (1024 * 1024)
            }
            
            context.set_output("storage_result", storage_result)
            return "stored"
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
      message: |
        ✅ Data Pipeline Completed Successfully
        
        📊 **Processing Summary:**
        • Records processed: {{ processing_summary.total_records_processed }}
        • Batches: {{ processing_summary.total_batches }}
        • Format: {{ processing_summary.output_format }}
        
        💾 **Storage Details:**
        • Location: {{ storage_result.location }}
        • Size: {{ storage_result.size_mb | round(2) }} MB
        • Stored at: {{ storage_result.stored_at }}
        
        🕐 **Execution Time:** {{ execution_duration }} seconds
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
      message: |
        ❌ Data Pipeline Failed
        
        **Error:** {{ last_error }}
        **Failed State:** {{ failed_state_name }}
        **Timestamp:** {{ current_timestamp }}
        
        Please check the logs for more details.
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
    config:
      function: |
        async def process(context):
            import datetime
            
            health_check = context.get_state("http_response")
            status_log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "healthy",
                "response_time": health_check.get("response_time", 0),
                "status_code": health_check.get("status_code", 200)
            }
            
            context.set_output("health_status", status_log)
            return "healthy"
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
    config:
      function: |
        async def process(context):
            import datetime
            
            current_failures = context.get_variable("consecutive_failures", 0) + 1
            context.set_variable("consecutive_failures", current_failures)
            
            failure_log = {
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "failed",
                "consecutive_failures": current_failures,
                "error": context.get_state("last_error")
            }
            
            context.set_output("failure_status", failure_log)
            return current_failures
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
      message: |
        🚨 **CRITICAL: API Health Check Failed**
        
        **API Endpoint:** https://api.example.com/health
        **Consecutive Failures:** {{ consecutive_failures }}
        **Alert Threshold:** {{ ALERT_THRESHOLD }}
        **Last Error:** {{ last_error }}
        **Timestamp:** {{ current_timestamp }}
        
        @channel Please investigate immediately!
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
      headers:
        Authorization: "Token token={{ PAGERDUTY_TOKEN }}"
        Content-Type: "application/json"
      body: |
        {
          "routing_key": "{{ PAGERDUTY_ROUTING_KEY }}",
          "event_action": "trigger",
          "payload": {
            "summary": "API Health Check Critical Failure",
            "source": "workflow-monitor",
            "severity": "critical",
            "custom_details": {
              "consecutive_failures": "{{ consecutive_failures }}",
              "api_endpoint": "https://api.example.com/health",
              "workflow_id": "{{ workflow_id }}"
            }
          }
        }
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
  }
}

export default function WorkflowYamlCreator({ onSuccess, onCancel }: Props) {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    yamlContent: ''
  })
  const [selectedTemplate, setSelectedTemplate] = useState<string>('')
  const [validationResult, setValidationResult] = useState<any>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [isGeneratingCode, setIsGeneratingCode] = useState(false)
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
      setValidationResult(null)
    }
  }

  const extractNameFromYaml = (yaml: string): string => {
    try {
      const nameMatch = yaml.match(/^name:\s*(.+)$/m)
      return nameMatch ? nameMatch[1].replace(/["']/g, '').trim() : ''
    } catch {
      return ''
    }
  }

  const validateYaml = async () => {
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
          const workflowName = formData.name || extractNameFromYaml(formData.yamlContent) || 'workflow'
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
          alert('Code generated and downloaded successfully!')
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

  const handleCreate = async () => {
    if (!formData.name.trim()) {
      alert('Please enter a workflow name')
      return
    }
    
    if (!formData.yamlContent.trim()) {
      alert('Please enter YAML content')
      return
    }

    // Auto-validate if not already validated
    if (!validationResult) {
      await validateYaml()
      return
    }

    if (!validationResult.is_valid) {
      alert('Please fix validation errors before creating the workflow')
      return
    }

    setIsCreating(true)
    try {
      console.log('🚀 Creating workflow from YAML...')
      const workflow = {
        name: formData.name,
        yaml_content: formData.yamlContent,
        auto_start: false
      }

      const result = await workflowApi.createWorkflowFromYaml(workflow)
      console.log('✅ Workflow created:', result)
      
      alert(`Workflow "${formData.name}" created successfully!`)
      onSuccess?.()
    } catch (error: any) {
      console.error('❌ Failed to create workflow:', error)
      alert(`Failed to create workflow: ${error.message || 'Unknown error'}`)
    } finally {
      setIsCreating(false)
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

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex justify-between items-center p-6 border-b bg-gray-50">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center">
            <Plus className="w-6 h-6 mr-2 text-blue-600" />
            Create New Workflow
          </h2>
          <p className="text-gray-600 mt-1">Design your workflow using YAML configuration with our advanced editor</p>
        </div>
        <button
          onClick={onCancel}
          className="text-gray-400 hover:text-gray-600 p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          ✕
        </button>
      </div>

      <div className="flex-1 overflow-hidden">
        <div className="h-full flex">
          {/* Left Sidebar - Templates & Settings */}
          <div className="w-80 border-r bg-gray-50 overflow-y-auto">
            <div className="p-4">
              {/* Workflow Info */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Workflow Details</h3>
                
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
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                    />
                  </div>
                </div>
              </div>

              {/* Templates */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Wand2 className="w-5 h-5 mr-2 text-purple-600" />
                  Templates
                </h3>
                <div className="space-y-2">
                  {Object.entries(WORKFLOW_TEMPLATES).map(([key, template]) => (
                    <button
                      key={key}
                      onClick={() => handleTemplateSelect(key)}
                      className={`w-full text-left p-3 rounded-lg border transition-all ${
                        selectedTemplate === key
                          ? 'border-blue-500 bg-blue-50 shadow-sm'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-white'
                      }`}
                    >
                      <div className="font-medium text-gray-900 text-sm">{template.name}</div>
                      <div className="text-xs text-gray-600 mt-1 line-clamp-2">{template.description}</div>
                    </button>
                  ))}
                  
                  <button
                    onClick={clearEditor}
                    className="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-gray-300 hover:bg-white transition-all"
                  >
                    <div className="font-medium text-gray-900 text-sm">🆕 Start from Scratch</div>
                    <div className="text-xs text-gray-600 mt-1">Create a custom workflow from scratch</div>
                  </button>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="space-y-2">
                <button
                  onClick={validateYaml}
                  disabled={isValidating || !formData.yamlContent.trim()}
                  className="w-full px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isValidating ? 'Validating...' : '✓ Validate YAML'}
                </button>
                
                <button
                  onClick={copyToClipboard}
                  disabled={!formData.yamlContent.trim()}
                  className="w-full px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                >
                  <Copy className="w-4 h-4 mr-1" />
                  Copy YAML
                </button>
              </div>
            </div>
          </div>

          {/* Center - Code Editor */}
          <div className="flex-1 flex flex-col">
            <div className="flex justify-between items-center px-4 py-2 border-b bg-white">
              <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                <FileText className="w-5 h-5 mr-2 text-gray-600" />
                YAML Editor
              </h3>
              <div className="flex space-x-2">
                <button
                  onClick={handleGenerateCode}
                  disabled={isGeneratingCode || !formData.yamlContent.trim()}
                  className="px-3 py-1 bg-green-100 hover:bg-green-200 text-green-700 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
                >
                  <Code className="w-4 h-4 mr-1" />
                  {isGeneratingCode ? 'Generating...' : 'Generate Code'}
                </button>
              </div>
            </div>
            
            <div className="flex-1">
              <Editor
                height="100%"
                defaultLanguage="yaml"
                value={formData.yamlContent}
                onChange={(value) => setFormData({ ...formData, yamlContent: value || '' })}
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
            </div>
          </div>

          {/* Right Panel - Validation Results */}
          <div className="w-96 border-l bg-gray-50 overflow-y-auto">
            <div className="p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Play className="w-5 h-5 mr-2 text-green-600" />
                Validation & Preview
              </h3>
              
              {!validationResult && (
                <div className="text-center py-8 text-gray-500">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
                    <Play className="w-8 h-8 text-gray-400" />
                  </div>
                  <p className="text-sm">Click "Validate YAML" to check your workflow configuration</p>
                </div>
              )}

              {validationResult && (
                <div className="space-y-4">
                  {/* Status Badge */}
                  <div className={`p-4 rounded-lg border ${
                    validationResult.is_valid 
                      ? 'bg-green-50 border-green-200' 
                      : 'bg-red-50 border-red-200'
                  }`}>
                    <div className="flex items-center">
                      {validationResult.is_valid ? (
                        <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
                      ) : (
                        <div className="w-3 h-3 bg-red-500 rounded-full mr-3"></div>
                      )}
                      <span className={`font-semibold ${
                        validationResult.is_valid ? 'text-green-800' : 'text-red-800'
                      }`}>
                        {validationResult.is_valid ? '✓ Valid Configuration' : '✗ Invalid Configuration'}
                      </span>
                    </div>
                  </div>

                  {/* Errors */}
                  {validationResult.errors && validationResult.errors.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-red-800 mb-2 flex items-center">
                        <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
                        Errors ({validationResult.errors.length})
                      </h4>
                      <div className="space-y-2">
                        {validationResult.errors.map((error: any, index: number) => (
                          <div key={index} className="text-red-700 text-sm bg-red-50 p-3 rounded-md border border-red-200">
                            {error.message || error}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Warnings */}
                  {validationResult.warnings && validationResult.warnings.length > 0 && (
                    <div>
                      <h4 className="font-semibold text-yellow-800 mb-2 flex items-center">
                        <span className="w-2 h-2 bg-yellow-500 rounded-full mr-2"></span>
                        Warnings ({validationResult.warnings.length})
                      </h4>
                      <div className="space-y-2">
                        {validationResult.warnings.map((warning: any, index: number) => (
                          <div key={index} className="text-yellow-700 text-sm bg-yellow-50 p-3 rounded-md border border-yellow-200">
                            {warning.message || warning}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Info */}
                  {validationResult.info && Object.keys(validationResult.info).length > 0 && (
                    <div>
                      <h4 className="font-semibold text-blue-800 mb-2 flex items-center">
                        <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                        Workflow Info
                      </h4>
                      <div className="bg-blue-50 p-3 rounded-md border border-blue-200">
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

      {/* Footer Actions */}
      <div className="flex justify-between items-center px-6 py-4 border-t bg-gray-50">
        <div className="flex space-x-2">
          <button
            onClick={clearEditor}
            className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
          >
            Clear All
          </button>
        </div>
        
        <div className="flex space-x-3">
          <button
            onClick={onCancel}
            className="px-6 py-2 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={isCreating || !formData.name.trim() || !formData.yamlContent.trim()}
            className="px-8 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
          >
            <Save className="w-4 h-4 mr-2" />
            {isCreating ? 'Creating...' : 'Create Workflow'}
          </button>
        </div>
      </div>
    </div>
  )
}