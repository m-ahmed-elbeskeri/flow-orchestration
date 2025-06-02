import { useState } from 'react'
import { Code } from 'lucide-react'
import { workflowApi } from '../api/client'
import { downloadFile } from '../utils/download'
import type { WorkflowCreateRequest } from '../types/workflow'

interface Props {
  onSuccess?: () => void
  onCancel?: () => void
}

const EXAMPLE_WORKFLOWS = {
  simple: `name: simple_workflow
version: 1.0.0
description: "A simple workflow example"
author: "User"

states:
  - name: start
    type: builtin.start
    transitions:
      - on_success: process_data

  - name: process_data
    type: builtin.transform
    config:
      message: "Processing data..."
    transitions:
      - on_success: end

  - name: end
    type: builtin.end`,

  email_processor: `name: email_processor
version: 1.0.0
description: "Process urgent emails and send notifications"
author: "Team Workflow"

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

  - name: error_handler
    type: builtin.error_handler
    config:
      log_level: ERROR
    transitions:
      - on_complete: end

  - name: end
    type: builtin.end`,

  api_integration: `name: api_integration
version: 1.0.0
description: "Fetch data from API and process it"
author: "API Team"

states:
  - name: start
    type: builtin.start
    transitions:
      - on_success: fetch_data

  - name: fetch_data
    type: webhook.http_request
    config:
      url: "https://api.example.com/data"
      method: GET
      headers:
        Authorization: "Bearer {{ API_TOKEN }}"
    transitions:
      - on_success: process_response
      - on_failure: error_handler

  - name: process_response
    type: builtin.transform
    config:
      script: |
        data = context.get_state("api_response")
        processed = [item for item in data if item.get("active")]
        context.set_state("processed_data", processed)
    transitions:
      - on_success: store_results

  - name: store_results
    type: builtin.transform
    config:
      message: "Storing {{ processed_data|length }} items"
    transitions:
      - on_success: end

  - name: error_handler
    type: builtin.error_handler
    transitions:
      - on_complete: end

  - name: end
    type: builtin.end`,

  data_pipeline: `name: data_pipeline
version: 1.0.0
description: "Complete data processing pipeline"
author: "Data Team"

environment:
  variables:
    INPUT_PATH: "/data/input"
    OUTPUT_PATH: "/data/output"
    BATCH_SIZE: "1000"
  secrets:
    - DATABASE_URL
    - API_KEY

states:
  - name: start
    type: builtin.start
    transitions:
      - on_success: validate_input

  - name: validate_input
    type: builtin.conditional
    config:
      condition: os.path.exists(context.get_constant("INPUT_PATH"))
    transitions:
      - on_true: load_data
      - on_false: error_handler

  - name: load_data
    type: builtin.transform
    config:
      script: |
        import pandas as pd
        input_path = context.get_constant("INPUT_PATH")
        data = pd.read_csv(f"{input_path}/data.csv")
        context.set_state("raw_data", data.to_dict())
    transitions:
      - on_success: clean_data

  - name: clean_data
    type: builtin.transform
    config:
      script: |
        raw_data = context.get_state("raw_data")
        # Clean and validate data
        cleaned_data = {k: v for k, v in raw_data.items() if v is not None}
        context.set_state("cleaned_data", cleaned_data)
    transitions:
      - on_success: process_batch

  - name: process_batch
    type: builtin.transform
    config:
      script: |
        cleaned_data = context.get_state("cleaned_data")
        batch_size = int(context.get_constant("BATCH_SIZE"))
        # Process in batches
        processed_count = len(cleaned_data)
        context.set_state("processed_count", processed_count)
    transitions:
      - on_success: save_results

  - name: save_results
    type: builtin.transform
    config:
      script: |
        output_path = context.get_constant("OUTPUT_PATH")
        processed_count = context.get_state("processed_count")
        # Save results
        context.set_output("results_saved", True)
        context.set_output("records_processed", processed_count)
    transitions:
      - on_success: end

  - name: error_handler
    type: builtin.error_handler
    config:
      log_level: ERROR
      notify: true
    transitions:
      - on_complete: end

  - name: end
    type: builtin.end`
}

export default function WorkflowYamlCreator({ onSuccess, onCancel }: Props) {
  const [yamlContent, setYamlContent] = useState('')
  const [workflowName, setWorkflowName] = useState('')
  const [isCreating, setIsCreating] = useState(false)
  const [isGeneratingCode, setIsGeneratingCode] = useState(false)
  const [validationResult, setValidationResult] = useState<any>(null)
  const [selectedExample, setSelectedExample] = useState<keyof typeof EXAMPLE_WORKFLOWS | ''>('')

  const handleExampleSelect = (exampleKey: keyof typeof EXAMPLE_WORKFLOWS) => {
    setYamlContent(EXAMPLE_WORKFLOWS[exampleKey])
    setSelectedExample(exampleKey)
    
    // Extract name from YAML for convenience
    const nameMatch = EXAMPLE_WORKFLOWS[exampleKey].match(/name:\s*(.+)/)
    if (nameMatch) {
      setWorkflowName(nameMatch[1].trim())
    }
    
    // Clear previous validation
    setValidationResult(null)
  }

  const validateYaml = async () => {
    if (!yamlContent.trim()) {
      alert('Please enter YAML content first')
      return
    }

    try {
      console.log('🔍 Validating YAML...')
      const response = await workflowApi.validate(yamlContent)
      setValidationResult(response.data)
      
      if (response.data.valid) {
        alert('YAML is valid! ✅')
      } else {
        alert(`YAML validation failed:\n${response.data.errors.join('\n')}`)
      }
    } catch (error: any) {
      console.error('Validation failed:', error)
      alert(`Validation error: ${error.response?.data?.detail || error.message}`)
    }
  }

  const handleGenerateCode = async () => {
    if (!yamlContent.trim()) {
      alert('Please enter YAML content first')
      return
    }

    try {
      setIsGeneratingCode(true)
      console.log('🔧 Generating code from YAML...')
      
      const response = await workflowApi.generateCodeFromYaml(yamlContent)
      console.log('✅ Code generated:', response.data)
      
      if (response.data.success) {
        downloadFile(
          response.data.python_code,
          response.data.file_name,
          'text/x-python'
        )
        alert(`Python code downloaded: ${response.data.file_name}`)
      } else {
        alert('Code generation failed')
      }
      
    } catch (error: any) {
      console.error('❌ Code generation failed:', error)
      
      let errorMessage = 'Unknown error'
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message
      } else if (error.message) {
        errorMessage = error.message
      }
      
      alert(`Failed to generate code: ${errorMessage}`)
    } finally {
      setIsGeneratingCode(false)
    }
  }

  const handleCreate = async () => {
    if (!workflowName.trim()) {
      alert('Please enter a workflow name')
      return
    }

    if (!yamlContent.trim()) {
      alert('Please enter YAML content')
      return
    }

    try {
      setIsCreating(true)

      const workflowData: WorkflowCreateRequest = {
        name: workflowName.trim(),
        yaml_content: yamlContent,
        auto_start: false
      }

      console.log('Creating workflow from YAML:', workflowData)

      const response = await workflowApi.createFromYaml(workflowData)
      console.log('Workflow created:', response.data)

      alert('Workflow created successfully! ✅')
      
      // Reset form
      setWorkflowName('')
      setYamlContent('')
      setValidationResult(null)
      setSelectedExample('')
      
      onSuccess?.()

    } catch (error: any) {
      console.error('Failed to create workflow:', error)
      
      let errorMessage = 'Unknown error'
      if (error.response?.data?.detail) {
        errorMessage = Array.isArray(error.response.data.detail) 
          ? error.response.data.detail.map((e: any) => e.msg || e).join(', ')
          : error.response.data.detail
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message
      } else if (error.message) {
        errorMessage = error.message
      }

      alert(`Failed to create workflow: ${errorMessage}`)
    } finally {
      setIsCreating(false)
    }
  }

  const clearForm = () => {
    setYamlContent('')
    setSelectedExample('')
    setWorkflowName('')
    setValidationResult(null)
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg">
        <div className="p-6 border-b">
          <h2 className="text-2xl font-bold">Create Workflow from YAML</h2>
          <p className="text-gray-600 mt-2">
            Define your workflow using YAML configuration. You can use the examples below or write your own.
          </p>
        </div>

        <div className="p-6">
          {/* Example Templates */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-3">
              Choose an example template:
            </label>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => handleExampleSelect('simple')}
                className={`px-4 py-2 rounded text-sm transition-colors ${
                  selectedExample === 'simple' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                📝 Simple Workflow
              </button>
              <button
                onClick={() => handleExampleSelect('email_processor')}
                className={`px-4 py-2 rounded text-sm transition-colors ${
                  selectedExample === 'email_processor' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                📧 Email Processor
              </button>
              <button
                onClick={() => handleExampleSelect('api_integration')}
                className={`px-4 py-2 rounded text-sm transition-colors ${
                  selectedExample === 'api_integration' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                🌐 API Integration
              </button>
              <button
                onClick={() => handleExampleSelect('data_pipeline')}
                className={`px-4 py-2 rounded text-sm transition-colors ${
                  selectedExample === 'data_pipeline' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                📊 Data Pipeline
              </button>
              <button
                onClick={clearForm}
                className="px-4 py-2 rounded text-sm bg-gray-400 text-white hover:bg-gray-500 transition-colors"
              >
                🗑️ Clear
              </button>
            </div>
          </div>

          {/* Workflow Name */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Workflow Name <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={workflowName}
              onChange={(e) => setWorkflowName(e.target.value)}
              className="w-full px-3 py-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Enter workflow name"
              required
            />
          </div>

          {/* YAML Content */}
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <label className="block text-sm font-medium">
                YAML Content <span className="text-red-500">*</span>
              </label>
              <div className="flex gap-2">
                <button
                  onClick={validateYaml}
                  disabled={!yamlContent.trim()}
                  className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700 disabled:opacity-50 transition-colors"
                >
                  ✅ Validate YAML
                </button>
                <button
                  onClick={handleGenerateCode}
                  disabled={!yamlContent.trim() || isGeneratingCode}
                  className="px-3 py-1 bg-purple-600 text-white text-sm rounded hover:bg-purple-700 disabled:opacity-50 transition-colors flex items-center gap-1"
                >
                  {isGeneratingCode ? (
                    <>
                      <div className="w-3 h-3 animate-spin rounded-full border border-white border-t-transparent"></div>
                      Generating...
                    </>
                  ) : (
                    <>
                      <Code className="w-3 h-3" />
                      Generate Code
                    </>
                  )}
                </button>
              </div>
            </div>
            <textarea
              value={yamlContent}
              onChange={(e) => setYamlContent(e.target.value)}
              className="w-full h-96 px-3 py-2 border rounded-md font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
              placeholder="Enter your workflow YAML here..."
              required
            />
            <div className="text-xs text-gray-500 mt-1">
              Lines: {yamlContent.split('\n').length} | Characters: {yamlContent.length}
            </div>
          </div>

          {/* Validation Result */}
          {validationResult && (
            <div className={`mb-4 p-4 rounded-lg ${
              validationResult.valid 
                ? 'bg-green-50 border border-green-200' 
                : 'bg-red-50 border border-red-200'
            }`}>
              <div className="font-medium mb-2 flex items-center gap-2">
                {validationResult.valid ? (
                  <>
                    <span className="text-green-600">✅ Valid YAML</span>
                    {validationResult.workflow_info && (
                      <span className="text-sm text-green-700">
                        ({validationResult.workflow_info.states_count} states)
                      </span>
                    )}
                  </>
                ) : (
                  <span className="text-red-600">❌ Invalid YAML</span>
                )}
              </div>
              
              {validationResult.workflow_info && validationResult.valid && (
                <div className="text-sm text-green-700 mb-2">
                  <div><strong>Name:</strong> {validationResult.workflow_info.name}</div>
                  <div><strong>Description:</strong> {validationResult.workflow_info.description}</div>
                  <div><strong>Version:</strong> {validationResult.workflow_info.version}</div>
                </div>
              )}
              
              {validationResult.errors?.length > 0 && (
                <div className="mb-2">
                  <div className="text-sm font-medium text-red-800 mb-1">Errors:</div>
                  <ul className="text-sm text-red-600 list-disc list-inside">
                    {validationResult.errors.map((error: string, index: number) => (
                      <li key={index}>{error}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {validationResult.warnings?.length > 0 && (
                <div>
                  <div className="text-sm font-medium text-yellow-800 mb-1">Warnings:</div>
                  <ul className="text-sm text-yellow-600 list-disc list-inside">
                    {validationResult.warnings.map((warning: string, index: number) => (
                      <li key={index}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4 border-t">
            <button
              onClick={handleCreate}
              disabled={isCreating || !workflowName.trim() || !yamlContent.trim()}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              {isCreating ? (
                <>
                  <div className="w-4 h-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
                  Creating...
                </>
              ) : (
                '📝 Create Workflow'
              )}
            </button>
            
            <button
              onClick={handleGenerateCode}
              disabled={!yamlContent.trim() || isGeneratingCode}
              className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 disabled:opacity-50 transition-colors flex items-center gap-2"
            >
              {isGeneratingCode ? (
                <>
                  <div className="w-4 h-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
                  Generating...
                </>
              ) : (
                <>
                  <Code className="h-4 w-4" />
                  Generate Code
                </>
              )}
            </button>
            
            {onCancel && (
              <button
                onClick={onCancel}
                className="bg-gray-500 text-white px-6 py-3 rounded-lg hover:bg-gray-600 transition-colors"
              >
                Cancel
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}