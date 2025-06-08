/**
 * Utility functions for downloading files
 */

export const downloadFile = (content: string, filename: string, contentType: string = 'text/plain') => {
  const blob = new Blob([content], { type: contentType })
  const url = URL.createObjectURL(blob)
  
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.style.display = 'none'
  
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  
  // Clean up the URL object
  URL.revokeObjectURL(url)
}

export const downloadBase64File = (base64Content: string, filename: string, contentType: string = 'application/octet-stream') => {
  // Convert base64 to blob
  const byteCharacters = atob(base64Content)
  const byteNumbers = new Array(byteCharacters.length)
  
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i)
  }
  
  const byteArray = new Uint8Array(byteNumbers)
  const blob = new Blob([byteArray], { type: contentType })
  
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.style.display = 'none'
  
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  
  URL.revokeObjectURL(url)
}

export const downloadJson = (data: any, filename: string) => {
  const jsonString = JSON.stringify(data, null, 2)
  downloadFile(jsonString, filename, 'application/json')
}

export const downloadYaml = (yamlContent: string, filename: string) => {
  downloadFile(yamlContent, filename, 'application/x-yaml')
}

export const downloadBlob = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.style.display = 'none'
  
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  
  URL.revokeObjectURL(url)
}

export const downloadCSV = (data: any[], filename: string, headers?: string[]) => {
  if (!data || data.length === 0) {
    throw new Error('No data to export')
  }

  // Generate headers from first object if not provided
  const csvHeaders = headers || Object.keys(data[0])
  
  // Create CSV content
  const csvContent = [
    csvHeaders.join(','), // Header row
    ...data.map(row => 
      csvHeaders.map(header => {
        const value = row[header]
        // Escape values that contain commas or quotes
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`
        }
        return value
      }).join(',')
    )
  ].join('\n')

  downloadFile(csvContent, filename, 'text/csv')
}

export const downloadExecutionData = async (
  workflowApi: any,
  workflowId: string,
  executionId: string,
  format: 'json' | 'csv' = 'json'
) => {
  try {
    const blob = await workflowApi.exportExecutionData(workflowId, executionId, format)
    const filename = `execution-${executionId}-data.${format}`
    downloadBlob(blob, filename)
  } catch (error) {
    console.error('Failed to download execution data:', error)
    throw error
  }
}

export const downloadWorkflowData = async (
  workflowApi: any,
  workflowId: string,
  format: 'json' | 'csv' | 'xlsx' = 'json'
) => {
  try {
    const blob = await workflowApi.exportWorkflowData(workflowId, format)
    const filename = `workflow-${workflowId}-data.${format}`
    downloadBlob(blob, filename)
  } catch (error) {
    console.error('Failed to download workflow data:', error)
    throw error
  }
}