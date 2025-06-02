export const downloadFile = (content: string, filename: string, contentType: string = 'text/plain') => {
  try {
    const blob = new Blob([content], { type: contentType })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    link.style.display = 'none'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    console.log('✅ File downloaded:', filename)
  } catch (error) {
    console.error('❌ Download failed:', error)
    alert('Download failed. Please try again.')
  }
}

export const downloadBase64File = (base64Content: string, filename: string, contentType: string = 'application/octet-stream') => {
  try {
    const byteCharacters = atob(base64Content)
    const byteNumbers = new Array(byteCharacters.length)
    
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i)
    }
    
    const byteArray = new Uint8Array(byteNumbers)
    const blob = new Blob([byteArray], { type: contentType })
    
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    link.style.display = 'none'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    console.log('✅ Base64 file downloaded:', filename)
  } catch (error) {
    console.error('❌ Base64 download failed:', error)
    alert('Download failed. Please try again.')
  }
}

export const downloadJson = (data: any, filename: string) => {
  const jsonString = JSON.stringify(data, null, 2)
  downloadFile(jsonString, filename, 'application/json')
}

export const downloadYaml = (yamlContent: string, filename: string) => {
  downloadFile(yamlContent, filename, 'text/yaml')
}