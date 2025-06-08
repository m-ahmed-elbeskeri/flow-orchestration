import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/Layout'
import WorkflowList from './components/WorkflowList'
import WorkflowCreator from './components/WorkflowCreator'
import WorkflowDesigner from './components/WorkflowDesigner'
import ExecutionMonitor from './components/ExecutionMonitor'
import Dashboard from './components/Dashboard'

// Create QueryClient instance
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            {/* Default redirect */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            
            {/* Dashboard */}
            <Route path="/dashboard" element={<Dashboard />} />
            
            {/* Workflows */}
            <Route path="/workflows" element={<WorkflowList />} />
            <Route path="/workflows/create" element={<WorkflowCreator />} />
            <Route path="/workflows/:id/edit" element={<WorkflowCreator />} />
            <Route path="/workflows/:id/designer" element={<WorkflowDesigner />} />
            
            {/* Executions */}
            <Route path="/workflows/:workflowId/executions/:executionId" element={<ExecutionMonitor />} />
            
            {/* Catch all - redirect to dashboard */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App