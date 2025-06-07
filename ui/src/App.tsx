import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/Layout'
import WorkflowList from './components/WorkflowList'
import WorkflowCreator from './components/WorkflowCreator'
import WorkflowDesigner from './components/WorkflowDesigner'
import ExecutionMonitor from './components/ExecutionMonitor'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<WorkflowList />} />
            <Route path="/workflows" element={<WorkflowList />} />
            <Route path="/workflows/create" element={<WorkflowCreator />} />
            <Route path="/workflows/:id/designer" element={<WorkflowDesigner />} />
            <Route path="/workflows/:workflowId/executions/:executionId" element={<ExecutionMonitor />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App