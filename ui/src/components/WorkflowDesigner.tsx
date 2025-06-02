// src/components/WorkflowDesigner.tsx
import { useCallback, useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  Panel,
} from 'reactflow'
import { Save, Play, Plus } from 'lucide-react'
import 'reactflow/dist/style.css'

const nodeTypes = {
  start: { label: 'Start', color: '#10b981' },
  action: { label: 'Action', color: '#3b82f6' },
  condition: { label: 'Condition', color: '#f59e0b' },
  end: { label: 'End', color: '#ef4444' },
}

export default function WorkflowDesigner() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  
  // Load workflow
  useEffect(() => {
    // TODO: Load from API
    // For now, create a simple example
    setNodes([
      {
        id: '1',
        type: 'default',
        position: { x: 250, y: 0 },
        data: { label: 'Start', type: 'start' },
        style: { background: '#10b981', color: 'white' },
      },
      {
        id: '2',
        type: 'default',
        position: { x: 250, y: 100 },
        data: { label: 'Process Data', type: 'action' },
        style: { background: '#3b82f6', color: 'white' },
      },
      {
        id: '3',
        type: 'default',
        position: { x: 250, y: 200 },
        data: { label: 'End', type: 'end' },
        style: { background: '#ef4444', color: 'white' },
      },
    ])
    setEdges([
      { id: 'e1-2', source: '1', target: '2' },
      { id: 'e2-3', source: '2', target: '3' },
    ])
  }, [id])
  
  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  )
  
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node)
  }, [])
  
  const addNode = (type: keyof typeof nodeTypes) => {
    const newNode: Node = {
      id: `${nodes.length + 1}`,
      type: 'default',
      position: { x: 250, y: nodes.length * 100 },
      data: { label: nodeTypes[type].label, type },
      style: { background: nodeTypes[type].color, color: 'white' },
    }
    setNodes((nds) => [...nds, newNode])
  }
  
  const saveWorkflow = async () => {
    // TODO: Save to API
    console.log('Saving workflow:', { nodes, edges })
    alert('Workflow saved!')
  }
  
  const executeWorkflow = () => {
    navigate(`/workflows/${id}/execute`)
  }
  
  return (
    <div className="h-screen flex">
      {/* Main Canvas */}
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          fitView
        >
          <Background />
          <Controls />
          <MiniMap />
          
          <Panel position="top-left" className="bg-white p-4 rounded shadow-lg">
            <div className="space-y-2">
              <h3 className="font-semibold mb-2">Add Node</h3>
              {Object.entries(nodeTypes).map(([type, config]) => (
                <button
                  key={type}
                  onClick={() => addNode(type as keyof typeof nodeTypes)}
                  className="flex items-center gap-2 px-3 py-1 rounded text-sm hover:bg-gray-100 w-full"
                  style={{ color: config.color }}
                >
                  <Plus className="h-4 w-4" />
                  {config.label}
                </button>
              ))}
            </div>
          </Panel>
          
          <Panel position="top-right" className="space-x-2">
            <button
              onClick={saveWorkflow}
              className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              <Save className="h-4 w-4 mr-2" />
              Save
            </button>
            <button
              onClick={executeWorkflow}
              className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              <Play className="h-4 w-4 mr-2" />
              Execute
            </button>
          </Panel>
        </ReactFlow>
      </div>
      
      {/* Properties Panel */}
      {selectedNode && (
        <div className="w-80 bg-white border-l border-gray-200 p-4">
          <h3 className="font-semibold mb-4">Node Properties</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Label</label>
              <input
                type="text"
                value={selectedNode.data.label}
                onChange={(e) => {
                  setNodes((nds) =>
                    nds.map((node) =>
                      node.id === selectedNode.id
                        ? { ...node, data: { ...node.data, label: e.target.value } }
                        : node
                    )
                  )
                }}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Type</label>
              <select
                value={selectedNode.data.type}
                onChange={(e) => {
                  const newType = e.target.value as keyof typeof nodeTypes
                  setNodes((nds) =>
                    nds.map((node) =>
                      node.id === selectedNode.id
                        ? {
                            ...node,
                            data: { ...node.data, type: newType },
                            style: { background: nodeTypes[newType].color, color: 'white' },
                          }
                        : node
                    )
                  )
                }}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                {Object.entries(nodeTypes).map(([type, config]) => (
                  <option key={type} value={type}>
                    {config.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}