// ui/src/components/VisualWorkflowDesigner.tsx
import React, { useState, useCallback, useMemo, useRef } from 'react';
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
  MarkerType,
  EdgeProps,
  getBezierPath,
  EdgeLabelRenderer,
  BaseEdge,
  ConnectionLineType,
  useReactFlow,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  Plus,
  Settings,
  Trash2,
  X,
  Check,
  Clock,
  Zap,
  Mail,
  Database,
  Code,
  GitBranch,
  AlertTriangle,
  Eye,
  Play
} from 'lucide-react';

interface VisualWorkflowDesignerProps {
  initialYaml?: string;
  onChange?: (yaml: string) => void;
  onSave?: (yaml: string) => void;
}

// Custom Edge Component with Click Handler
const CustomEdge = ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, selected }: EdgeProps) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const onEdgeClick = (event: React.MouseEvent) => {
    event.stopPropagation();
    if (data?.onClick) {
      data.onClick(id);
    }
  };

  const getDependencyIcon = (type: string) => {
    switch (type) {
      case 'required': return '⚡';
      case 'optional': return '⭕';
      case 'conditional': return '🔀';
      case 'timeout': return '⏰';
      default: return '→';
    }
  };

  return (
    <>
      <BaseEdge 
        path={edgePath} 
        style={{ 
          stroke: selected ? '#00d4ff' : data?.dependencyType === 'required' ? '#4ade80' : 
                  data?.dependencyType === 'optional' ? '#f59e0b' :
                  data?.dependencyType === 'conditional' ? '#8b5cf6' : '#6b7280',
          strokeWidth: selected ? 3 : 2,
          strokeDasharray: data?.dependencyType === 'optional' ? '5,5' : 'none'
        }}
      />
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            fontSize: 12,
            pointerEvents: 'all',
          }}
          className="nodrag nopan"
        >
          <button
            onClick={onEdgeClick}
            className={`
              flex items-center justify-center w-6 h-6 rounded-full text-white text-xs font-bold
              ${selected ? 'bg-blue-500 ring-2 ring-blue-300' : 'bg-gray-600 hover:bg-gray-500'}
              transition-all duration-200 shadow-lg
            `}
            title={`Click to configure dependency (${data?.dependencyType || 'default'})`}
          >
            {getDependencyIcon(data?.dependencyType)}
          </button>
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

// Custom Node Components
const WorkflowNode = ({ id, data, selected }: { id: string; data: any; selected: boolean }) => {
  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'builtin.start': return <Play className="w-4 h-4" />;
      case 'builtin.end': return <Check className="w-4 h-4" />;
      case 'builtin.conditional': return <GitBranch className="w-4 h-4" />;
      case 'builtin.transform': return <Code className="w-4 h-4" />;
      case 'builtin.delay': return <Clock className="w-4 h-4" />;
      case 'gmail.send_email': return <Mail className="w-4 h-4" />;
      case 'gmail.read_emails': return <Eye className="w-4 h-4" />;
      case 'slack.send_message': return <Zap className="w-4 h-4" />;
      case 'webhook.http_request': return <Database className="w-4 h-4" />;
      default: return <Settings className="w-4 h-4" />;
    }
  };

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'builtin.start': return 'bg-green-500';
      case 'builtin.end': return 'bg-red-500';
      case 'builtin.conditional': return 'bg-purple-500';
      case 'builtin.transform': return 'bg-blue-500';
      case 'builtin.delay': return 'bg-yellow-500';
      case 'gmail.send_email': 
      case 'gmail.read_emails': return 'bg-red-600';
      case 'slack.send_message': return 'bg-green-600';
      case 'webhook.http_request': return 'bg-orange-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className={`
      relative bg-white rounded-lg shadow-lg border-2 min-w-[200px] max-w-[250px]
      ${selected ? 'border-blue-500 shadow-blue-200' : 'border-gray-300'}
      transition-all duration-200 hover:shadow-xl
    `}>
      {/* Node Header */}
      <div className={`
        flex items-center gap-2 p-3 rounded-t-lg text-white
        ${getNodeColor(data.type)}
      `}>
        {getNodeIcon(data.type)}
        <span className="font-semibold text-sm truncate">{data.label}</span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            data.onEdit?.(id);
          }}
          className="ml-auto p-1 hover:bg-white/20 rounded"
        >
          <Settings className="w-3 h-3" />
        </button>
      </div>

      {/* Node Content */}
      <div className="p-3">
        <div className="text-xs text-gray-600 mb-2">
          {data.type}
        </div>
        {data.description && (
          <div className="text-xs text-gray-800 mb-2 line-clamp-2">
            {data.description}
          </div>
        )}
        {data.config && Object.keys(data.config).length > 0 && (
          <div className="text-xs text-gray-500">
            {Object.keys(data.config).length} config item(s)
          </div>
        )}
      </div>

      {/* Connection Points */}
      {data.type !== 'builtin.start' && (
        <div className="absolute -top-2 left-1/2 transform -translate-x-1/2">
          <div className="w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-lg"></div>
        </div>
      )}
      {data.type !== 'builtin.end' && (
        <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2">
          <div className="w-4 h-4 bg-green-500 rounded-full border-2 border-white shadow-lg"></div>
        </div>
      )}
    </div>
  );
};

// Node Types Configuration
const nodeTypes = {
  workflowNode: WorkflowNode,
};

const edgeTypes = {
  custom: CustomEdge,
};

// Main Workflow Creator Component
const VisualWorkflowDesignerContent = ({ initialYaml, onChange, onSave }: VisualWorkflowDesignerProps) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showNodePalette, setShowNodePalette] = useState(false);
  const [showDependencyConfig, setShowDependencyConfig] = useState(false);
  const [showNodeConfig, setShowNodeConfig] = useState(false);
  const [workflowName, setWorkflowName] = useState('New Workflow');

  const reactFlowInstance = useReactFlow();

  const nodeTemplates = [
    { type: 'builtin.start', label: 'Start', description: 'Workflow starting point' },
    { type: 'builtin.end', label: 'End', description: 'Workflow completion' },
    { type: 'builtin.conditional', label: 'Conditional', description: 'Branch workflow based on conditions' },
    { type: 'builtin.transform', label: 'Transform', description: 'Process and transform data' },
    { type: 'builtin.delay', label: 'Delay', description: 'Wait for specified time' },
    { type: 'gmail.send_email', label: 'Send Email', description: 'Send email via Gmail' },
    { type: 'gmail.read_emails', label: 'Read Emails', description: 'Read emails from Gmail' },
    { type: 'slack.send_message', label: 'Slack Message', description: 'Send message to Slack' },
    { type: 'webhook.http_request', label: 'HTTP Request', description: 'Make HTTP request' },
  ];

  const dependencyTypes = [
    { value: 'required', label: 'Required', description: 'Must complete successfully', color: 'green' },
    { value: 'optional', label: 'Optional', description: 'Can skip if fails', color: 'yellow' },
    { value: 'conditional', label: 'Conditional', description: 'Based on condition', color: 'purple' },
    { value: 'timeout', label: 'Timeout', description: 'With timeout limit', color: 'red' },
  ];

  // Initialize from YAML if provided
  React.useEffect(() => {
    if (initialYaml) {
      try {
        parseYamlToNodes(initialYaml);
      } catch (error) {
        console.error('Failed to parse initial YAML:', error);
      }
    }
  }, [initialYaml]);

  // Emit changes
  React.useEffect(() => {
    if (nodes.length > 0 || edges.length > 0) {
      const yaml = generateYAML();
      onChange?.(yaml);
    }
  }, [nodes, edges, workflowName]);

  const parseYamlToNodes = (yamlContent: string) => {
    // Basic YAML parsing - you might want to use a proper YAML parser
    const lines = yamlContent.split('\n');
    const nameMatch = lines.find(line => line.startsWith('name:'));
    if (nameMatch) {
      setWorkflowName(nameMatch.split(':')[1].trim());
    }
    
    // For now, just create a start and end node as example
    // You can enhance this to properly parse existing YAML
    const startNode: Node = {
      id: 'start-node',
      type: 'workflowNode',
      position: { x: 100, y: 100 },
      data: {
        type: 'builtin.start',
        label: 'Start',
        description: 'Workflow starting point',
        config: {},
        onEdit: (nodeId: string) => {
          setSelectedNode(nodeId);
          setShowNodeConfig(true);
        }
      },
    };

    const endNode: Node = {
      id: 'end-node',
      type: 'workflowNode',
      position: { x: 100, y: 300 },
      data: {
        type: 'builtin.end',
        label: 'End',
        description: 'Workflow completion',
        config: {},
        onEdit: (nodeId: string) => {
          setSelectedNode(nodeId);
          setShowNodeConfig(true);
        }
      },
    };

    setNodes([startNode, endNode]);
  };

  const onConnect = useCallback((params: Connection) => {
    const newEdge = {
      ...params,
      id: `e${params.source}-${params.target}`,
      type: 'custom',
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 20,
        height: 20,
        color: '#6b7280',
      },
      data: {
        dependencyType: 'required',
        onClick: (edgeId: string) => {
          setSelectedEdge(edgeId);
          setShowDependencyConfig(true);
        }
      }
    };
    setEdges((eds) => addEdge(newEdge, eds));
  }, [setEdges]);

  const addNode = useCallback((template: any, position?: { x: number; y: number }) => {
    const newNode: Node = {
      id: `node-${Date.now()}`,
      type: 'workflowNode',
      position: position || { x: Math.random() * 400 + 100, y: Math.random() * 400 + 100 },
      data: {
        ...template,
        config: {},
        onEdit: (nodeId: string) => {
          setSelectedNode(nodeId);
          setShowNodeConfig(true);
        }
      },
    };
    setNodes((nds) => [...nds, newNode]);
    setShowNodePalette(false);
  }, [setNodes]);

  const updateEdgeDependency = useCallback((edgeId: string, dependencyType: string, config?: any) => {
    setEdges((eds) =>
      eds.map((edge) =>
        edge.id === edgeId
          ? {
              ...edge,
              data: {
                ...edge.data,
                dependencyType,
                config,
              },
              style: {
                stroke: dependencyType === 'required' ? '#4ade80' : 
                       dependencyType === 'optional' ? '#f59e0b' :
                       dependencyType === 'conditional' ? '#8b5cf6' : '#6b7280',
                strokeDasharray: dependencyType === 'optional' ? '5,5' : 'none'
              }
            }
          : edge
      )
    );
  }, [setEdges]);

  const updateNodeConfig = useCallback((nodeId: string, updates: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              data: {
                ...node.data,
                ...updates,
              },
            }
          : node
      )
    );
  }, [setNodes]);

  const deleteSelected = useCallback(() => {
    setNodes((nds) => nds.filter((node) => !node.selected));
    setEdges((eds) => eds.filter((edge) => !edge.selected));
  }, [setNodes, setEdges]);

  const generateYAML = () => {
    const workflow = {
      name: workflowName,
      version: '1.0.0',
      description: 'Created with visual workflow designer',
      author: 'Visual Designer',
      nodes: nodes.map(node => ({
        id: node.id,
        type: node.data.type,
        position: node.position,
        config: node.data.config || {},
        label: node.data.label,
        description: node.data.description,
      })),
      edges: edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        dependencyType: edge.data?.dependencyType || 'required',
        config: edge.data?.config || {},
      })),
    };

    const states = workflow.nodes.map((node: any) => {
      const state: any = {
        name: node.id.replace('node-', 'state_').replace('-node', ''),
        type: node.type,
        description: node.description || `${node.label} state`,
      };

      if (Object.keys(node.config).length > 0) {
        state.config = node.config;
      }

      // Find outgoing connections for transitions
      const outgoingEdges = workflow.edges.filter((edge: any) => edge.source === node.id);
      if (outgoingEdges.length > 0) {
        state.transitions = outgoingEdges.map((edge: any) => ({
          on_success: edge.target.replace('node-', 'state_').replace('-node', ''),
          condition: edge.dependencyType === 'conditional' ? edge.config?.condition : undefined,
        })).filter((t: any) => t.condition !== undefined || t.on_success);
      }

      // Find incoming connections for dependencies
      const incomingEdges = workflow.edges.filter((edge: any) => edge.target === node.id);
      if (incomingEdges.length > 0) {
        state.dependencies = incomingEdges.map((edge: any) => ({
          name: edge.source.replace('node-', 'state_').replace('-node', ''),
          type: edge.dependencyType,
          ...(edge.config || {}),
        }));
      }

      return state;
    });

    return `name: ${workflow.name}
version: "${workflow.version}"
description: "${workflow.description}"
author: "${workflow.author}"

config:
  timeout: 300
  max_concurrent: 5

environment:
  variables:
    LOG_LEVEL: INFO
  secrets: []

states:
${states.map((state: any) => `  - name: ${state.name}
    type: ${state.type}
    description: "${state.description}"${
      state.config ? `
    config:
${Object.entries(state.config).map(([key, value]) => `      ${key}: ${JSON.stringify(value)}`).join('\n')}` : ''
    }${
      state.dependencies ? `
    dependencies:
${state.dependencies.map((dep: any) => `      - name: ${dep.name}
        type: ${dep.type}`).join('\n')}` : ''
    }${
      state.transitions ? `
    transitions:
${state.transitions.map((trans: any) => `      - on_success: ${trans.on_success}`).join('\n')}` : ''
    }`).join('\n\n')}`;
  };

  const selectedEdgeData = useMemo(() => {
    if (!selectedEdge) return null;
    return edges.find(edge => edge.id === selectedEdge);
  }, [selectedEdge, edges]);

  const selectedNodeData = useMemo(() => {
    if (!selectedNode) return null;
    return nodes.find(node => node.id === selectedNode);
  }, [selectedNode, nodes]);

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="bg-white border-b px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <input
            type="text"
            value={workflowName}
            onChange={(e) => setWorkflowName(e.target.value)}
            className="text-lg font-semibold bg-transparent border-none outline-none"
            placeholder="Workflow Name"
          />
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowNodePalette(true)}
            className="flex items-center gap-2 px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            <Plus className="w-4 h-4" />
            Add Node
          </button>
          <button
            onClick={deleteSelected}
            className="flex items-center gap-2 px-3 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            <Trash2 className="w-4 h-4" />
            Delete
          </button>
        </div>
      </div>

      {/* Main Canvas */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          connectionLineType={ConnectionLineType.Bezier}
          fitView
          attributionPosition="bottom-left"
          onEdgeClick={(event, edge) => {
            setSelectedEdge(edge.id);
            setShowDependencyConfig(true);
          }}
          className="bg-gray-50"
        >
          <Controls />
          <MiniMap />
          <Background variant="dots" gap={20} size={1} />
          
          {/* Instructions Panel */}
          <Panel position="top-left" className="bg-white p-4 rounded-lg shadow-lg max-w-xs">
            <h3 className="font-semibold mb-2">Instructions</h3>
            <ul className="text-xs space-y-1 text-gray-600">
              <li>• Drag from bottom circle to create connections</li>
              <li>• Click edge buttons to configure dependencies</li>
              <li>• Click node settings to configure</li>
              <li>• Select and delete unwanted elements</li>
            </ul>
          </Panel>
        </ReactFlow>
      </div>

      {/* Modals */}
      {showNodePalette && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Add Node</h2>
              <button
                onClick={() => setShowNodePalette(false)}
                className="p-2 hover:bg-gray-100 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {nodeTemplates.map((template) => (
                <button
                  key={template.type}
                  onClick={() => addNode(template)}
                  className="p-4 border rounded-lg hover:border-blue-500 hover:bg-blue-50 text-left"
                >
                  <div className="font-semibold">{template.label}</div>
                  <div className="text-sm text-gray-500 mt-1">{template.type}</div>
                  <div className="text-xs text-gray-600 mt-2">{template.description}</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Dependency Configuration Modal */}
      {showDependencyConfig && selectedEdgeData && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Configure Dependency</h2>
              <button
                onClick={() => setShowDependencyConfig(false)}
                className="p-2 hover:bg-gray-100 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Dependency Type</label>
                <div className="space-y-2">
                  {dependencyTypes.map((type) => (
                    <label key={type.value} className="flex items-center gap-3 p-3 border rounded cursor-pointer hover:bg-gray-50">
                      <input
                        type="radio"
                        name="dependencyType"
                        value={type.value}
                        checked={selectedEdgeData.data?.dependencyType === type.value}
                        onChange={(e) => updateEdgeDependency(selectedEdge!, e.target.value)}
                        className="text-blue-500"
                      />
                      <div className="flex-1">
                        <div className="font-medium">{type.label}</div>
                        <div className="text-sm text-gray-600">{type.description}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {selectedEdgeData.data?.dependencyType === 'conditional' && (
                <div>
                  <label className="block text-sm font-medium mb-2">Condition</label>
                  <input
                    type="text"
                    placeholder="e.g., context.get_state('success') == true"
                    className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    onChange={(e) => {
                      const config = { ...selectedEdgeData.data?.config, condition: e.target.value };
                      updateEdgeDependency(selectedEdge!, selectedEdgeData.data?.dependencyType || 'required', config);
                    }}
                  />
                </div>
              )}

              {selectedEdgeData.data?.dependencyType === 'timeout' && (
                <div>
                  <label className="block text-sm font-medium mb-2">Timeout (seconds)</label>
                  <input
                    type="number"
                    placeholder="300"
                    className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    onChange={(e) => {
                      const config = { ...selectedEdgeData.data?.config, timeout: parseInt(e.target.value) };
                      updateEdgeDependency(selectedEdge!, selectedEdgeData.data?.dependencyType || 'required', config);
                    }}
                  />
                </div>
              )}
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={() => setShowDependencyConfig(false)}
                className="flex-1 px-4 py-2 border rounded hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => setShowDependencyConfig(false)}
                className="flex-1 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Node Configuration Modal */}
      {showNodeConfig && selectedNodeData && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Configure Node</h2>
              <button
                onClick={() => setShowNodeConfig(false)}
                className="p-2 hover:bg-gray-100 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Label</label>
                <input
                  type="text"
                  value={selectedNodeData.data.label}
                  onChange={(e) => updateNodeConfig(selectedNode!, { label: e.target.value })}
                  className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Description</label>
                <textarea
                  value={selectedNodeData.data.description || ''}
                  onChange={(e) => updateNodeConfig(selectedNode!, { description: e.target.value })}
                  className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  rows={3}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Type</label>
                <select
                  value={selectedNodeData.data.type}
                  onChange={(e) => updateNodeConfig(selectedNode!, { type: e.target.value })}
                  className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  {nodeTemplates.map((template) => (
                    <option key={template.type} value={template.type}>
                      {template.label} ({template.type})
                    </option>
                  ))}
                </select>
              </div>

              {/* Dynamic configuration fields based on node type */}
              {selectedNodeData.data.type === 'builtin.delay' && (
                <div>
                  <label className="block text-sm font-medium mb-2">Delay (seconds)</label>
                  <input
                    type="number"
                    placeholder="5"
                    value={selectedNodeData.data.config?.seconds || ''}
                    onChange={(e) => {
                      const config = { ...selectedNodeData.data.config, seconds: parseInt(e.target.value) };
                      updateNodeConfig(selectedNode!, { config });
                    }}
                    className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              )}

              {selectedNodeData.data.type === 'builtin.conditional' && (
                <div>
                  <label className="block text-sm font-medium mb-2">Condition</label>
                  <input
                    type="text"
                    placeholder="len(context.get_state('data', [])) > 0"
                    value={selectedNodeData.data.config?.condition || ''}
                    onChange={(e) => {
                      const config = { ...selectedNodeData.data.config, condition: e.target.value };
                      updateNodeConfig(selectedNode!, { config });
                    }}
                    className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              )}

              {/* Add more node-specific configurations as needed */}
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={() => setShowNodeConfig(false)}
                className="flex-1 px-4 py-2 border rounded hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => setShowNodeConfig(false)}
                className="flex-1 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const VisualWorkflowDesigner = (props: VisualWorkflowDesignerProps) => {
  return (
    <ReactFlowProvider>
      <VisualWorkflowDesignerContent {...props} />
    </ReactFlowProvider>
  );
};

export default VisualWorkflowDesigner;