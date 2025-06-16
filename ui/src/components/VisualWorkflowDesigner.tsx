import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
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
  Play,
  CheckCircle,
  MessageCircle,
  Globe,
  MessageSquare,
  Layers,
  Upload,
  PlusCircle,
  Server,
  Send,
  Search,
  Inbox,
  MailOpen,
  Box,
  PlayCircle
} from 'lucide-react';

import { workflowApi } from '../api/client';
import type { NodeAppearance } from '../types/workflow';

interface VisualWorkflowDesignerProps {
  initialYaml?: string;
  onChange?: (yaml: string) => void;
  onSave?: (yaml: string) => void;
}

const CustomEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  data,
  markerEnd,
}: EdgeProps) => {
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
    console.log('Edge clicked:', id);
  };

  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />
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
          {data?.dependencyType && (
            <button
              className="edge-button"
              onClick={onEdgeClick}
            >
              {getDependencyIcon(data.dependencyType)}
            </button>
          )}
        </div>
      </EdgeLabelRenderer>
    </>
  );
};

const getDependencyIcon = (type: string) => {
  switch (type) {
    case 'required': return '🔗';
    case 'optional': return '⚪';
    case 'conditional': return '🔀';
    default: return '→';
  }
};

const iconMap: Record<string, any> = {
  'play-circle': PlayCircle,
  'check-circle': CheckCircle,
  'git-branch': GitBranch,
  'zap': Zap,
  'clock': Clock,
  'alert-triangle': AlertTriangle,
  'mail': Mail,
  'mail-open': MailOpen,
  'search': Search,
  'message-circle': MessageCircle,
  'inbox': Inbox,
  'upload': Upload,
  'plus-circle': PlusCircle,
  'message-square': MessageSquare,
  'layers': Layers,
  'code': Code,
  'globe': Globe,
  'server': Server,
  'send': Send,
  'box': Box,
  'database': Database,
  'eye': Eye,
};

const getNodeIcon = (type: string, appearance?: NodeAppearance) => {
  if (!appearance) {
    return iconMap['box'] || Box;
  }
  return iconMap[appearance.icon] || Box;
};

const getNodeColor = (type: string, appearance?: NodeAppearance) => {
  if (!appearance) {
    return '#6b7280';
  }
  return appearance.color;
};

const VisualWorkflowDesignerInner: React.FC<VisualWorkflowDesignerProps> = ({
  initialYaml,
  onChange,
  onSave
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [availableNodes, setAvailableNodes] = useState<Record<string, NodeAppearance>>({});
  const [nodeCategories, setNodeCategories] = useState<Record<string, string[]>>({});
  const [isAddingNode, setIsAddingNode] = useState(false);

  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { project } = useReactFlow();

  useEffect(() => {
    loadAvailableNodes();
  }, []);

  useEffect(() => {
    if (initialYaml) {
      const { nodes: parsedNodes, edges: parsedEdges } = parseYamlToNodes(initialYaml);
      setNodes(parsedNodes);
      setEdges(parsedEdges);
    }
  }, [initialYaml, setNodes, setEdges]);

  const loadAvailableNodes = async () => {
    try {
      const nodes = await workflowApi.getAvailableNodes();
      setAvailableNodes(nodes);
      
      // Group by category
      const categories: Record<string, string[]> = {};
      Object.entries(nodes).forEach(([nodeType, appearance]) => {
        const category = appearance.category || 'other';
        if (!categories[category]) categories[category] = [];
        categories[category].push(nodeType);
      });
      setNodeCategories(categories);
    } catch (error) {
      console.error('Failed to load available nodes:', error);
    }
  };

  const parseYamlToNodes = (yamlContent: string) => {
    try {
      // Simple YAML parsing for states
      const lines = yamlContent.split('\n');
      const states: any[] = [];
      let currentState: any = null;
      let inStates = false;

      for (const line of lines) {
        const trimmed = line.trim();
        
        if (trimmed === 'states:') {
          inStates = true;
          continue;
        }
        
        if (!inStates) continue;
        
        if (trimmed.startsWith('- name:')) {
          if (currentState) {
            states.push(currentState);
          }
          currentState = {
            name: trimmed.replace('- name:', '').trim(),
            type: 'builtin.transform',
            dependencies: [],
            transitions: []
          };
        } else if (trimmed.startsWith('type:') && currentState) {
          currentState.type = trimmed.replace('type:', '').trim();
        }
      }
      
      if (currentState) {
        states.push(currentState);
      }

      // Convert to nodes and edges
      const nodes: Node[] = states.map((state, index) => {
        const appearance = availableNodes[state.type];
        const IconComponent = getNodeIcon(state.type, appearance);
        
        return {
          id: state.name,
          type: 'default',
          position: { x: 200 + (index % 3) * 250, y: 100 + Math.floor(index / 3) * 150 },
          data: {
            label: (
              <div className="flex items-center space-x-2">
                <IconComponent 
                  size={16} 
                  style={{ color: getNodeColor(state.type, appearance) }}
                />
                <span>{state.name}</span>
              </div>
            ),
            type: state.type,
            config: state.config || {},
            description: appearance?.description || ''
          },
          style: {
            background: '#fff',
            border: `2px solid ${getNodeColor(state.type, appearance)}`,
            borderRadius: '8px',
            padding: '10px',
            minWidth: '150px'
          }
        };
      });

      // Create edges based on simple sequential flow
      const edges: Edge[] = [];
      for (let i = 0; i < states.length - 1; i++) {
        edges.push({
          id: `${states[i].name}-${states[i + 1].name}`,
          source: states[i].name,
          target: states[i + 1].name,
          type: 'default',
          markerEnd: {
            type: MarkerType.ArrowClosed,
          },
        });
      }

      return { nodes, edges };
    } catch (error) {
      console.error('Failed to parse YAML:', error);
      return { nodes: [], edges: [] };
    }
  };

  const generateYAML = () => {
    try {
      const workflowName = 'visual_workflow';
      const states = nodes.map(node => {
        const stateData = node.data;
        return {
          name: node.id,
          type: stateData.type || 'builtin.transform',
          description: stateData.description || '',
          config: stateData.config || {},
        };
      });

      const yaml = `name: ${workflowName}
version: "1.0.0"
description: "Workflow created with visual designer"
author: "Visual Designer"

config:
  timeout: 300
  max_concurrent: 5

environment:
  variables:
    LOG_LEVEL: INFO
  secrets: []

states:
${states.map(state => `  - name: ${state.name}
    type: ${state.type}
    description: "${state.description}"${Object.keys(state.config).length > 0 ? `
    config:
${Object.entries(state.config).map(([key, value]) => `      ${key}: ${typeof value === 'string' ? `"${value}"` : value}`).join('\n')}` : ''}`).join('\n')}`;

      onChange?.(yaml);
      return yaml;
    } catch (error) {
      console.error('Failed to generate YAML:', error);
      return '';
    }
  };

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    setIsAddingNode(false);
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      const nodeType = event.dataTransfer.getData('application/reactflow');

      if (!nodeType || !reactFlowBounds) {
        return;
      }

      const position = project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const appearance = availableNodes[nodeType];
      const IconComponent = getNodeIcon(nodeType, appearance);
      const newNodeId = `${nodeType.split('.')[1] || 'node'}_${Date.now()}`;

      const newNode: Node = {
        id: newNodeId,
        type: 'default',
        position,
        data: {
          label: (
            <div className="flex items-center space-x-2">
              <IconComponent 
                size={16} 
                style={{ color: getNodeColor(nodeType, appearance) }}
              />
              <span>{newNodeId}</span>
            </div>
          ),
          type: nodeType,
          config: {},
          description: appearance?.description || ''
        },
        style: {
          background: '#fff',
          border: `2px solid ${getNodeColor(nodeType, appearance)}`,
          borderRadius: '8px',
          padding: '10px',
          minWidth: '150px'
        }
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [project, availableNodes, setNodes]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const deleteSelectedNode = useCallback(() => {
    if (selectedNode) {
      setNodes((nds) => nds.filter((node) => node.id !== selectedNode.id));
      setEdges((eds) => eds.filter((edge) => 
        edge.source !== selectedNode.id && edge.target !== selectedNode.id
      ));
      setSelectedNode(null);
    }
  }, [selectedNode, setNodes, setEdges]);

  const updateNodeConfig = useCallback((nodeId: string, config: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, config } }
          : node
      )
    );
  }, [setNodes]);

  return (
    <div className="h-full flex">
      {/* Node Palette */}
      <div className="w-64 bg-gray-50 border-r border-gray-200 p-4 overflow-y-auto">
        <h3 className="text-lg font-semibold mb-4">Available Nodes</h3>
        
        {Object.entries(nodeCategories).map(([category, nodeTypes]) => (
          <div key={category} className="mb-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2 capitalize">
              {category}
            </h4>
            <div className="space-y-2">
              {nodeTypes.map((nodeType) => {
                const appearance = availableNodes[nodeType];
                const IconComponent = getNodeIcon(nodeType, appearance);
                
                return (
                  <div
                    key={nodeType}
                    className="flex items-center p-2 bg-white border border-gray-200 rounded cursor-grab hover:shadow-sm"
                    draggable
                    onDragStart={(event) => {
                      event.dataTransfer.setData('application/reactflow', nodeType);
                      event.dataTransfer.effectAllowed = 'move';
                    }}
                  >
                    <IconComponent 
                      size={16} 
                      className="mr-2"
                      style={{ color: getNodeColor(nodeType, appearance) }}
                    />
                    <div>
                      <div className="text-sm font-medium">
                        {nodeType.split('.')[1] || nodeType}
                      </div>
                      <div className="text-xs text-gray-500 truncate">
                        {appearance?.description || ''}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Canvas */}
      <div className="flex-1 relative">
        <div ref={reactFlowWrapper} className="h-full">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            onDrop={onDrop}
            onDragOver={onDragOver}
            connectionLineType={ConnectionLineType.Bezier}
            edgeTypes={{ default: CustomEdge }}
            fitView
          >
            <Controls />
            <MiniMap />
            <Background variant="dots" gap={12} size={1} />
            
            <Panel position="top-right">
              <div className="flex space-x-2">
                <button
                  onClick={() => onChange?.(generateYAML())}
                  className="px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  <Eye size={16} />
                </button>
                <button
                  onClick={() => onSave?.(generateYAML())}
                  className="px-3 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                >
                  <Check size={16} />
                </button>
                {selectedNode && (
                  <button
                    onClick={deleteSelectedNode}
                    className="px-3 py-2 bg-red-600 text-white rounded hover:bg-red-700"
                  >
                    <Trash2 size={16} />
                  </button>
                )}
              </div>
            </Panel>
          </ReactFlow>
        </div>
      </div>

      {/* Properties Panel */}
      {selectedNode && (
        <div className="w-80 bg-gray-50 border-l border-gray-200 p-4 overflow-y-auto">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Properties</h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              <X size={16} />
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Node ID
              </label>
              <input
                type="text"
                value={selectedNode.id}
                disabled
                className="w-full px-3 py-2 border border-gray-300 rounded bg-gray-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Type
              </label>
              <input
                type="text"
                value={selectedNode.data.type}
                disabled
                className="w-full px-3 py-2 border border-gray-300 rounded bg-gray-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={selectedNode.data.description || ''}
                onChange={(e) => {
                  const updatedNode = {
                    ...selectedNode,
                    data: { ...selectedNode.data, description: e.target.value }
                  };
                  setSelectedNode(updatedNode);
                  setNodes((nds) =>
                    nds.map((node) =>
                      node.id === selectedNode.id ? updatedNode : node
                    )
                  );
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded"
                rows={3}
              />
            </div>

            {/* Dynamic configuration fields based on node type */}
            {availableNodes[selectedNode.data.type]?.inputs && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Configuration</h4>
                {Object.entries(availableNodes[selectedNode.data.type].inputs).map(([key, input]) => (
                  <div key={key} className="mb-3">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      {key}
                      {input.required && <span className="text-red-500 ml-1">*</span>}
                    </label>
                    <input
                      type={input.type === 'number' ? 'number' : 'text'}
                      value={selectedNode.data.config[key] || input.default || ''}
                      onChange={(e) => {
                        const newConfig = {
                          ...selectedNode.data.config,
                          [key]: input.type === 'number' ? parseFloat(e.target.value) : e.target.value
                        };
                        updateNodeConfig(selectedNode.id, newConfig);
                      }}
                      className="w-full px-3 py-2 border border-gray-300 rounded"
                      placeholder={input.description}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const VisualWorkflowDesigner: React.FC<VisualWorkflowDesignerProps> = (props) => {
  return (
    <ReactFlowProvider>
      <VisualWorkflowDesignerInner {...props} />
    </ReactFlowProvider>
  );
};

export default VisualWorkflowDesigner;