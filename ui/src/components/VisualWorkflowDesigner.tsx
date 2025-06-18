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
  BackgroundVariant,
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
  PlayCircle,
  Edit,
  ArrowRight,
  CircleDot,
  Workflow,
  Link,
  MoreHorizontal,
  Activity,
  RefreshCw,
  AlertCircle,
  Package
} from 'lucide-react';

import { workflowApi } from '../api/client';
import type { NodeAppearance } from '../types/workflow';

interface VisualWorkflowDesignerProps {
  initialYaml?: string;
  onChange?: (yaml: string) => void;
  onSave?: (yaml: string) => void;
}

// Enhanced edge types
type EdgeType = 'required' | 'optional' | 'conditional' | 'parallel';

interface EdgeData {
  label?: string;
  edgeType: EdgeType;
  condition?: string;
  config?: Record<string, any>;
}

// Enhanced icon mapping - moved outside component
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
  'workflow': Workflow,
  'activity': Activity,
  'package': Package,
};

const getNodeIcon = (type: string, appearance?: NodeAppearance) => {
  if (!appearance?.icon) return iconMap['box'] || Box;
  return iconMap[appearance.icon] || iconMap['box'] || Box;
};

const getNodeColor = (type: string, appearance?: NodeAppearance) => {
  if (!appearance?.color) return '#00d4ff';
  return appearance.color;
};

// Custom dark styled edge component - moved outside and memoized
const CustomEdge = React.memo(({
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
  selected,
}: EdgeProps & { selected?: boolean }) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const edgeType = (data as EdgeData)?.edgeType || 'required';
  
  const getEdgeStyle = () => {
    const baseStyle = {
      strokeWidth: selected ? 3 : 2,
      filter: selected ? 'drop-shadow(0 0 8px rgba(0, 212, 255, 0.6))' : undefined,
      transition: 'all 0.3s ease',
    };

    switch (edgeType) {
      case 'required':
        return { ...baseStyle, stroke: '#00d4ff', strokeDasharray: 'none' };
      case 'optional':
        return { ...baseStyle, stroke: '#10b981', strokeDasharray: '8,4' };
      case 'conditional':
        return { ...baseStyle, stroke: '#f59e0b', strokeDasharray: '12,4,4,4' };
      case 'parallel':
        return { ...baseStyle, stroke: '#8b5cf6', strokeDasharray: 'none', strokeWidth: selected ? 4 : 3 };
      default:
        return baseStyle;
    }
  };

  const getEdgeIcon = () => {
    switch (edgeType) {
      case 'required': return '→';
      case 'optional': return '⚬';
      case 'conditional': return '◊';
      case 'parallel': return '∥';
      default: return '→';
    }
  };

  const getEdgeColor = () => {
    switch (edgeType) {
      case 'required': return '#00d4ff';
      case 'optional': return '#10b981';
      case 'conditional': return '#f59e0b';
      case 'parallel': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  return (
    <>
      <BaseEdge 
        path={edgePath} 
        markerEnd={markerEnd} 
        style={{ ...style, ...getEdgeStyle() }} 
      />
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: 'all',
          }}
          className="nodrag nopan"
        >
          <div
            className={`
              px-3 py-1.5 rounded-full text-xs font-medium cursor-pointer
              transition-all duration-200 hover:scale-110 shadow-lg
              ${selected 
                ? 'bg-gray-800 border-2 shadow-xl text-white' 
                : 'bg-gray-800/90 border hover:bg-gray-700 hover:shadow-xl text-gray-200'
              }
            `}
            style={{ 
              borderColor: getEdgeColor(),
              backdropFilter: 'blur(8px)',
            }}
          >
            <span className="mr-1">{getEdgeIcon()}</span>
            <span className="text-xs">{edgeType}</span>
          </div>
        </div>
      </EdgeLabelRenderer>
    </>
  );
});

// Enhanced Dark-themed Node Component - moved outside and memoized
const CustomNode = React.memo(({ data, selected }: any) => {
  const appearance = data.appearance;
  const IconComponent = getNodeIcon(data.type, appearance);
  const color = getNodeColor(data.type, appearance);

  return (
    <div 
      className={`
        relative px-4 py-3 rounded-xl border-2 shadow-xl min-w-[200px]
        transition-all duration-300 hover:shadow-2xl backdrop-blur-sm
        ${selected ? 'ring-2 ring-blue-400 ring-opacity-60 scale-105' : 'hover:scale-102'}
      `}
      style={{ 
        borderColor: selected ? '#00d4ff' : '#444',
        background: selected 
          ? `linear-gradient(135deg, ${color}15 0%, rgba(45, 45, 45, 0.95) 100%)`
          : `linear-gradient(135deg, ${color}08 0%, rgba(45, 45, 45, 0.9) 100%)`,
        backdropFilter: 'blur(12px)',
        boxShadow: selected 
          ? `0 0 20px rgba(0, 212, 255, 0.3), 0 10px 25px rgba(0, 0, 0, 0.3)`
          : '0 8px 25px rgba(0, 0, 0, 0.2)',
      }}
    >
      <div className="flex items-center space-x-3">
        <div 
          className="flex items-center justify-center w-10 h-10 rounded-lg transition-all duration-300"
          style={{ 
            backgroundColor: `${color}20`,
            boxShadow: `0 0 10px ${color}30`,
          }}
        >
          <IconComponent 
            size={20} 
            style={{ color: color }}
          />
        </div>
        <div className="flex-1">
          <div className="font-semibold text-white text-sm">
            {data.label || data.id}
          </div>
          {data.description && (
            <div className="text-xs text-gray-300 mt-1 truncate max-w-[140px]">
              {data.description}
            </div>
          )}
        </div>
      </div>
      
      {/* Status indicator with glow */}
      <div 
        className="absolute -top-1 -right-1 w-3 h-3 rounded-full border-2 border-gray-700"
        style={{ 
          backgroundColor: color,
          boxShadow: `0 0 8px ${color}80`,
        }}
      />

      {/* Connection points */}
      <div className="absolute -top-1.5 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-gray-600 rounded-full border-2 border-gray-700 hover:bg-blue-400 hover:border-blue-400 transition-all duration-200 hover:scale-125 hover:shadow-lg" />
      <div className="absolute -bottom-1.5 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-gray-600 rounded-full border-2 border-gray-700 hover:bg-blue-400 hover:border-blue-400 transition-all duration-200 hover:scale-125 hover:shadow-lg" />
    </div>
  );
});

// AI Thinking Bubble Component
const ThinkingBubble = React.memo(({ message, position }: { message: string; position: { x: number; y: number } }) => {
  return (
    <div 
      className="absolute z-50 px-3 py-2 bg-blue-500/20 border border-blue-400/30 rounded-2xl text-blue-300 text-xs font-medium backdrop-blur-sm animate-pulse"
      style={{
        left: position.x + 220,
        top: position.y - 40,
        boxShadow: '0 0 20px rgba(0, 212, 255, 0.3)',
      }}
    >
      <div className="flex items-center space-x-2">
        <Activity size={12} />
        <span>{message}</span>
      </div>
    </div>
  );
});

// Empty State Component
const EmptyNodeLibrary = React.memo(({ onRetry, isLoading }: { onRetry: () => void; isLoading: boolean }) => {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center max-w-sm">
        <div className="w-20 h-20 mx-auto mb-6 bg-gray-700/50 rounded-full flex items-center justify-center backdrop-blur-sm">
          {isLoading ? (
            <RefreshCw className="w-10 h-10 text-gray-500 animate-spin" />
          ) : (
            <Package className="w-10 h-10 text-gray-500" />
          )}
        </div>
        <h3 className="text-lg font-semibold text-gray-300 mb-2">
          {isLoading ? 'Loading Integrations...' : 'No Integrations Available'}
        </h3>
        <p className="text-sm text-gray-400 mb-6">
          {isLoading 
            ? 'Fetching available plugin integrations from the server...'
            : 'Unable to load plugin integrations. Please check your connection and try again.'
          }
        </p>
        {!isLoading && (
          <button
            onClick={onRetry}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center space-x-2 mx-auto"
          >
            <RefreshCw size={16} />
            <span>Retry</span>
          </button>
        )}
      </div>
    </div>
  );
});

// Edge Type Editor Modal (Dark themed)
const EdgeTypeEditor = React.memo(({ 
  edge, 
  onUpdate, 
  onClose, 
  isOpen 
}: {
  edge: Edge | null;
  onUpdate: (edgeId: string, data: EdgeData) => void;
  onClose: () => void;
  isOpen: boolean;
}) => {
  const [edgeType, setEdgeType] = useState<EdgeType>('required');
  const [condition, setCondition] = useState('');

  useEffect(() => {
    if (edge?.data) {
      setEdgeType((edge.data as EdgeData).edgeType || 'required');
      setCondition((edge.data as EdgeData).condition || '');
    }
  }, [edge]);

  const handleSave = useCallback(() => {
    if (edge) {
      onUpdate(edge.id, {
        edgeType,
        condition: edgeType === 'conditional' ? condition : undefined,
        label: edgeType,
      });
    }
    onClose();
  }, [edge, edgeType, condition, onUpdate, onClose]);

  if (!isOpen || !edge) return null;

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="bg-gray-800 rounded-xl shadow-2xl p-6 w-96 max-w-90vw border border-gray-600">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Edit Connection</h3>
          <button 
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Connection Type
            </label>
            <div className="grid grid-cols-2 gap-2">
              {(['required', 'optional', 'conditional', 'parallel'] as EdgeType[]).map((type) => (
                <button
                  key={type}
                  onClick={() => setEdgeType(type)}
                  className={`
                    p-3 rounded-lg border-2 text-sm font-medium transition-all
                    ${edgeType === type 
                      ? 'border-blue-500 bg-blue-500/20 text-blue-300' 
                      : 'border-gray-600 hover:border-gray-500 text-gray-300 hover:bg-gray-700/50'
                    }
                  `}
                >
                  <div className="flex items-center space-x-2">
                    <span>{type === 'required' ? '→' : type === 'optional' ? '⚬' : type === 'conditional' ? '◊' : '∥'}</span>
                    <span className="capitalize">{type}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {edgeType === 'conditional' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Condition
              </label>
              <input
                type="text"
                value={condition}
                onChange={(e) => setCondition(e.target.value)}
                placeholder="e.g., success === true"
                className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-700 text-white"
              />
            </div>
          )}

          <div className="flex space-x-3 pt-4">
            <button
              onClick={handleSave}
              className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              Save Changes
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-400 hover:text-gray-200 transition-colors font-medium"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

const VisualWorkflowDesignerInner: React.FC<VisualWorkflowDesignerProps> = ({
  initialYaml,
  onChange,
  onSave
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<Edge | null>(null);
  const [availableNodes, setAvailableNodes] = useState<Record<string, NodeAppearance>>({});
  const [nodeCategories, setNodeCategories] = useState<Record<string, string[]>>({});
  const [showEdgeEditor, setShowEdgeEditor] = useState(false);
  const [thinkingBubble, setThinkingBubble] = useState<{ message: string; position: { x: number; y: number } } | null>(null);
  const [isLoadingNodes, setIsLoadingNodes] = useState(true);
  const [nodeLoadError, setNodeLoadError] = useState<string | null>(null);

  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { project } = useReactFlow();

  // Memoize nodeTypes and edgeTypes to prevent recreation on every render
  const edgeTypes = useMemo(() => ({
    default: CustomEdge,
  }), []);

  const nodeTypes = useMemo(() => ({
    default: CustomNode,
  }), []);

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

  useEffect(() => {
    if (nodes.length > 0 || edges.length > 0) {
      const yaml = generateYAML();
      if (yaml && onChange) {
        onChange(yaml);
      }
    }
  }, [nodes, edges, onChange]);

  const showThinkingBubbleAtPosition = useCallback((message: string, position: { x: number; y: number }) => {
    setThinkingBubble({ message, position });
    setTimeout(() => setThinkingBubble(null), 2000);
  }, []);

  const loadAvailableNodes = useCallback(async () => {
    setIsLoadingNodes(true);
    setNodeLoadError(null);
    
    try {
      console.log('🔍 Loading available plugin integrations...');
      const nodes = await workflowApi.getAvailableNodes();
      console.log('✅ Loaded plugin integrations:', nodes);
      
      if (!nodes || Object.keys(nodes).length === 0) {
        setNodeLoadError('No plugin integrations found. Please install and configure plugins first.');
        setAvailableNodes({});
        setNodeCategories({});
        return;
      }

      setAvailableNodes(nodes);
      
      // Group nodes by category
      const categories: Record<string, string[]> = {};
      Object.entries(nodes).forEach(([nodeType, appearance]) => {
        const category = appearance.category || 'other';
        if (!categories[category]) categories[category] = [];
        categories[category].push(nodeType);
      });
      setNodeCategories(categories);
      
      console.log('📂 Organized into categories:', categories);
      
    } catch (error: any) {
      console.error('❌ Failed to load plugin integrations:', error);
      setNodeLoadError(error.message || 'Failed to load plugin integrations. Please check your server connection.');
      setAvailableNodes({});
      setNodeCategories({});
    } finally {
      setIsLoadingNodes(false);
    }
  }, []);

  const parseYamlToNodes = useCallback((yamlContent: string) => {
    try {
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
        } else if (trimmed.startsWith('description:') && currentState) {
          currentState.description = trimmed.replace('description:', '').replace(/"/g, '').trim();
        }
      }
      
      if (currentState) {
        states.push(currentState);
      }

      const nodes: Node[] = states.map((state, index) => {
        const appearance = availableNodes[state.type];
        
        return {
          id: state.name,
          type: 'default',
          position: { x: 300 + (index % 3) * 300, y: 150 + Math.floor(index / 3) * 200 },
          data: {
            label: state.name,
            type: state.type,
            config: state.config || {},
            description: state.description || appearance?.description || '',
            appearance: appearance
          }
        };
      });

      const edges: Edge[] = [];
      for (let i = 0; i < states.length - 1; i++) {
        edges.push({
          id: `${states[i].name}-${states[i + 1].name}`,
          source: states[i].name,
          target: states[i + 1].name,
          type: 'default',
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: '#00d4ff',
          },
          data: {
            edgeType: 'required' as EdgeType,
            label: 'required'
          }
        });
      }

      return { nodes, edges };
    } catch (error) {
      console.error('Failed to parse YAML:', error);
      return { nodes: [], edges: [] };
    }
  }, [availableNodes]);

  const generateYAML = useCallback(() => {
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

      return yaml;
    } catch (error) {
      console.error('Failed to generate YAML:', error);
      return '';
    }
  }, [nodes]);

  const onConnect = useCallback(
    (params: Connection) => {
      const newEdge = {
        ...params,
        type: 'default',
        markerEnd: { type: MarkerType.ArrowClosed, color: '#00d4ff' },
        data: { edgeType: 'required' as EdgeType, label: 'required' }
      };
      setEdges((eds) => addEdge(newEdge, eds));
    },
    [setEdges]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
    setSelectedEdge(null);
  }, []);

  const onEdgeClick = useCallback((event: React.MouseEvent, edge: Edge) => {
    event.stopPropagation();
    setSelectedEdge(edge);
    setSelectedNode(null);
    setShowEdgeEditor(true);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    setSelectedEdge(null);
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

      // Show thinking bubble
      showThinkingBubbleAtPosition("🤖 AI adding node...", position);

      setTimeout(() => {
        const appearance = availableNodes[nodeType];
        const newNodeId = `${nodeType.split('.')[1] || 'node'}_${Date.now()}`;

        const newNode: Node = {
          id: newNodeId,
          type: 'default',
          position,
          data: {
            label: newNodeId,
            type: nodeType,
            config: {},
            description: appearance?.description || '',
            appearance: appearance
          }
        };

        setNodes((nds) => nds.concat(newNode));
      }, 500);
    },
    [project, availableNodes, setNodes, showThinkingBubbleAtPosition]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const deleteSelected = useCallback(() => {
    if (selectedNode) {
      setNodes((nds) => nds.filter((node) => node.id !== selectedNode.id));
      setEdges((eds) => eds.filter((edge) => 
        edge.source !== selectedNode.id && edge.target !== selectedNode.id
      ));
      setSelectedNode(null);
    }
    if (selectedEdge) {
      setEdges((eds) => eds.filter((edge) => edge.id !== selectedEdge.id));
      setSelectedEdge(null);
    }
  }, [selectedNode, selectedEdge, setNodes, setEdges]);

  const updateNodeConfig = useCallback((nodeId: string, config: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, config } }
          : node
      )
    );
  }, [setNodes]);

  const updateEdgeData = useCallback((edgeId: string, data: EdgeData) => {
    setEdges((eds) =>
      eds.map((edge) =>
        edge.id === edgeId
          ? { ...edge, data: { ...edge.data, ...data } }
          : edge
      )
    );
  }, [setEdges]);

  const hasAvailableNodes = Object.keys(availableNodes).length > 0;

  return (
    <div className="h-full flex" style={{ height: '100%', background: '#1a1a1a' }}>
      {/* Enhanced Dark Node Palette */}
      <div className="w-80 bg-gray-900/95 border-r border-gray-700 shadow-2xl overflow-y-auto backdrop-blur-sm">
        <div className="p-4 border-b border-gray-700 bg-gradient-to-r from-gray-800/50 to-gray-900/50 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-lg font-bold text-white flex items-center">
              <Workflow className="mr-3 text-blue-400" size={22} />
              Plugin Integrations
            </h3>
            <div className="flex items-center space-x-2">
              {hasAvailableNodes && (
                <div className="flex items-center space-x-2 bg-green-500/20 px-3 py-1 rounded-full border border-green-400/30">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-xs text-green-300 font-medium">{Object.keys(availableNodes).length} Available</span>
                </div>
              )}
              <button
                onClick={loadAvailableNodes}
                disabled={isLoadingNodes}
                className="p-1 text-gray-400 hover:text-blue-400 transition-colors"
                title="Refresh integrations"
              >
                <RefreshCw size={16} className={isLoadingNodes ? 'animate-spin' : ''} />
              </button>
            </div>
          </div>
          <p className="text-sm text-gray-400">
            {hasAvailableNodes 
              ? 'Drag plugin integrations to canvas to build your workflow'
              : 'No plugin integrations available'
            }
          </p>
        </div>
        
        {/* Show empty state or nodes */}
        {isLoadingNodes || !hasAvailableNodes ? (
          <EmptyNodeLibrary onRetry={loadAvailableNodes} isLoading={isLoadingNodes} />
        ) : (
          <div className="p-4 space-y-6">
            {Object.entries(nodeCategories).map(([category, nodeTypes]) => (
              <div key={category}>
                <h4 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider flex items-center">
                  <div className="w-1 h-4 bg-gradient-to-b from-blue-400 to-purple-500 rounded-full mr-3"></div>
                  {category}
                </h4>
                <div className="space-y-3">
                  {nodeTypes.map((nodeType) => {
                    const appearance = availableNodes[nodeType];
                    const IconComponent = getNodeIcon(nodeType, appearance);
                    const color = getNodeColor(nodeType, appearance);
                    
                    return (
                      <div
                        key={nodeType}
                        className="group flex items-center p-4 bg-gray-800/60 hover:bg-gray-700/80 border border-gray-600/50 hover:border-gray-500 rounded-xl cursor-grab active:cursor-grabbing transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/20 backdrop-blur-sm"
                        draggable
                        onDragStart={(event) => {
                          event.dataTransfer.setData('application/reactflow', nodeType);
                          event.dataTransfer.effectAllowed = 'move';
                        }}
                        style={{
                          background: `linear-gradient(135deg, ${color}08 0%, rgba(31, 41, 55, 0.6) 100%)`,
                        }}
                      >
                        <div 
                          className="flex items-center justify-center w-12 h-12 rounded-xl mr-4 group-hover:scale-110 transition-all duration-300"
                          style={{ 
                            backgroundColor: `${color}20`,
                            boxShadow: `0 0 15px ${color}30`,
                          }}
                        >
                          <IconComponent 
                            size={22} 
                            style={{ color: color }}
                          />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-semibold text-sm text-white group-hover:text-blue-300 transition-colors">
                            {nodeType.split('.').pop()?.replace('_', ' ') || nodeType}
                          </div>
                          <div className="text-xs text-gray-400 truncate mt-1">
                            {appearance?.description || 'No description available'}
                          </div>
                        </div>
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                          <Plus size={16} className="text-gray-400" />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Error message */}
        {nodeLoadError && !isLoadingNodes && (
          <div className="p-4 m-4 bg-red-500/20 border border-red-400/30 rounded-xl">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm text-red-300 font-medium">Error Loading Integrations</p>
                <p className="text-xs text-red-400 mt-1">{nodeLoadError}</p>
                <button
                  onClick={loadAvailableNodes}
                  className="mt-2 text-xs text-red-300 hover:text-red-200 underline"
                >
                  Try again
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Enhanced Dark Canvas */}
      <div className="flex-1 relative" style={{ height: '100%' }}>
        <div ref={reactFlowWrapper} className="h-full">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            onEdgeClick={onEdgeClick}
            onPaneClick={onPaneClick}
            onDrop={onDrop}
            onDragOver={onDragOver}
            connectionLineType={ConnectionLineType.Bezier}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            colorMode="dark"
            style={{ 
              height: '100%',
              background: '#1a1a1a',
            }}
          >
            <Controls 
              className="!bg-gray-800/90 !shadow-2xl !border !border-gray-600 !rounded-xl !backdrop-blur-sm"
              style={{
                background: 'rgba(31, 41, 55, 0.9)',
                backdropFilter: 'blur(12px)',
              }}
            />
            <MiniMap 
              className="!bg-gray-800/90 !shadow-2xl !border !border-gray-600 !rounded-xl !backdrop-blur-sm"
              nodeColor={(node) => node.data.appearance?.color || '#00d4ff'}
              maskColor="rgba(0, 0, 0, 0.6)"
              style={{
                background: 'rgba(31, 41, 55, 0.9)',
                backdropFilter: 'blur(12px)',
              }}
            />
            <Background 
              variant={BackgroundVariant.Dots} 
              gap={20} 
              size={1.5} 
              color="#333" 
              style={{ opacity: 0.4 }}
            />
            
            {/* Enhanced Dark Control Panel */}
            <Panel position="top-right" className="space-y-3">
              <div className="bg-gray-800/90 rounded-xl shadow-2xl border border-gray-600 p-3 flex space-x-3 backdrop-blur-sm">
                <button
                  onClick={() => onChange?.(generateYAML())}
                  className="px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-200 flex items-center space-x-2 text-sm font-medium shadow-lg"
                  title="Update YAML"
                >
                  <Eye size={16} />
                  <span>Preview</span>
                </button>
                <button
                  onClick={() => onSave?.(generateYAML())}
                  className="px-4 py-2 bg-gradient-to-r from-emerald-600 to-emerald-700 text-white rounded-lg hover:from-emerald-700 hover:to-emerald-800 transition-all duration-200 flex items-center space-x-2 text-sm font-medium shadow-lg"
                  title="Save Workflow"
                >
                  <Check size={16} />
                  <span>Save</span>
                </button>
                {(selectedNode || selectedEdge) && (
                  <button
                    onClick={deleteSelected}
                    className="px-4 py-2 bg-gradient-to-r from-red-600 to-red-700 text-white rounded-lg hover:from-red-700 hover:to-red-800 transition-all duration-200 flex items-center space-x-2 text-sm font-medium shadow-lg"
                    title="Delete Selected"
                  >
                    <Trash2 size={16} />
                    <span>Delete</span>
                  </button>
                )}
              </div>
            </Panel>

            {/* Empty Canvas State */}
            {!hasAvailableNodes && !isLoadingNodes && (
              <Panel position="center" className="pointer-events-none">
                <div className="text-center p-8 bg-gray-800/90 rounded-xl border border-gray-600 backdrop-blur-sm">
                  <Package className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-300 mb-2">No Plugin Integrations</h3>
                  <p className="text-sm text-gray-400 max-w-sm">
                    Install and configure plugin integrations to start building workflows visually.
                  </p>
                </div>
              </Panel>
            )}
          </ReactFlow>
        </div>

        {/* AI Thinking Bubble */}
        {thinkingBubble && (
          <ThinkingBubble 
            message={thinkingBubble.message} 
            position={thinkingBubble.position} 
          />
        )}
      </div>

      {/* Enhanced Dark Properties Panel */}
      {(selectedNode || selectedEdge) && (
        <div className="w-80 bg-gray-900/95 border-l border-gray-700 shadow-2xl overflow-y-auto backdrop-blur-sm">
          <div className="p-4 border-b border-gray-700 bg-gradient-to-r from-gray-800/50 to-gray-900/50">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-white flex items-center">
                <Settings className="mr-3 text-blue-400" size={20} />
                Properties
              </h3>
              <button
                onClick={() => {
                  setSelectedNode(null);
                  setSelectedEdge(null);
                }}
                className="text-gray-400 hover:text-white transition-colors p-1 rounded hover:bg-gray-700"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          <div className="p-4 space-y-6">
            {selectedNode && (
              <>
                <div>
                  <label className="block text-sm font-semibold text-gray-300 mb-2">
                    Node ID
                  </label>
                  <input
                    type="text"
                    value={selectedNode.id}
                    disabled
                    className="w-full px-3 py-2 border border-gray-600 rounded-lg bg-gray-700/50 text-gray-300 backdrop-blur-sm"
                  />
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-300 mb-2">
                    Type
                  </label>
                  <div className="flex items-center space-x-3 p-3 bg-gray-700/50 rounded-lg backdrop-blur-sm border border-gray-600">
                    {selectedNode.data.appearance && (
                      <div 
                        className="w-8 h-8 rounded-lg flex items-center justify-center"
                        style={{ 
                          backgroundColor: `${selectedNode.data.appearance.color}20`,
                          boxShadow: `0 0 10px ${selectedNode.data.appearance.color}30`,
                        }}
                      >
                        {React.createElement(
                          getNodeIcon(selectedNode.data.type, selectedNode.data.appearance), 
                          { size: 16, style: { color: selectedNode.data.appearance.color } }
                        )}
                      </div>
                    )}
                    <span className="font-medium text-white">{selectedNode.data.type}</span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-300 mb-2">
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
                    className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-gray-700/50 text-white backdrop-blur-sm"
                    rows={3}
                    placeholder="Enter node description..."
                  />
                </div>

                {/* Enhanced Configuration Section */}
                {availableNodes[selectedNode.data.type]?.inputs && Object.keys(availableNodes[selectedNode.data.type].inputs).length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
                      <Code size={16} className="mr-2 text-blue-400" />
                      Configuration
                    </h4>
                    <div className="space-y-3">
                      {Object.entries(availableNodes[selectedNode.data.type].inputs).map(([key, input]) => (
                        <div key={key}>
                          <label className="block text-sm font-medium text-gray-300 mb-1">
                            {key}
                            {input.required && <span className="text-red-400 ml-1">*</span>}
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
                            className="w-full px-3 py-2 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-700/50 text-white backdrop-blur-sm"
                            placeholder={input.description}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {selectedEdge && (
              <div>
                <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center">
                  <Link size={16} className="mr-2 text-blue-400" />
                  Connection Details
                </h4>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      From
                    </label>
                    <div className="p-3 bg-gray-700/50 rounded-lg text-sm text-white backdrop-blur-sm border border-gray-600">{selectedEdge.source}</div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      To
                    </label>
                    <div className="p-3 bg-gray-700/50 rounded-lg text-sm text-white backdrop-blur-sm border border-gray-600">{selectedEdge.target}</div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      Type
                    </label>
                    <div className="p-3 bg-gray-700/50 rounded-lg text-sm text-white capitalize backdrop-blur-sm border border-gray-600">
                      {(selectedEdge.data as EdgeData)?.edgeType || 'required'}
                    </div>
                  </div>
                  <button
                    onClick={() => setShowEdgeEditor(true)}
                    className="w-full px-4 py-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-200 flex items-center justify-center space-x-2 shadow-lg"
                  >
                    <Edit size={16} />
                    <span>Edit Connection</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Edge Type Editor Modal */}
      <EdgeTypeEditor
        edge={selectedEdge}
        onUpdate={updateEdgeData}
        onClose={() => setShowEdgeEditor(false)}
        isOpen={showEdgeEditor}
      />
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