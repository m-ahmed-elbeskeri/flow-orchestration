// ui/src/utils/yamlUtils.ts

export interface WorkflowState {
  name: string;
  type: string;
  description?: string;
  config?: Record<string, any>;
  dependencies?: Array<{
    name: string;
    type: string;
    condition?: string;
    timeout?: number;
  }>;
  transitions?: Array<{
    on_success?: string;
    on_failure?: string;
    on_true?: string;
    on_false?: string;
    condition?: string;
  }>;
  resources?: {
    cpu_units?: number;
    memory_mb?: number;
    network_weight?: number;
    priority?: number;
    timeout?: number;
  };
}

export interface WorkflowDefinition {
  name: string;
  version: string;
  description: string;
  author: string;
  config?: {
    timeout?: number;
    max_concurrent?: number;
    retry_policy?: Record<string, any>;
  };
  environment?: {
    variables?: Record<string, string>;
    secrets?: string[];
  };
  states: WorkflowState[];
  schedule?: {
    cron?: string;
    timezone?: string;
  };
}

export interface VisualNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: {
    label: string;
    type: string;
    description?: string;
    config?: Record<string, any>;
  };
}

export interface VisualEdge {
  id: string;
  source: string;
  target: string;
  data?: {
    dependencyType?: string;
    config?: Record<string, any>;
  };
}

/**
 * Simple YAML parser for workflow definitions
 * Note: This is a basic implementation. For production use, consider using a proper YAML library
 */
export const parseWorkflowYaml = (yamlContent: string): WorkflowDefinition | null => {
  try {
    const lines = yamlContent.split('\n');
    const workflow: Partial<WorkflowDefinition> = {};
    
    // Parse basic properties
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('name:')) {
        workflow.name = trimmed.split(':')[1].trim().replace(/["']/g, '');
      } else if (trimmed.startsWith('version:')) {
        workflow.version = trimmed.split(':')[1].trim().replace(/["']/g, '');
      } else if (trimmed.startsWith('description:')) {
        workflow.description = trimmed.split(':')[1].trim().replace(/["']/g, '');
      } else if (trimmed.startsWith('author:')) {
        workflow.author = trimmed.split(':')[1].trim().replace(/["']/g, '');
      }
    }

    // Parse states (simplified)
    const states: WorkflowState[] = [];
    let inStatesSection = false;
    let currentState: Partial<WorkflowState> = {};
    let currentIndent = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();
      const indent = line.length - line.trimLeft().length;

      if (trimmed === 'states:') {
        inStatesSection = true;
        continue;
      }

      if (inStatesSection) {
        if (trimmed.startsWith('- name:')) {
          // Save previous state if exists
          if (currentState.name) {
            states.push(currentState as WorkflowState);
          }
          // Start new state
          currentState = {
            name: trimmed.split(':')[1].trim(),
          };
          currentIndent = indent;
        } else if (indent > currentIndent && trimmed.startsWith('type:')) {
          currentState.type = trimmed.split(':')[1].trim();
        } else if (indent > currentIndent && trimmed.startsWith('description:')) {
          currentState.description = trimmed.split(':')[1].trim().replace(/["']/g, '');
        }
        // Add more parsing for config, dependencies, transitions as needed
      }
    }

    // Save last state
    if (currentState.name) {
      states.push(currentState as WorkflowState);
    }

    if (!workflow.name || !workflow.version || states.length === 0) {
      return null;
    }

    return {
      name: workflow.name,
      version: workflow.version || '1.0.0',
      description: workflow.description || '',
      author: workflow.author || '',
      states,
    } as WorkflowDefinition;
  } catch (error) {
    console.error('Failed to parse YAML:', error);
    return null;
  }
};

/**
 * Convert workflow definition to visual nodes and edges
 */
export const workflowToVisual = (workflow: WorkflowDefinition): { nodes: VisualNode[]; edges: VisualEdge[] } => {
  const nodes: VisualNode[] = [];
  const edges: VisualEdge[] = [];

  // Create nodes
  workflow.states.forEach((state, index) => {
    const node: VisualNode = {
      id: `state-${state.name}`,
      type: 'workflowNode',
      position: { 
        x: 100 + (index % 3) * 300, 
        y: 100 + Math.floor(index / 3) * 200 
      },
      data: {
        label: state.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        type: state.type,
        description: state.description,
        config: state.config || {},
      },
    };
    nodes.push(node);
  });

  // Create edges from transitions
  workflow.states.forEach((state) => {
    if (state.transitions) {
      state.transitions.forEach((transition) => {
        const targetStateName = transition.on_success || transition.on_true || transition.on_false;
        if (targetStateName) {
          const edge: VisualEdge = {
            id: `edge-${state.name}-${targetStateName}`,
            source: `state-${state.name}`,
            target: `state-${targetStateName}`,
            data: {
              dependencyType: transition.condition ? 'conditional' : 'required',
              config: transition.condition ? { condition: transition.condition } : {},
            },
          };
          edges.push(edge);
        }
      });
    }
  });

  // Create edges from dependencies
  workflow.states.forEach((state) => {
    if (state.dependencies) {
      state.dependencies.forEach((dependency) => {
        const edgeId = `edge-${dependency.name}-${state.name}`;
        // Check if edge already exists from transitions
        if (!edges.find(e => e.id === edgeId)) {
          const edge: VisualEdge = {
            id: edgeId,
            source: `state-${dependency.name}`,
            target: `state-${state.name}`,
            data: {
              dependencyType: dependency.type,
              config: {
                condition: dependency.condition,
                timeout: dependency.timeout,
              },
            },
          };
          edges.push(edge);
        }
      });
    }
  });

  return { nodes, edges };
};

/**
 * Convert visual nodes and edges back to workflow definition
 */
export const visualToWorkflow = (
  nodes: VisualNode[], 
  edges: VisualEdge[], 
  metadata: { name: string; version: string; description: string; author: string }
): WorkflowDefinition => {
  const states: WorkflowState[] = nodes.map((node) => {
    const state: WorkflowState = {
      name: node.id.replace('state-', ''),
      type: node.data.type,
      description: node.data.description,
      config: node.data.config,
    };

    // Find transitions (outgoing edges)
    const outgoingEdges = edges.filter(edge => edge.source === node.id);
    if (outgoingEdges.length > 0) {
      state.transitions = outgoingEdges.map(edge => {
        const targetStateName = edge.target.replace('state-', '');
        const transition: any = {};
        
        if (edge.data?.dependencyType === 'conditional') {
          transition.on_true = targetStateName;
          transition.condition = edge.data.config?.condition;
        } else {
          transition.on_success = targetStateName;
        }
        
        return transition;
      });
    }

    // Find dependencies (incoming edges)
    const incomingEdges = edges.filter(edge => edge.target === node.id);
    if (incomingEdges.length > 0) {
      state.dependencies = incomingEdges.map(edge => ({
        name: edge.source.replace('state-', ''),
        type: edge.data?.dependencyType || 'required',
        condition: edge.data?.config?.condition,
        timeout: edge.data?.config?.timeout,
      }));
    }

    return state;
  });

  return {
    name: metadata.name,
    version: metadata.version,
    description: metadata.description,
    author: metadata.author,
    config: {
      timeout: 300,
      max_concurrent: 5,
    },
    environment: {
      variables: {
        LOG_LEVEL: 'INFO',
      },
      secrets: [],
    },
    states,
  };
};

/**
 * Generate YAML string from workflow definition
 */
export const workflowToYaml = (workflow: WorkflowDefinition): string => {
  const lines = [
    `name: ${workflow.name}`,
    `version: "${workflow.version}"`,
    `description: "${workflow.description}"`,
    `author: "${workflow.author}"`,
    '',
    'config:',
    `  timeout: ${workflow.config?.timeout || 300}`,
    `  max_concurrent: ${workflow.config?.max_concurrent || 5}`,
    '',
    'environment:',
    '  variables:',
  ];

  // Add environment variables
  if (workflow.environment?.variables) {
    Object.entries(workflow.environment.variables).forEach(([key, value]) => {
      lines.push(`    ${key}: ${value}`);
    });
  } else {
    lines.push('    LOG_LEVEL: INFO');
  }

  lines.push('  secrets: []');
  lines.push('');
  lines.push('states:');

  // Add states
  workflow.states.forEach((state) => {
    lines.push(`  - name: ${state.name}`);
    lines.push(`    type: ${state.type}`);
    if (state.description) {
      lines.push(`    description: "${state.description}"`);
    }

    // Add config
    if (state.config && Object.keys(state.config).length > 0) {
      lines.push('    config:');
      Object.entries(state.config).forEach(([key, value]) => {
        lines.push(`      ${key}: ${JSON.stringify(value)}`);
      });
    }

    // Add dependencies
    if (state.dependencies && state.dependencies.length > 0) {
      lines.push('    dependencies:');
      state.dependencies.forEach((dep) => {
        lines.push(`      - name: ${dep.name}`);
        lines.push(`        type: ${dep.type}`);
        if (dep.condition) {
          lines.push(`        condition: "${dep.condition}"`);
        }
        if (dep.timeout) {
          lines.push(`        timeout: ${dep.timeout}`);
        }
      });
    }

    // Add transitions
    if (state.transitions && state.transitions.length > 0) {
      lines.push('    transitions:');
      state.transitions.forEach((trans) => {
        if (trans.on_success) {
          lines.push(`      - on_success: ${trans.on_success}`);
        }
        if (trans.on_failure) {
          lines.push(`      - on_failure: ${trans.on_failure}`);
        }
        if (trans.on_true) {
          lines.push(`      - on_true: ${trans.on_true}`);
        }
        if (trans.on_false) {
          lines.push(`      - on_false: ${trans.on_false}`);
        }
      });
    }

    lines.push('');
  });

  return lines.join('\n');
};