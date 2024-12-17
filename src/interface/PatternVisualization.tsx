import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Play, Pause, RefreshCw } from 'lucide-react';

const PatternVisualization = () => {
  const [activePattern, setActivePattern] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [agentPositions, setAgentPositions] = useState([]);
  const [metrics, setMetrics] = useState([]);
  
  const patterns = [
    { id: 'circle', name: 'Circular Pattern', 
      description: 'Agents organize in a circular formation with emergent rotational behavior' },
    { id: 'grid', name: 'Grid Pattern', 
      description: 'Agents self-organize into a stable grid structure' },
    { id: 'random_clusters', name: 'Random Clusters', 
      description: 'Agents form dynamic, emergent cluster patterns' }
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Pattern Formation Experiments</h2>
        <div className="flex gap-2">
          <button 
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center gap-2 px-4 py-2 text-white rounded
              ${isRunning ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'}`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? 'Stop' : 'Start'}
          </button>
          <button 
            onClick={() => {
              setActivePattern(null);
              setAgentPositions([]);
              setMetrics([]);
            }}
            className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            <RefreshCw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {patterns.map((pattern) => (
          <div 
            key={pattern.id}
            className={`p-4 border rounded-lg cursor-pointer transition-all
              ${activePattern === pattern.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-300'}`}
            onClick={() => setActivePattern(pattern.id)}
          >
            <h3 className="font-semibold text-lg mb-2">{pattern.name}</h3>
            <p className="text-sm text-gray-600">{pattern.description}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="border rounded-lg p-4 bg-white">
          <h3 className="text-lg font-semibold mb-4">Agent Positions</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={agentPositions}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="x" stroke="#8884d8" />
                <Line type="monotone" dataKey="y" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="border rounded-lg p-4 bg-white">
          <h3 className="text-lg font-semibold mb-4">Pattern Metrics</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="cohesion" stroke="#8884d8" />
                <Line type="monotone" dataKey="alignment" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="border rounded-lg p-4 bg-gray-50">
        <h3 className="font-semibold mb-2">Experiment Status</h3>
        <div className="flex items-center gap-4">
          <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-green-500' : 'bg-gray-400'}`} />
          <span>{isRunning ? 'Running' : 'Stopped'}</span>
          {activePattern && (
            <span className="text-gray-600">
              Current Pattern: {patterns.find(p => p.id === activePattern)?.name}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default PatternVisualization;