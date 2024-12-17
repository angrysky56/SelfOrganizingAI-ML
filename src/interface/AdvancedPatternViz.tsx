import React, { useRef, useEffect, useState } from 'react';
import { Stage, Layer, Circle, Arrow } from 'react-konva';
import { Line } from 'react-chartjs-2';
import { Slider, Button } from '@/components/ui/slider';
import { Chart } from 'react-chartjs-2';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useHelper } from '@react-three/drei';

interface SimulationParams {
  numAgents: number;
  learningRate: number;
  interactionRadius: number;
  forceMagnitude: number;
  patternType: string;
}

const HeatMap = ({ data, width, height }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!ctx || !data) return;

    const imageData = ctx.createImageData(width, height);
    // Create heat map using agent density
    const densityMap = new Float32Array(width * height).fill(0);

    data.positions.forEach(pos => {
      const x = Math.floor((pos[0] + 1) * width / 2);
      const y = Math.floor((pos[1] + 1) * height / 2);
      if (x >= 0 && x < width && y >= 0 && y < height) {
        densityMap[y * width + x] += 1;
      }
    });

    // Convert density to colors
    for (let i = 0; i < densityMap.length; i++) {
      const value = Math.min(densityMap[i] * 50, 255); // Adjust multiplier for visibility
      const idx = i * 4;
      imageData.data[idx] = value;     // Red
      imageData.data[idx + 1] = 0;     // Green
      imageData.data[idx + 2] = 255 - value; // Blue
      imageData.data[idx + 3] = 255;   // Alpha
    }

    ctx.putImageData(imageData, 0, 0);
  }, [data, width, height]);

  return <canvas ref={canvasRef} width={width} height={height} />;
};

const ThreeDView = ({ data }) => {
  const pointsRef = useRef();

  useFrame(() => {
    if (pointsRef.current && data) {
      const positions = pointsRef.current.geometry.attributes.position.array;
      data.positions.forEach((pos, i) => {
        positions[i * 3] = pos[0];
        positions[i * 3 + 1] = pos[1];
        positions[i * 3 + 2] = 0;
      });
      pointsRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  return (
    <Canvas camera={{ position: [0, 0, 5] }}>
      <OrbitControls />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attachObject={['attributes', 'position']}
            count={data?.positions?.length || 0}
            array={new Float32Array((data?.positions?.length || 0) * 3)}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial size={0.05} color="#8884d8" />
      </points>
    </Canvas>
  );
};

const ParameterControls = ({ params, setParams }) => {
  return (
    <div className="space-y-4 p-4 border rounded-lg bg-white">
      <h3 className="text-lg font-semibold">Simulation Parameters</h3>
      
      <div className="space-y-2">
        <label className="text-sm font-medium">Number of Agents</label>
        <Slider 
          min={50} 
          max={500} 
          step={10}
          value={[params.numAgents]}
          onValueChange={([value]) => setParams({...params, numAgents: value})}
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Learning Rate</label>
        <Slider 
          min={0.001} 
          max={0.1} 
          step={0.001}
          value={[params.learningRate]}
          onValueChange={([value]) => setParams({...params, learningRate: value})}
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Interaction Radius</label>
        <Slider 
          min={0.1} 
          max={2.0} 
          step={0.1}
          value={[params.interactionRadius]}
          onValueChange={([value]) => setParams({...params, interactionRadius: value})}
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Force Magnitude</label>
        <Slider 
          min={0.1} 
          max={5.0} 
          step={0.1}
          value={[params.forceMagnitude]}
          onValueChange={([value]) => setParams({...params, forceMagnitude: value})}
        />
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Pattern Type</label>
        <select 
          className="w-full p-2 border rounded"
          value={params.patternType}
          onChange={(e) => setParams({...params, patternType: e.target.value})}
        >
          <option value="circle">Circle</option>
          <option value="grid">Grid</option>
          <option value="random_clusters">Random Clusters</option>
        </select>
      </div>
    </div>
  );
};

const AdvancedPatternViz: React.FC = () => {
  const [data, setData] = useState<any>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [viewMode, setViewMode] = useState<'2d' | '3d' | 'heat'>('2d');
  const [params, setParams] = useState<SimulationParams>({
    numAgents: 200,
    learningRate: 0.005,
    interactionRadius: 0.3,
    forceMagnitude: 1.0,
    patternType: 'circle'
  });

  const wsRef = useRef<WebSocket | null>(null);
  const canvasWidth = 600;
  const canvasHeight = 600;

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const newData = JSON.parse(event.data);
      setData(newData);
      setMetrics(prev => [...prev, {
        time: new Date(newData.timestamp),
        pattern_score: newData.pattern_score,
        density: newData.density_variation,
        order: newData.spatial_order
      }].slice(-100));
    };

    return () => ws.close();
  }, []);

  const updateParams = () => {
    wsRef.current?.send(JSON.stringify({
      type: 'update_params',
      params: params
    }));
  };

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex justify-between items-center mb-4">
        <div className="flex gap-2">
          <Button 
            onClick={() => setViewMode('2d')}
            variant={viewMode === '2d' ? 'default' : 'outline'}
          >
            2D View
          </Button>
          <Button 
            onClick={() => setViewMode('3d')}
            variant={viewMode === '3d' ? 'default' : 'outline'}
          >
            3D View
          </Button>
          <Button 
            onClick={() => setViewMode('heat')}
            variant={viewMode === 'heat' ? 'default' : 'outline'}
          >
            Heat Map
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Visualization Panel */}
        <div className="col-span-2 border rounded-lg bg-white p-4">
          {viewMode === '2d' && (
            <Stage width={canvasWidth} height={canvasHeight}>
              <Layer>
                {data?.positions.map((pos, i) => (
                  <Circle
                    key={i}
                    x={(pos[0] + 1) * canvasWidth / 2}
                    y={(pos[1] + 1) * canvasHeight / 2}
                    radius={3}
                    fill="#8884d8"
                    opacity={0.7}
                  />
                ))}
              </Layer>
            </Stage>
          )}
          {viewMode === '3d' && (
            <div style={{ height: canvasHeight }}>
              <ThreeDView data={data} />
            </div>
          )}
          {viewMode === 'heat' && (
            <HeatMap data={data} width={canvasWidth} height={canvasHeight} />
          )}
        </div>

        {/* Parameters Panel */}
        <div className="col-span-1">
          <ParameterControls params={params} setParams={setParams} />
        </div>

        {/* Metrics Panel */}
        <div className="col-span-3 border rounded-lg bg-white p-4">
          <h3 className="text-lg font-semibold mb-4">Pattern Metrics</h3>
          <div className="h-64">
            <Line
              data={{
                labels: metrics.map(m => m.time.toLocaleTimeString()),
                datasets: [
                  {
                    label: 'Pattern Score',
                    data: metrics.map(m => m.pattern_score),
                    borderColor: '#8884d8',
                    fill: false,
                  },
                  {
                    label: 'Density Variation',
                    data: metrics.map(m => m.density),
                    borderColor: '#82ca9d',
                    fill: false,
                  },
                  {
                    label: 'Spatial Order',
                    data: metrics.map(m => m.order),
                    borderColor: '#ffc658',
                    fill: false,
                  }
                ]
              }}
              options={{
                animation: false,
                scales: {
                  x: {
                    display: true,
                    title: { display: true, text: 'Time' }
                  },
                  y: {
                    display: true,
                    title: { display: true, text: 'Value' }
                  }
                }
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedPatternViz;