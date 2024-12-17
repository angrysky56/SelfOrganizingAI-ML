import React, { useRef, useEffect, useState } from 'react';
import { Stage, Layer, Circle, Arrow } from 'react-konva';
import { Line } from 'react-chartjs-2';

interface AgentData {
  positions: number[][];
  velocities: number[][];
  pattern_score: number;
  density_variation: number;
  spatial_order: number;
  timestamp: string;
}

const LivePatternViz: React.FC = () => {
  const [data, setData] = useState<AgentData | null>(null);
  const [metrics, setMetrics] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const canvasWidth = 600;
  const canvasHeight = 600;

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const newData = JSON.parse(event.data);
      setData(newData);
      
      // Update metrics history
      setMetrics(prev => [...prev, {
        time: new Date(newData.timestamp),
        pattern_score: newData.pattern_score,
        density: newData.density_variation,
        order: newData.spatial_order
      }].slice(-100));  // Keep last 100 points
    };

    return () => ws.close();
  }, []);

  const scalePosition = (pos: number[]) => {
    return [
      (pos[0] + 1) * canvasWidth / 2,
      (pos[1] + 1) * canvasHeight / 2
    ];
  };

  return (
    <div className="grid grid-cols-2 gap-4 p-4">
      {/* Agent Visualization */}
      <div className="border rounded-lg p-4 bg-white">
        <h3 className="text-lg font-semibold mb-4">Agent Positions</h3>
        <Stage width={canvasWidth} height={canvasHeight}>
          <Layer>
            {data?.positions.map((pos, i) => {
              const [x, y] = scalePosition(pos);
              const [vx, vy] = data.velocities[i];
              const velocity = Math.sqrt(vx*vx + vy*vy);
              
              return (
                <React.Fragment key={i}>
                  <Circle
                    x={x}
                    y={y}
                    radius={3}
                    fill="#8884d8"
                    opacity={0.7}
                  />
                  {velocity > 0.01 && (
                    <Arrow
                      x={x}
                      y={y}
                      points={[0, 0, vx * 20, vy * 20]}
                      pointerLength={5}
                      pointerWidth={5}
                      fill="#82ca9d"
                      stroke="#82ca9d"
                      opacity={0.5}
                    />
                  )}
                </React.Fragment>
              );
            })}
          </Layer>
        </Stage>
      </div>

      {/* Metrics Visualization */}
      <div className="border rounded-lg p-4 bg-white">
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
  );
};

export default LivePatternViz;