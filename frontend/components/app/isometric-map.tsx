'use client';

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Box, Grid, Text } from '@react-three/drei';
import * as THREE from 'three';

// Animated glowing dashed path that flows toward the destination
const GlowingPath = ({ points }: { points: [number, number, number][] }) => {
  const lineRef = useRef<any>(null);
  useFrame((_, delta) => {
    if (lineRef.current?.material) {
      lineRef.current.material.dashOffset -= delta * 2;
    }
  });
  if (points.length < 2) return null;
  return (
    <Line ref={lineRef} points={points} color="#ef4444" lineWidth={6} dashed dashSize={0.5} gapSize={0.3} />
  );
};

// Pulsing destination marker
const DestinationMarker = ({ position, label }: { position: [number, number, number], label: string }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (meshRef.current) {
      meshRef.current.position.y = position[1] + 0.3 + Math.sin(t * 3) * 0.2;
    }
    if (ringRef.current) {
      ringRef.current.scale.setScalar(1 + Math.sin(t * 2) * 0.3);
      (ringRef.current.material as THREE.MeshBasicMaterial).opacity = 0.4 + Math.sin(t * 2) * 0.2;
    }
  });

  return (
    <group>
      {/* Pulsing ring on the ground */}
      <mesh ref={ringRef} position={[position[0], 0.05, position[2]]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.6, 0.9, 32]} />
        <meshBasicMaterial color="#22c55e" transparent opacity={0.5} side={THREE.DoubleSide} />
      </mesh>
      {/* Floating marker */}
      <mesh ref={meshRef} position={position}>
        <octahedronGeometry args={[0.3]} />
        <meshStandardMaterial color="#22c55e" emissive="#22c55e" emissiveIntensity={0.8} />
      </mesh>
      {/* Label */}
      <Text
        position={[position[0], position[1] + 1.5, position[2]]}
        fontSize={0.5}
        color="#22c55e"
        anchorX="center"
        anchorY="middle"
        fontWeight="bold"
      >
        📍 {label}
      </Text>
    </group>
  );
};

interface NavigationMapProps {
  path: number[][];         // Array of [x, y, z] world coordinates
  nodes: any[];             // All nodes from the map
  buildings: any;           // Building positions/sizes
  destination: string;      // Destination label
  onClose?: () => void;     // Close callback
}

export default function NavigationMap({ path, nodes, buildings, destination, onClose }: NavigationMapProps) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Animate in
    setTimeout(() => setVisible(true), 50);
    
    // Auto-dismiss after 20 seconds
    const timer = setTimeout(() => {
      setVisible(false);
      setTimeout(() => onClose?.(), 400);
    }, 20000);

    return () => clearTimeout(timer);
  }, [onClose]);

  const handleClose = () => {
    setVisible(false);
    setTimeout(() => onClose?.(), 400);
  };

  const pathPoints = useMemo(() => {
    return path.map(p => [p[0], 0.3, p[2]] as [number, number, number]);
  }, [path]);

  const destNode = useMemo(() => {
    return nodes.find((n: any) => n.label?.toLowerCase() === destination.toLowerCase() && n.type !== 'waypoint');
  }, [nodes, destination]);

  const buildingEntries = useMemo(() => {
    return Object.entries(buildings || {}) as [string, any][];
  }, [buildings]);

  return (
    <div 
      className={`fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex flex-col items-center justify-center transition-all duration-500 ${
        visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
      }`}
      onClick={handleClose}
    >
      {/* Header */}
      <div className="absolute top-6 left-0 right-0 flex justify-center z-10" onClick={(e) => e.stopPropagation()}>
        <div className="bg-gray-900/90 border border-gray-700 rounded-2xl px-8 py-4 flex items-center gap-4 shadow-2xl">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
          <span className="text-white text-lg font-bold">Navigating to: {destination}</span>
          <button onClick={handleClose} className="ml-4 text-gray-400 hover:text-white text-2xl font-bold">&times;</button>
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="w-full h-full" onClick={(e) => e.stopPropagation()}>
        <Canvas shadows orthographic camera={{ position: [20, 20, 20], zoom: 35 }}>
          <ambientLight intensity={0.6} />
          <directionalLight position={[10, 20, 10]} intensity={1} castShadow />
          <pointLight position={[0, 5, 0]} intensity={0.5} color="#ef4444" distance={15} />

          {/* Building Grids */}
          {buildingEntries.map(([bId, b]) => (
            <group key={bId} position={b.position}>
              <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
                <planeGeometry args={[b.size[0], b.size[1]]} />
                <meshStandardMaterial color={b.color} transparent opacity={0.7} />
              </mesh>
              <Grid
                args={[b.size[0], b.size[1]]}
                position={[0, 0.01, 0]}
                cellSize={1}
                cellThickness={1}
                cellColor="#555"
                sectionSize={1}
                sectionThickness={1.5}
                sectionColor="#333"
                fadeDistance={40}
              />
              <Text position={[0, 0.02, b.size[1] / 2 + 0.5]} rotation={[-Math.PI / 2, 0, 0]} fontSize={0.7} color={b.color}>
                {b.name}
              </Text>
            </group>
          ))}

          {/* Room Blocks */}
          {nodes.filter((n: any) => n.type !== 'waypoint').map((node: any) => {
            const size = node.size || [1, 1, 1];
            const isDestination = node.label?.toLowerCase() === destination.toLowerCase();
            return (
              <group key={node.id} position={[node.world[0], size[1] / 2, node.world[2]]}>
                <Box args={size} castShadow>
                  <meshStandardMaterial
                    color={isDestination ? '#22c55e' : '#334155'}
                    emissive={isDestination ? '#22c55e' : '#000000'}
                    emissiveIntensity={isDestination ? 0.3 : 0}
                    transparent
                    opacity={isDestination ? 1 : 0.8}
                  />
                </Box>
                <Text position={[0, size[1] / 2 + 0.4, 0]} fontSize={0.35} color="#ffffff" anchorX="center" anchorY="middle">
                  {node.label}
                </Text>
              </group>
            );
          })}

          {/* Glowing Animated Path */}
          {pathPoints.length >= 2 && <GlowingPath points={pathPoints} />}

          {/* Destination Marker */}
          {destNode && (
            <DestinationMarker
              position={[destNode.world[0], destNode.world[1], destNode.world[2]]}
              label={destination}
            />
          )}

          <OrbitControls enableZoom={true} enablePan={true} maxPolarAngle={Math.PI / 2 - 0.1} />
        </Canvas>
      </div>

      {/* Bottom hint */}
      <div className="absolute bottom-6 left-0 right-0 flex justify-center z-10">
        <p className="text-gray-400 text-sm">Tap anywhere to close • Auto-closes in 20 seconds</p>
      </div>
    </div>
  );
}
