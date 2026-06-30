import type { NextConfig } from 'next';

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8080';

const nextConfig: NextConfig = {
  output: 'standalone',
  eslint: {
    ignoreDuringBuilds: true,
  },
  async rewrites() {
    return [
      { source: '/api/facebook', destination: `${BACKEND_URL}/api/facebook` },
      { source: '/api/map', destination: `${BACKEND_URL}/api/map` },
      { source: '/api/network-ip', destination: `${BACKEND_URL}/api/network-ip` },
      { source: '/api/upload-poster', destination: `${BACKEND_URL}/api/upload-poster` },
      { source: '/api/upload-status', destination: `${BACKEND_URL}/api/upload-status` },
      { source: '/api/weather', destination: `${BACKEND_URL}/api/weather` },
      { source: '/api/eye-color', destination: `${BACKEND_URL}/api/eye-color` },
      { source: '/api/image', destination: `${BACKEND_URL}/api/image` },
    ];
  },
  // Disable SWC for ARM compatibility (Raspberry Pi)
  experimental: {
    forceSwcTransforms: false,
  },
  // Use Babel instead of SWC
  compiler: undefined,
};

export default nextConfig;
