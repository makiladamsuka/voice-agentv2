"""
System Information Tools for Raspberry Pi
Provides CPU temperature, usage, memory, and other system metrics.
"""

import subprocess
import re
import os
from pathlib import Path
from livekit.agents import RunContext

class SystemTools:
    """Tools for getting Raspberry Pi system information"""
    
    def __init__(self):
        pass
    
    def _get_cpu_temp(self) -> float:
        """Get CPU temperature in Celsius from Raspberry Pi"""
        try:
            # Try /sys/class/thermal/thermal_zone0/temp (most common)
            temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_file.exists():
                temp = int(temp_file.read_text().strip()) / 1000.0
                return temp
            
            # Try vcgencmd (Raspberry Pi specific)
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                match = re.search(r"temp=([\d.]+)'C", result.stdout)
                if match:
                    return float(match.group(1))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, PermissionError):
            pass
        
        return None
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            # Use top command to get CPU usage
            result = subprocess.run(
                ["top", "-bn1"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # Parse top output for CPU usage
                lines = result.stdout.split('\n')
                for line in lines:
                    if '%Cpu(s)' in line or 'Cpu(s)' in line:
                        # Extract idle percentage and calculate usage
                        match = re.search(r'(\d+\.?\d*)%?\s*id', line)
                        if match:
                            idle = float(match.group(1))
                            return 100.0 - idle
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, PermissionError):
            pass
        
        # Fallback: use psutil if available
        try:
            import psutil
            return psutil.cpu_percent(interval=0.5)
        except ImportError:
            pass
        
        return None
    
    def _get_memory_info(self) -> dict:
        """Get memory usage information"""
        try:
            # Read from /proc/meminfo
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                mem_data = {}
                with open(meminfo, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            # Extract number (in kB)
                            match = re.search(r'(\d+)', value)
                            if match:
                                mem_data[key.strip()] = int(match.group(1))
                
                if 'MemTotal' in mem_data and 'MemAvailable' in mem_data:
                    total_kb = mem_data['MemTotal']
                    available_kb = mem_data['MemAvailable']
                    used_kb = total_kb - available_kb
                    
                    return {
                        'total_mb': round(total_kb / 1024, 1),
                        'used_mb': round(used_kb / 1024, 1),
                        'available_mb': round(available_kb / 1024, 1),
                        'percent': round((used_kb / total_kb) * 100, 1)
                    }
        except (FileNotFoundError, ValueError, PermissionError):
            pass
        
        # Fallback: use psutil if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'total_mb': round(mem.total / (1024 * 1024), 1),
                'used_mb': round(mem.used / (1024 * 1024), 1),
                'available_mb': round(mem.available / (1024 * 1024), 1),
                'percent': mem.percent
            }
        except ImportError:
            pass
        
        return None
    
    def _get_disk_usage(self) -> dict:
        """Get disk usage information"""
        try:
            result = subprocess.run(
                ["df", "-h", "/"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        return {
                            'total': parts[1],
                            'used': parts[2],
                            'available': parts[3],
                            'percent': parts[4]
                        }
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, PermissionError):
            pass
        
        # Fallback: use psutil if available
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return {
                'total': f"{disk.total / (1024**3):.1f}G",
                'used': f"{disk.used / (1024**3):.1f}G",
                'available': f"{disk.free / (1024**3):.1f}G",
                'percent': f"{disk.percent:.1f}%"
            }
        except ImportError:
            pass
        
        return None
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            uptime_file = Path("/proc/uptime")
            if uptime_file.exists():
                uptime_seconds = float(uptime_file.read_text().split()[0])
                
                days = int(uptime_seconds // 86400)
                hours = int((uptime_seconds % 86400) // 3600)
                minutes = int((uptime_seconds % 3600) // 60)
                
                parts = []
                if days > 0:
                    parts.append(f"{days} day{'s' if days != 1 else ''}")
                if hours > 0:
                    parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
                if minutes > 0 or not parts:
                    parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
                
                return ", ".join(parts)
        except (FileNotFoundError, ValueError):
            pass
        
        return None
    
    async def get_cpu_temperature(self, context: RunContext) -> str:
        """
        Gets the CPU temperature of the Raspberry Pi.
        Use when user asks about CPU temperature, temperature, or system heat.
        """
        print("🌡️ [TOOL] get_cpu_temperature called")
        
        temp = self._get_cpu_temp()
        
        if temp is None:
            return "I'm unable to read the CPU temperature at the moment. The system may not support temperature monitoring."
        
        # Format temperature
        temp_c = temp
        temp_f = (temp_c * 9/5) + 32
        
        # Determine status
        if temp_c < 40:
            status = "cool"
        elif temp_c < 60:
            status = "normal"
        elif temp_c < 70:
            status = "warm"
        else:
            status = "hot"
        
        return f"The CPU temperature is {temp_c:.1f} degrees Celsius ({temp_f:.1f} degrees Fahrenheit). The system is running {status}."
    
    async def get_system_info(self, context: RunContext) -> str:
        """
        Gets comprehensive system information including CPU temperature, usage, memory, disk, and uptime.
        Use when user asks about system status, system information, or wants to know how the system is performing.
        """
        print("💻 [TOOL] get_system_info called")
        
        info_parts = []
        
        # CPU Temperature
        temp = self._get_cpu_temp()
        if temp is not None:
            temp_f = (temp * 9/5) + 32
            info_parts.append(f"CPU temperature: {temp:.1f}°C ({temp_f:.1f}°F)")
        
        # CPU Usage
        cpu_usage = self._get_cpu_usage()
        if cpu_usage is not None:
            info_parts.append(f"CPU usage: {cpu_usage:.1f}%")
        
        # Memory
        mem = self._get_memory_info()
        if mem:
            info_parts.append(f"Memory: {mem['used_mb']:.0f}MB used out of {mem['total_mb']:.0f}MB ({mem['percent']:.1f}%)")
        
        # Disk
        disk = self._get_disk_usage()
        if disk:
            info_parts.append(f"Disk: {disk['used']} used out of {disk['total']} ({disk['percent']})")
        
        # Uptime
        uptime = self._get_uptime()
        if uptime:
            info_parts.append(f"Uptime: {uptime}")
        
        if not info_parts:
            return "I'm unable to retrieve system information at the moment."
        
        return "Here's the current system information: " + ". ".join(info_parts) + "."
    
    async def get_cpu_usage(self, context: RunContext) -> str:
        """
        Gets the CPU usage percentage.
        Use when user asks about CPU usage, CPU load, or processor performance.
        """
        print("⚡ [TOOL] get_cpu_usage called")
        
        usage = self._get_cpu_usage()
        
        if usage is None:
            return "I'm unable to read the CPU usage at the moment."
        
        if usage < 30:
            status = "low"
        elif usage < 70:
            status = "moderate"
        else:
            status = "high"
        
        return f"The CPU usage is {usage:.1f}%, which is {status}."
    
    async def get_memory_usage(self, context: RunContext) -> str:
        """
        Gets the memory (RAM) usage information.
        Use when user asks about memory, RAM usage, or available memory.
        """
        print("🧠 [TOOL] get_memory_usage called")
        
        mem = self._get_memory_info()
        
        if mem is None:
            return "I'm unable to read the memory usage at the moment."
        
        status = "low" if mem['percent'] < 50 else "moderate" if mem['percent'] < 80 else "high"
        
        return f"Memory usage: {mem['used_mb']:.0f} megabytes used out of {mem['total_mb']:.0f} megabytes total ({mem['percent']:.1f}%). The memory usage is {status}."
