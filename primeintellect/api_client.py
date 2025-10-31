"""
Prime Intellect API Client

This module provides a Python interface to the Prime Intellect API for managing
multi-node clusters and GPU instances.

Reference: https://docs.primeintellect.ai/api-reference/introduction
"""

import os
import time
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ClusterAvailability:
    """Represents available cluster configuration"""
    gpu_type: str
    gpu_count: int
    gpu_memory: int
    cloud_id: str
    provider: str
    region: str
    country: str
    price_per_hour: float
    stock_status: str
    interconnect: Optional[int] = None
    interconnect_type: Optional[str] = None


@dataclass
class PodInfo:
    """Represents a Prime Intellect pod"""
    id: str
    name: str
    status: str
    gpu_name: str
    gpu_count: int
    price_per_hour: float
    ip: Optional[str] = None
    ssh_connection: Optional[str] = None
    cluster_id: Optional[str] = None


class PrimeIntellectClient:
    """Client for interacting with Prime Intellect API"""
    
    BASE_URL = "https://api.primeintellect.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Prime Intellect API client
        
        Args:
            api_key: Prime Intellect API key. If not provided, will look for PRIME_INTELLECT_API_KEY env var
        """
        self.api_key = api_key or os.environ.get("PRIME_INTELLECT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Prime Intellect API key required. Set PRIME_INTELLECT_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        })
    
    def get_cluster_availability(
        self,
        gpu_type: Optional[str] = None,
        gpu_count: Optional[int] = None,
        regions: Optional[List[str]] = None
    ) -> Dict[str, List[ClusterAvailability]]:
        """
        Get available cluster configurations
        
        Args:
            gpu_type: GPU type filter (e.g., "H100_80GB")
            gpu_count: Desired number of GPUs
            regions: List of regions to filter
            
        Returns:
            Dictionary mapping GPU types to available clusters
        """
        url = f"{self.BASE_URL}/availability/clusters"
        params = {}
        
        if gpu_type:
            params["gpu_type"] = gpu_type
        if gpu_count:
            params["gpu_count"] = gpu_count
        if regions:
            params["regions"] = regions
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse availability data
        availability_map = {}
        for gpu_type_key, clusters in data.items():
            availability_list = []
            for cluster in clusters:
                if cluster.get("stockStatus") == "Available":
                    pricing = cluster.get("prices", {})
                    availability_list.append(ClusterAvailability(
                        gpu_type=cluster.get("gpuType", ""),
                        gpu_count=cluster.get("gpuCount", 0),
                        gpu_memory=cluster.get("gpuMemory", 0),
                        cloud_id=cluster.get("cloudId", ""),
                        provider=cluster.get("provider", ""),
                        region=cluster.get("region", ""),
                        country=cluster.get("country", ""),
                        price_per_hour=pricing.get("onDemand", 0.0),
                        stock_status=cluster.get("stockStatus", ""),
                        interconnect=cluster.get("interconnect"),
                        interconnect_type=cluster.get("interconnectType")
                    ))
            availability_map[gpu_type_key] = availability_list
        
        return availability_map
    
    def find_best_cluster(
        self,
        gpu_type: str = "H100_80GB",
        gpu_count: int = 8,
        max_price_per_hour: Optional[float] = None
    ) -> Optional[ClusterAvailability]:
        """
        Find the best available cluster configuration
        
        Args:
            gpu_type: GPU type (e.g., "H100_80GB")
            gpu_count: Desired number of GPUs
            max_price_per_hour: Maximum price per hour
            
        Returns:
            Best matching ClusterAvailability or None
        """
        availability = self.get_cluster_availability(gpu_type=gpu_type, gpu_count=gpu_count)
        
        # Find clusters for the requested GPU type
        clusters = availability.get(gpu_type, [])
        
        if not clusters:
            return None
        
        # Filter by price if specified
        if max_price_per_hour:
            clusters = [c for c in clusters if c.price_per_hour <= max_price_per_hour]
        
        if not clusters:
            return None
        
        # Sort by price and return cheapest
        clusters.sort(key=lambda c: c.price_per_hour)
        return clusters[0]
    
    def create_pod(
        self,
        name: str,
        cloud_id: str,
        gpu_type: str = "H100_80GB",
        gpu_count: int = 8,
        disk_size: int = 100,
        image: str = "ubuntu_22_cuda_12",
        auto_restart: bool = False
    ) -> PodInfo:
        """
        Create a new pod (cluster node)
        
        Args:
            name: Name for the pod
            cloud_id: Cloud ID from availability check
            gpu_type: GPU type (e.g., "H100_80GB")
            gpu_count: Number of GPUs (typically 8 per node)
            disk_size: Disk size in GB
            image: Base image to use
            auto_restart: Whether to auto-restart on failure
            
        Returns:
            PodInfo object with pod details
        """
        url = f"{self.BASE_URL}/pods/"
        
        payload = {
            "pod": {
                "name": name,
                "cloudId": cloud_id,
                "gpuType": gpu_type,
                "socket": "SXM5",  # H100 typically uses SXM5
                "gpuCount": gpu_count,
                "diskSize": disk_size,
                "image": image,
                "autoRestart": auto_restart,
                "security": "secure_cloud"
            }
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        return PodInfo(
            id=data.get("id", ""),
            name=data.get("name", ""),
            status=data.get("status", ""),
            gpu_name=data.get("gpuName", ""),
            gpu_count=data.get("gpuCount", 0),
            price_per_hour=data.get("priceHr", 0.0),
            cluster_id=data.get("clusterId")
        )
    
    def get_pods(self) -> List[PodInfo]:
        """
        List all pods for the current user
        
        Returns:
            List of PodInfo objects
        """
        url = f"{self.BASE_URL}/pods/"
        response = self.session.get(url)
        response.raise_for_status()
        
        data = response.json()
        pods = []
        
        for pod_data in data.get("data", []):
            pods.append(PodInfo(
                id=pod_data.get("id", ""),
                name=pod_data.get("name", ""),
                status=pod_data.get("status", ""),
                gpu_name=pod_data.get("gpuName", ""),
                gpu_count=pod_data.get("gpuCount", 0),
                price_per_hour=pod_data.get("priceHr", 0.0),
                ip=pod_data.get("ip"),
                ssh_connection=pod_data.get("sshConnection"),
                cluster_id=pod_data.get("clusterId")
            ))
        
        return pods
    
    def get_pod(self, pod_id: str) -> PodInfo:
        """
        Get details for a specific pod
        
        Args:
            pod_id: Pod ID
            
        Returns:
            PodInfo object
        """
        url = f"{self.BASE_URL}/pods/{pod_id}"
        response = self.session.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        return PodInfo(
            id=data.get("id", ""),
            name=data.get("name", ""),
            status=data.get("status", ""),
            gpu_name=data.get("gpuName", ""),
            gpu_count=data.get("gpuCount", 0),
            price_per_hour=data.get("priceHr", 0.0),
            ip=data.get("ip"),
            ssh_connection=data.get("sshConnection"),
            cluster_id=data.get("clusterId")
        )
    
    def delete_pod(self, pod_id: str) -> Dict[str, Any]:
        """
        Delete a pod
        
        Args:
            pod_id: Pod ID to delete
            
        Returns:
            Response dictionary
        """
        url = f"{self.BASE_URL}/pods/{pod_id}"
        response = self.session.delete(url)
        response.raise_for_status()
        
        return response.json()
    
    def wait_for_pod_ready(self, pod_id: str, timeout: int = 600) -> PodInfo:
        """
        Wait for a pod to be ready
        
        Args:
            pod_id: Pod ID
            timeout: Maximum time to wait in seconds
            
        Returns:
            PodInfo when ready
            
        Raises:
            TimeoutError if pod doesn't become ready in time
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            pod = self.get_pod(pod_id)
            
            if pod.status == "RUNNING" and pod.ip:
                return pod
            
            if pod.status in ["FAILED", "TERMINATED"]:
                raise RuntimeError(f"Pod {pod_id} failed with status: {pod.status}")
            
            time.sleep(10)  # Wait 10 seconds before checking again
        
        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout} seconds")

