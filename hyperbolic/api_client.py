"""
Hyperbolic Labs API Client

This module provides a Python interface to the Hyperbolic Labs GPU Marketplace API.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class GPUInstance:
    """Represents a GPU instance on Hyperbolic Labs"""
    id: str
    cluster_name: str
    node_name: str
    gpu_count: int
    gpu_type: str
    status: str
    pricing: float
    ram_gb: int
    storage_gb: int
    gpus_total: int
    gpus_reserved: int
    reserved: bool


class HyperbolicClient:
    """Client for interacting with Hyperbolic Labs API"""
    
    BASE_URL = "https://api.hyperbolic.xyz/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Hyperbolic Labs API client
        
        Args:
            api_key: Hyperbolic API key. If not provided, will look for HYPERBOLIC_API_KEY env var
        """
        self.api_key = api_key or os.environ.get("HYPERBOLIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Hyperbolic API key required. Set HYPERBOLIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        })
    
    def list_available_machines(self) -> List[GPUInstance]:
        """
        List all available machines in the marketplace
        
        Returns:
            List of GPUInstance objects representing available machines
        """
        url = f"{self.BASE_URL}/marketplace"
        response = self.session.post(url, json={})
        response.raise_for_status()
        
        machines = response.json()
        instances = []
        
        for machine in machines:
            # Parse GPU type from hardware info
            gpu_type = "Unknown"
            if "hardware" in machine:
                hw = machine["hardware"]
                if "gpu" in hw and "model" in hw["gpu"]:
                    gpu_type = hw["gpu"]["model"]
            
            instances.append(GPUInstance(
                id=machine.get("id", ""),
                cluster_name=machine.get("cluster_name", ""),
                node_name=machine.get("id", ""),  # node_name is same as id
                gpu_count=machine.get("gpus_total", 0),
                gpu_type=gpu_type,
                status=machine.get("status", ""),
                pricing=float(machine.get("pricing", {}).get("price", {}).get("amount", 0)),
                ram_gb=machine.get("hardware", {}).get("ram_gb", 0),
                storage_gb=machine.get("hardware", {}).get("storage_gb", 0),
                gpus_total=machine.get("gpus_total", 0),
                gpus_reserved=machine.get("gpus_reserved", 0),
                reserved=machine.get("reserved", True)
            ))
        
        return instances
    
    def find_best_machine(
        self,
        min_gpus: int = 1,
        gpu_type_preference: Optional[List[str]] = None,
        max_price_per_hour: Optional[float] = None
    ) -> Optional[GPUInstance]:
        """
        Find the best available machine based on criteria
        
        Args:
            min_gpus: Minimum number of GPUs required
            gpu_type_preference: Preferred GPU types in order of preference (e.g., ["H100", "A100"])
            max_price_per_hour: Maximum price per hour willing to pay
        
        Returns:
            Best matching GPUInstance or None if no suitable machine found
        """
        machines = self.list_available_machines()
        
        # Filter available machines
        available = [
            m for m in machines
            if not m.reserved
            and m.status == "node_ready"
            and (m.gpus_total - m.gpus_reserved) >= min_gpus
        ]
        
        if max_price_per_hour:
            available = [m for m in available if m.pricing <= max_price_per_hour]
        
        if not available:
            return None
        
        # Sort by preference
        if gpu_type_preference:
            def preference_score(machine: GPUInstance) -> tuple:
                # Lower score is better
                for i, pref in enumerate(gpu_type_preference):
                    if pref.upper() in machine.gpu_type.upper():
                        return (i, machine.pricing)
                return (len(gpu_type_preference), machine.pricing)
            
            available.sort(key=preference_score)
        else:
            # Sort by price if no preference
            available.sort(key=lambda m: m.pricing)
        
        return available[0]
    
    def create_instance(
        self,
        cluster_name: str,
        node_name: str,
        gpu_count: int,
        image_name: Optional[str] = None,
        image_username: Optional[str] = None,
        image_password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new GPU instance
        
        Args:
            cluster_name: Name of the cluster
            node_name: Name of the node (from list_available_machines)
            gpu_count: Number of GPUs to allocate
            image_name: Optional container image name
            image_username: Optional container registry username
            image_password: Optional container registry password
        
        Returns:
            Response dictionary with instance details
        """
        url = f"{self.BASE_URL}/marketplace/instances/create"
        
        payload = {
            "cluster_name": cluster_name,
            "node_name": node_name,
            "gpu_count": gpu_count
        }
        
        if image_name:
            payload["image"] = {
                "name": image_name
            }
            if image_username:
                payload["image"]["username"] = image_username
            if image_password:
                payload["image"]["password"] = image_password
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def list_instances(self) -> List[Dict[str, Any]]:
        """
        List all instances for the current user
        
        Returns:
            List of instance dictionaries
        """
        url = f"{self.BASE_URL}/marketplace/instances"
        response = self.session.post(url, json={})
        response.raise_for_status()
        
        return response.json()
    
    def get_instance_history(self) -> List[Dict[str, Any]]:
        """
        Get instance history for the current user
        
        Returns:
            List of historical instance records
        """
        url = f"{self.BASE_URL}/marketplace/instances/history"
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Terminate a running instance
        
        Args:
            instance_id: ID of the instance to terminate
        
        Returns:
            Response dictionary
        """
        url = f"{self.BASE_URL}/marketplace/instances/terminate"
        payload = {"instance_id": instance_id}
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def get_credit_balance(self) -> Dict[str, Any]:
        """
        Query current credit balance
        
        Returns:
            Dictionary with balance information
        """
        url = "https://api.hyperbolic.xyz/billing/get_current_balance"
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
