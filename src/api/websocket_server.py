from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Any
import asyncio
import json
import logging
import torch

logger = logging.getLogger(__name__)

class ExperimentStreamManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.experiment_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        if client_id in self.experiment_tasks:
            self.experiment_tasks[client_id].cancel()
            del self.experiment_tasks[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_update(self, client_id: str, data: Dict[str, Any]):
        if client_id in self.active_connections:
            # Convert tensors to lists for JSON serialization
            processed_data = self._process_data_for_json(data)
            await self.active_connections[client_id].send_json(processed_data)

    def _process_data_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PyTorch tensors to JSON-serializable format."""
        processed = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, dict):
                processed[key] = self._process_data_for_json(value)
            elif isinstance(value, (list, tuple)):
                processed[key] = [self._process_data_for_json(item) if isinstance(item, dict)
                                else item.detach().cpu().numpy().tolist() if isinstance(item, torch.Tensor)
                                else item for item in value]
            else:
                processed[key] = value
        return processed

    async def start_experiment_stream(self, client_id: str, experiment_instance: Any):
        """Start streaming experiment updates to client."""
        try:
            async for state in experiment_instance.run_with_updates():
                await self.send_update(client_id, {
                    'type': 'experiment_update',
                    'state': state
                })
        except Exception as e:
            logger.error(f"Error in experiment stream: {str(e)}")
            await self.send_update(client_id, {
                'type': 'error',
                'message': str(e)
            })
        finally:
            await self.send_update(client_id, {
                'type': 'experiment_completed'
            })