"""
Process Control Module for managing ethical analysis processes
"""

import asyncio
from typing import Dict, List, Any, Optional, Set
import logging
from datetime import datetime, timedelta
from .process import EthicalProcess, ProcessState, ProcessResult
from .drift_detector import DriftDetector

class ProcessController:
    """Controls and coordinates ethical analysis processes."""
    
    def __init__(self, 
                 max_concurrent: int = 5,
                 drift_monitoring: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize process controller.
        
        Args:
            max_concurrent: Maximum concurrent processes
            drift_monitoring: Whether to monitor for process drift
            logger: Optional logger instance
        """
        self.max_concurrent = max_concurrent
        self.logger = logger or logging.getLogger(__name__)
        
        self.active_processes: Dict[str, EthicalProcess] = {}
        self.queued_processes: List[EthicalProcess] = []
        self.completed_processes: Dict[str, ProcessResult] = {}
        self.process_history: List[Dict[str, Any]] = []
        
        self.dependencies: Dict[str, Set[str]] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize drift detector if monitoring enabled
        self.drift_monitoring = drift_monitoring
        if drift_monitoring:
            self.drift_detector = DriftDetector(
                callback=self._handle_drift_alert
            )
            
    async def submit_process(self,
                           process: EthicalProcess,
                           input_data: Dict[str, Any],
                           parameters: Optional[Dict[str, Any]] = None,
                           dependencies: Optional[List[str]] = None) -> str:
        """
        Submit a process for execution.
        
        Args:
            process: Process to execute
            input_data: Input data for process
            parameters: Optional execution parameters
            dependencies: Optional list of dependent process IDs
            
        Returns:
            Process ID
        """
        process_id = process.process_id
        
        # Store dependencies
        if dependencies:
            self.dependencies[process_id] = set(dependencies)
            
            # Check if dependencies are complete
            for dep_id in dependencies:
                if dep_id not in self.completed_processes:
                    self.logger.info(f"Process {process_id} waiting for dependency {dep_id}")
                    self.queued_processes.append(process)
                    return process_id
        
        # Check if we can execute immediately
        if len(self.active_processes) < self.max_concurrent:
            asyncio.create_task(self._execute_process(process, input_data, parameters))
        else:
            self.logger.info(f"Queueing process {process_id}")
            self.queued_processes.append(process)
            
        return process_id
        
    async def _execute_process(self,
                             process: EthicalProcess,
                             input_data: Dict[str, Any],
                             parameters: Optional[Dict[str, Any]] = None):
        """Execute a single process with resource management."""
        process_id = process.process_id
        start_time = datetime.now()
        
        async with self.semaphore:
            try:
                self.active_processes[process_id] = process
                
                # Execute process
                result = await process.execute(input_data, parameters)
                
                # Store result
                self.completed_processes[process_id] = result
                
                # Record in history
                self._record_history(process_id, result, start_time)
                
                # Check for dependent processes
                await self._check_dependencies(process_id)
                
                # Monitor for drift if enabled
                if self.drift_monitoring:
                    await self._monitor_process_drift(process_id, result)
                    
            except Exception as e:
                self.logger.error(f"Process {process_id} failed: {str(e)}")
                self._record_history(process_id, None, start_time, error=str(e))
                raise
            finally:
                # Cleanup
                if process_id in self.active_processes:
                    del self.active_processes[process_id]
                    
                # Schedule next process if any queued
                if self.queued_processes:
                    next_process = self.queued_processes.pop(0)
                    asyncio.create_task(
                        self._execute_process(next_process, input_data, parameters)
                    )
                    
    def _record_history(self, 
                       process_id: str, 
                       result: Optional[ProcessResult],
                       start_time: datetime,
                       error: Optional[str] = None):
        """Record process execution in history."""
        record = {
            'process_id': process_id,
            'start_time': start_time,
            'end_time': datetime.now(),
            'success': result.success if result else False,
            'error': error
        }
        
        if result:
            record.update({
                'execution_time': result.execution_time,
                'warnings': result.warnings,
                'metrics': result.metrics
            })
            
        self.process_history.append(record)
                    
    async def _check_dependencies(self, completed_process_id: str):
        """Check and schedule processes waiting on dependencies."""
        to_schedule = []
        
        for queued in self.queued_processes[:]:
            if queued.process_id in self.dependencies:
                deps = self.dependencies[queued.process_id]
                if completed_process_id in deps:
                    deps.remove(completed_process_id)
                    if not deps:  # All dependencies completed
                        to_schedule.append(queued)
                        self.queued_processes.remove(queued)
                        
        for process in to_schedule:
            if len(self.active_processes) < self.max_concurrent:
                self.logger.info(f"Scheduling dependent process {process.process_id}")
                asyncio.create_task(self._execute_process(process, {}, None))
            else:
                self.queued_processes.append(process)
                
    async def _monitor_process_drift(self,
                                   process_id: str,
                                   result: ProcessResult):
        """Monitor process execution for drift patterns."""
        if not self.drift_monitoring:
            return
            
        try:
            # Extract relevant metrics for drift analysis
            metrics = {
                'execution_time': result.execution_time,
                'success_rate': 1.0 if result.success else 0.0,
                **result.metrics
            }
            
            # Check for drift
            await self.drift_detector.check_drift(
                'process_metrics',
                metrics,
                metadata={'process_id': process_id}
            )
            
        except Exception as e:
            self.logger.error(f"Drift monitoring failed: {str(e)}")
            
    def _handle_drift_alert(self, alert):
        """Handle drift detection alerts."""
        self.logger.warning(
            f"Process drift detected: {alert.message}"
        )
        
    def get_process_status(self, process_id: str) -> Dict[str, Any]:
        """Get current status of a process."""
        if process_id in self.active_processes:
            process = self.active_processes[process_id]
            return {
                'state': process.state.value,
                'start_time': process.context.start_time,
                'is_active': True,
                'retry_count': process.retry_count
            }
        elif process_id in self.completed_processes:
            result = self.completed_processes[process_id]
            return {
                'state': ProcessState.COMPLETED.value,
                'start_time': result.context.start_time,
                'end_time': result.context.start_time + 
                           timedelta(seconds=result.execution_time),
                'is_active': False,
                'success': result.success,
                'metrics': result.metrics
            }
        else:
            for process in self.queued_processes:
                if process.process_id == process_id:
                    return {
                        'state': ProcessState.INITIALIZED.value,
                        'is_active': False,
                        'is_queued': True
                    }
        return None
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of process queue."""
        return {
            'active_count': len(self.active_processes),
            'queued_count': len(self.queued_processes),
            'completed_count': len(self.completed_processes),
            'active_processes': list(self.active_processes.keys()),
            'queued_processes': [p.process_id for p in self.queued_processes]
        }
        
    async def wait_for_process(self, process_id: str, 
                             timeout: Optional[float] = None) -> ProcessResult:
        """Wait for a specific process to complete."""
        start_time = datetime.now()
        
        while True:
            if process_id in self.completed_processes:
                return self.completed_processes[process_id]
                
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Process {process_id} did not complete within {timeout} seconds"
                    )
                    
            await asyncio.sleep(0.1)  # Avoid busy waiting
            
    def cleanup_completed(self, max_age: Optional[float] = None):
        """Cleanup old completed processes."""
        if not max_age:
            self.completed_processes.clear()
            return
            
        cutoff = datetime.now() - timedelta(seconds=max_age)
        to_remove = []
        
        for process_id, result in self.completed_processes.items():
            process_time = result.context.start_time + timedelta(seconds=result.execution_time)
            if process_time < cutoff:
                to_remove.append(process_id)
                
        for process_id in to_remove:
            del self.completed_processes[process_id]
            
    def get_process_history(self, 
                          limit: Optional[int] = None,
                          process_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get execution history of processes."""
        history = self.process_history
        
        if process_id:
            history = [h for h in history if h['process_id'] == process_id]
            
        if limit:
            history = history[-limit:]
            
        return history
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get overall performance metrics."""
        if not self.process_history:
            return {}
            
        total_processes = len(self.process_history)
        successful = sum(1 for h in self.process_history if h.get('success', False))
        execution_times = [h.get('execution_time', 0) for h in self.process_history]
        
        return {
            'success_rate': successful / total_processes,
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'total_processes': total_processes,
            'error_rate': (total_processes - successful) / total_processes
        }