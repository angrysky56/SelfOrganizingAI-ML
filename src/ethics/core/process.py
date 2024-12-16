"""
Process Module for ethical analysis workflows
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import asyncio
import logging

class ProcessState(Enum):
    """States for ethical process workflow."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_APPROVAL = "waiting_for_approval"

@dataclass
class ProcessContext:
    """Context information for process execution."""
    process_id: str
    start_time: datetime
    parameters: Dict[str, Any]
    state: ProcessState
    metadata: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None

@dataclass
class ProcessResult:
    """Results from process execution."""
    success: bool
    output: Any
    execution_time: float
    warnings: List[str]
    metrics: Dict[str, float]
    context: ProcessContext

class EthicalProcess:
    """Base class for ethical analysis processes."""
    
    def __init__(self, 
                 process_id: str,
                 approval_required: bool = False,
                 auto_retry: bool = True,
                 max_retries: int = 3,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize ethical process.
        
        Args:
            process_id: Unique identifier for the process
            approval_required: Whether manual approval is needed
            auto_retry: Whether to automatically retry on failure
            max_retries: Maximum number of retry attempts
            logger: Optional logger instance
        """
        self.process_id = process_id
        self.approval_required = approval_required
        self.auto_retry = auto_retry
        self.max_retries = max_retries
        
        self.logger = logger or logging.getLogger(__name__)
        self.state = ProcessState.INITIALIZED
        self.retry_count = 0
        self.approval_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
        # Initialize empty context
        self.context = ProcessContext(
            process_id=process_id,
            start_time=datetime.now(),
            parameters={},
            state=self.state
        )

    async def execute(self, 
                     input_data: Dict[str, Any],
                     parameters: Optional[Dict[str, Any]] = None) -> ProcessResult:
        """
        Execute the ethical process.
        
        Args:
            input_data: Input data for process
            parameters: Optional execution parameters
            
        Returns:
            ProcessResult containing execution results
        """
        start_time = datetime.now()
        self.context.parameters = parameters or {}
        self.context.start_time = start_time
        warnings = []
        
        try:
            # Validate inputs
            self._validate_inputs(input_data)
            
            # Check if approval is needed
            if self.approval_required and self.state != ProcessState.WAITING_FOR_APPROVAL:
                self.state = ProcessState.WAITING_FOR_APPROVAL
                await self._request_approval()
            
            # Execute process
            self.state = ProcessState.RUNNING
            output = await self._execute_internal(input_data)
            
            # Update state
            self.state = ProcessState.COMPLETED
            success = True
            
        except Exception as e:
            self.logger.error(f"Process {self.process_id} failed: {str(e)}")
            warnings.append(str(e))
            
            if self.auto_retry and self.retry_count < self.max_retries:
                self.retry_count += 1
                self.logger.info(f"Retrying process {self.process_id}, attempt {self.retry_count}")
                return await self.execute(input_data, parameters)
            
            self.state = ProcessState.FAILED
            success = False
            output = None
            
        finally:
            execution_time = (datetime.now() - start_time).total_seconds()
            
        # Prepare result
        result = ProcessResult(
            success=success,
            output=output,
            execution_time=execution_time,
            warnings=warnings,
            metrics=self._compute_metrics(),
            context=self.context
        )
        
        # Notify completion callbacks
        for callback in self.completion_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Completion callback failed: {str(e)}")
        
        return result

    async def _execute_internal(self, input_data: Dict[str, Any]) -> Any:
        """Internal execution logic to be implemented by subclasses."""
        raise NotImplementedError
        
    def _validate_inputs(self, input_data: Dict[str, Any]):
        """Validate process inputs."""
        if not input_data:
            raise ValueError("Input data cannot be empty")
            
    async def _request_approval(self):
        """Request manual approval if needed."""
        for callback in self.approval_callbacks:
            try:
                approved = await callback(self.context)
                if not approved:
                    raise ValueError("Process not approved")
            except Exception as e:
                self.logger.error(f"Approval callback failed: {str(e)}")
                raise
                
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute process execution metrics."""
        return {
            'retry_count': self.retry_count,
            'execution_time': (datetime.now() - self.context.start_time).total_seconds()
        }
        
    def add_approval_callback(self, callback: Callable):
        """Add callback for approval requests."""
        self.approval_callbacks.append(callback)
        
    def add_completion_callback(self, callback: Callable):
        """Add callback for process completion."""
        self.completion_callbacks.append(callback)
        
    def pause(self):
        """Pause process execution."""
        if self.state == ProcessState.RUNNING:
            self.state = ProcessState.PAUSED
            
    def resume(self):
        """Resume paused process."""
        if self.state == ProcessState.PAUSED:
            self.state = ProcessState.RUNNING
            
    @property
    def is_active(self) -> bool:
        """Check if process is actively running."""
        return self.state in [ProcessState.RUNNING, ProcessState.WAITING_FOR_APPROVAL]
        
    @property
    def is_complete(self) -> bool:
        """Check if process has completed."""
        return self.state == ProcessState.COMPLETED
        
    @property
    def has_failed(self) -> bool:
        """Check if process has failed."""
        return self.state == ProcessState.FAILED