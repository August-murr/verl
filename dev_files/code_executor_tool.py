# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CodeExecutorTool(BaseTool):
    """A utility tool for executing Python code using E2B sandbox.
    
    This tool acts like a calculator - it simply executes code and returns results.
    It does not provide task-specific feedback or rewards.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "A tool for executing Python code in a secure sandbox",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Import E2B here to avoid import errors if not installed
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            raise ImportError(
                "e2b_code_interpreter is required for CodeExecutorTool. "
                "Install it with: pip install e2b-code-interpreter"
            )
        
        # Create sandbox instance
        sandbox = Sandbox.create()
        
        self._instance_dict[instance_id] = {
            "sandbox": sandbox,
            "execution_count": 0,
        }
        return instance_id, ToolResponse(text="Code execution environment initialized")

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        
        if not code.strip():
            return ToolResponse(text="Error: Empty code provided"), 0.0, {"error": "empty_code"}
        
        instance = self._instance_dict[instance_id]
        sandbox = instance["sandbox"]
        
        try:
            # Execute the code in the sandbox
            execution = sandbox.run_code(code)
            
            # Prepare response
            response_parts = []
            
            # Add stdout if available
            if execution.logs.stdout:
                response_parts.append(f"Output:\n{execution.logs.stdout}")
            
            # Add stderr if available
            if execution.logs.stderr:
                response_parts.append(f"Error:\n{execution.logs.stderr}")
            
            # Add execution error if available
            if execution.error:
                response_parts.append(f"Execution Error: {execution.error}")
            
            # Update instance state (just for tracking)
            instance["execution_count"] += 1
            
            # Combine response parts
            if response_parts:
                response_text = "\n\n".join(response_parts)
            else:
                response_text = "Code executed successfully (no output)"
            
            # Prepare metrics for debugging/logging
            metrics = {
                "execution_count": instance["execution_count"],
                "has_stdout": bool(execution.logs.stdout),
                "has_stderr": bool(execution.logs.stderr),
                "has_error": execution.error is not None,
            }
            
            # No reward - this is just a utility tool
            return ToolResponse(text=response_text), 0.0, metrics
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error during code execution: {str(e)}"
            logger.error(error_msg)
            
            instance["execution_count"] += 1
            
            return ToolResponse(text=error_msg), 0.0, {"error": "unexpected_error"}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """No reward calculation needed - this is a utility tool"""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the sandbox and clean up instance"""
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            sandbox = instance["sandbox"]
            
            try:
                # Kill the sandbox to free resources
                sandbox.kill()
            except Exception as e:
                logger.warning(f"Error killing sandbox for instance {instance_id}: {e}")
            
            # Remove from instance dict
            del self._instance_dict[instance_id]