"""Ad-hoc Tool system — declarative tool creation without Python subclassing.

Supports four executor types:
  - http    : Call external HTTP APIs
  - script  : Execute sandboxed Python code
  - template: Format string templates
  - composite: Chain other tools together
"""

from __future__ import annotations

import asyncio
import importlib
import json
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from geny_executor.tools.base import Tool, ToolContext, ToolResult


# ── Configuration dataclasses ──────────────────────────────


@dataclass
class HttpToolConfig:
    """HTTP API call tool configuration."""

    url: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    body_template: Optional[str] = None
    query_template: Optional[Dict[str, str]] = None
    response_path: Optional[str] = None
    timeout: int = 30
    auth_type: Optional[str] = None  # "bearer" | "api_key" | "basic"
    auth_config: Optional[Dict[str, str]] = None


@dataclass
class ScriptToolConfig:
    """Python script execution tool configuration."""

    code: str
    runtime: str = "python"
    timeout: int = 60
    allowed_modules: List[str] = field(
        default_factory=lambda: [
            "json",
            "re",
            "math",
            "datetime",
            "urllib.parse",
            "hashlib",
            "base64",
        ]
    )
    sandbox: bool = True


@dataclass
class TemplateToolConfig:
    """String template formatting tool configuration."""

    template: str
    output_format: str = "text"  # "text" | "json" | "markdown"


@dataclass
class CompositeStep:
    """A single step in a composite tool pipeline."""

    tool_name: str
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: str = "result"
    condition: Optional[str] = None


@dataclass
class CompositeToolConfig:
    """Composite tool: chains multiple tools together."""

    steps: List[CompositeStep] = field(default_factory=list)


# ── AdhocToolDefinition ────────────────────────────────────


@dataclass
class AdhocToolDefinition:
    """Declarative tool definition — create tools without Python code."""

    name: str
    description: str
    input_schema: Dict[str, Any]

    executor_type: str  # "http" | "script" | "template" | "composite"

    http_config: Optional[HttpToolConfig] = None
    script_config: Optional[ScriptToolConfig] = None
    template_config: Optional[TemplateToolConfig] = None
    composite_config: Optional[CompositeToolConfig] = None

    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    author: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "executor_type": self.executor_type,
            "tags": self.tags,
            "version": self.version,
            "author": self.author,
        }
        if self.http_config:
            d["http_config"] = {
                "url": self.http_config.url,
                "method": self.http_config.method,
                "headers": self.http_config.headers,
                "body_template": self.http_config.body_template,
                "query_template": self.http_config.query_template,
                "response_path": self.http_config.response_path,
                "timeout": self.http_config.timeout,
                "auth_type": self.http_config.auth_type,
                "auth_config": self.http_config.auth_config,
            }
        if self.script_config:
            d["script_config"] = {
                "code": self.script_config.code,
                "runtime": self.script_config.runtime,
                "timeout": self.script_config.timeout,
                "allowed_modules": self.script_config.allowed_modules,
                "sandbox": self.script_config.sandbox,
            }
        if self.template_config:
            d["template_config"] = {
                "template": self.template_config.template,
                "output_format": self.template_config.output_format,
            }
        if self.composite_config:
            d["composite_config"] = {
                "steps": [
                    {
                        "tool_name": s.tool_name,
                        "input_mapping": s.input_mapping,
                        "output_key": s.output_key,
                        "condition": s.condition,
                    }
                    for s in self.composite_config.steps
                ]
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdhocToolDefinition:
        """Deserialize from dict."""
        http_cfg = None
        if "http_config" in data and data["http_config"]:
            hc = data["http_config"]
            http_cfg = HttpToolConfig(
                url=hc["url"],
                method=hc.get("method", "POST"),
                headers=hc.get("headers", {}),
                body_template=hc.get("body_template"),
                query_template=hc.get("query_template"),
                response_path=hc.get("response_path"),
                timeout=hc.get("timeout", 30),
                auth_type=hc.get("auth_type"),
                auth_config=hc.get("auth_config"),
            )

        script_cfg = None
        if "script_config" in data and data["script_config"]:
            sc = data["script_config"]
            script_cfg = ScriptToolConfig(
                code=sc["code"],
                runtime=sc.get("runtime", "python"),
                timeout=sc.get("timeout", 60),
                allowed_modules=sc.get(
                    "allowed_modules",
                    ["json", "re", "math", "datetime", "urllib.parse", "hashlib", "base64"],
                ),
                sandbox=sc.get("sandbox", True),
            )

        template_cfg = None
        if "template_config" in data and data["template_config"]:
            tc = data["template_config"]
            template_cfg = TemplateToolConfig(
                template=tc["template"],
                output_format=tc.get("output_format", "text"),
            )

        composite_cfg = None
        if "composite_config" in data and data["composite_config"]:
            cc = data["composite_config"]
            steps = [
                CompositeStep(
                    tool_name=s["tool_name"],
                    input_mapping=s.get("input_mapping", {}),
                    output_key=s.get("output_key", "result"),
                    condition=s.get("condition"),
                )
                for s in cc.get("steps", [])
            ]
            composite_cfg = CompositeToolConfig(steps=steps)

        return cls(
            name=data["name"],
            description=data["description"],
            input_schema=data["input_schema"],
            executor_type=data["executor_type"],
            http_config=http_cfg,
            script_config=script_cfg,
            template_config=template_cfg,
            composite_config=composite_cfg,
            tags=data.get("tags", []),
            version=data.get("version", "1.0"),
            author=data.get("author", ""),
        )


# ── Executor ABC ────────────────────────────────────────────


class AdhocExecutor(ABC):
    """Base class for ad-hoc tool executors."""

    def __init__(self, definition: AdhocToolDefinition):
        self.definition = definition

    @abstractmethod
    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        ...


# ── Executor implementations ────────────────────────────────


class HttpAdhocExecutor(AdhocExecutor):
    """Execute by calling an external HTTP API."""

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        config = self.definition.http_config
        if not config:
            return ToolResult(content="No http_config provided", is_error=True)

        try:
            import aiohttp
        except ImportError:
            return ToolResult(
                content="aiohttp not installed. Install with: pip install aiohttp",
                is_error=True,
            )

        url = self._render(config.url, input)
        headers = {k: self._render(v, input) for k, v in config.headers.items()}

        if config.auth_type == "bearer" and config.auth_config:
            headers["Authorization"] = f"Bearer {config.auth_config.get('token', '')}"
        elif config.auth_type == "api_key" and config.auth_config:
            key_name = config.auth_config.get("header", "X-API-Key")
            headers[key_name] = config.auth_config.get("key", "")

        body = None
        if config.body_template:
            body = json.loads(self._render(config.body_template, input))
        elif config.method in ("POST", "PUT", "PATCH"):
            body = input

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    config.method,
                    url,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=config.timeout),
                ) as resp:
                    data = await resp.json()

                    if config.response_path:
                        data = self._extract_path(data, config.response_path)

                    return ToolResult(
                        content=json.dumps(data, ensure_ascii=False)
                        if isinstance(data, (dict, list))
                        else str(data)
                    )
        except Exception as e:
            return ToolResult(content=f"HTTP error: {e}", is_error=True)

    @staticmethod
    def _render(template: str, data: Dict[str, Any]) -> str:
        """Safe string formatting using str.format_map."""
        try:
            return template.format_map(data)
        except (KeyError, IndexError):
            return template

    @staticmethod
    def _extract_path(data: Any, path: str) -> Any:
        """Simple dot-notation path extraction (e.g. 'data.results.0.text')."""
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, current)
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return current
            else:
                return current
        return current


class ScriptAdhocExecutor(AdhocExecutor):
    """Execute a sandboxed Python script."""

    # Safe subset of builtins
    _SAFE_BUILTINS = {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter",
        "float", "format", "frozenset", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "print",
        "range", "repr", "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "type", "zip",
        "True", "False", "None",
    }

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        config = self.definition.script_config
        if not config:
            return ToolResult(content="No script_config provided", is_error=True)

        if config.sandbox:
            return await self._execute_sandboxed(input, context, config)
        else:
            return await self._execute_direct(input, context, config)

    async def _execute_sandboxed(
        self, input: Dict, context: ToolContext, config: ScriptToolConfig
    ) -> ToolResult:
        """Execute with restricted builtins and allowed modules only."""
        import builtins

        safe_builtins = {
            k: getattr(builtins, k)
            for k in self._SAFE_BUILTINS
            if hasattr(builtins, k)
        }
        safe_builtins["__import__"] = self._make_restricted_import(
            config.allowed_modules
        )

        restricted_globals: Dict[str, Any] = {"__builtins__": safe_builtins}
        for mod_name in config.allowed_modules:
            try:
                restricted_globals[mod_name.split(".")[-1]] = importlib.import_module(
                    mod_name
                )
            except ImportError:
                pass

        restricted_globals["input"] = input
        restricted_globals["context"] = {
            "session_id": context.session_id,
            "working_dir": context.working_dir,
        }

        try:
            exec(compile(config.code, "<adhoc>", "exec"), restricted_globals)
            execute_fn = restricted_globals.get("execute")
            if not execute_fn:
                return ToolResult(
                    content="Error: execute(input, context) function not defined",
                    is_error=True,
                )

            result = await asyncio.wait_for(
                execute_fn(input, restricted_globals.get("context", {})),
                timeout=config.timeout,
            )
            return ToolResult(
                content=json.dumps(result, ensure_ascii=False, default=str)
                if isinstance(result, (dict, list))
                else str(result)
            )
        except asyncio.TimeoutError:
            return ToolResult(
                content=f"Script timed out after {config.timeout}s", is_error=True
            )
        except Exception as e:
            return ToolResult(content=f"Script error: {e}", is_error=True)

    async def _execute_direct(
        self, input: Dict, context: ToolContext, config: ScriptToolConfig
    ) -> ToolResult:
        """Execute without sandbox (for trusted scripts)."""
        exec_globals: Dict[str, Any] = {"input": input, "context": context}
        try:
            exec(compile(config.code, "<adhoc>", "exec"), exec_globals)
            execute_fn = exec_globals.get("execute")
            if not execute_fn:
                return ToolResult(
                    content="Error: execute(input, context) function not defined",
                    is_error=True,
                )
            result = await asyncio.wait_for(
                execute_fn(input, context),
                timeout=config.timeout,
            )
            return ToolResult(
                content=json.dumps(result, ensure_ascii=False, default=str)
                if isinstance(result, (dict, list))
                else str(result)
            )
        except asyncio.TimeoutError:
            return ToolResult(
                content=f"Script timed out after {config.timeout}s", is_error=True
            )
        except Exception as e:
            return ToolResult(content=f"Script error: {e}", is_error=True)

    @staticmethod
    def _make_restricted_import(allowed: List[str]):
        """Create a restricted __import__ that only allows specified modules."""

        def restricted_import(name, *args, **kwargs):
            if name not in allowed and not any(
                name.startswith(a + ".") for a in allowed
            ):
                raise ImportError(f"Import of '{name}' not allowed in sandbox")
            return importlib.import_module(name)

        return restricted_import


class TemplateAdhocExecutor(AdhocExecutor):
    """Format a string template with input variables."""

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        config = self.definition.template_config
        if not config:
            return ToolResult(content="No template_config provided", is_error=True)

        try:
            result = string.Template(config.template).safe_substitute(input)
            if config.output_format == "json":
                try:
                    parsed = json.loads(result)
                    result = json.dumps(parsed, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass
            return ToolResult(content=result)
        except Exception as e:
            return ToolResult(content=f"Template error: {e}", is_error=True)


class CompositeAdhocExecutor(AdhocExecutor):
    """Chain multiple tools together."""

    def __init__(self, definition: AdhocToolDefinition, tool_resolver=None):
        super().__init__(definition)
        self._tool_resolver = tool_resolver  # Callable[[str] -> Tool]

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        config = self.definition.composite_config
        if not config:
            return ToolResult(content="No composite_config provided", is_error=True)

        if not self._tool_resolver:
            return ToolResult(
                content="Composite tool has no tool_resolver", is_error=True
            )

        state: Dict[str, Any] = {"input": input}

        for step in config.steps:
            # Check condition
            if step.condition:
                try:
                    if not eval(step.condition, {"__builtins__": {}}, state):
                        continue
                except Exception:
                    continue

            # Resolve tool
            tool = self._tool_resolver(step.tool_name)
            if tool is None:
                return ToolResult(
                    content=f"Tool '{step.tool_name}' not found", is_error=True
                )

            # Map input
            step_input: Dict[str, Any] = {}
            for out_key, expr in step.input_mapping.items():
                try:
                    step_input[out_key] = eval(expr, {"__builtins__": {}}, state)
                except Exception:
                    step_input[out_key] = expr

            if not step_input:
                step_input = input

            # Execute
            result = await tool.execute(step_input, context)
            state[step.output_key] = (
                json.loads(result.content)
                if isinstance(result.content, str)
                and result.content.startswith(("{", "["))
                else result.content
            )

            if result.is_error:
                return result

        # Return last step result
        last_key = config.steps[-1].output_key if config.steps else "result"
        final = state.get(last_key, "")
        return ToolResult(
            content=json.dumps(final, ensure_ascii=False, default=str)
            if isinstance(final, (dict, list))
            else str(final)
        )


# ── AdhocTool wrapper ────────────────────────────────────


class AdhocTool(Tool):
    """A Tool instance created from an AdhocToolDefinition."""

    def __init__(
        self,
        definition: AdhocToolDefinition,
        executor: AdhocExecutor,
    ):
        self._definition = definition
        self._executor = executor

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def description(self) -> str:
        return self._definition.description

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._definition.input_schema

    @property
    def definition(self) -> AdhocToolDefinition:
        return self._definition

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return await self._executor.execute(input, context)


# ── Factory ──────────────────────────────────────────────


class AdhocToolFactory:
    """Create executable Tool instances from AdhocToolDefinitions."""

    _executors: Dict[str, Type[AdhocExecutor]] = {
        "http": HttpAdhocExecutor,
        "script": ScriptAdhocExecutor,
        "template": TemplateAdhocExecutor,
        "composite": CompositeAdhocExecutor,
    }

    @classmethod
    def create(
        cls,
        definition: AdhocToolDefinition,
        tool_resolver=None,
    ) -> AdhocTool:
        """Create a Tool from a definition.

        Args:
            definition: The tool definition.
            tool_resolver: For composite tools, a callable(name) → Tool.
        """
        executor_cls = cls._executors.get(definition.executor_type)
        if executor_cls is None:
            raise ValueError(
                f"Unknown executor type: '{definition.executor_type}'. "
                f"Available: {list(cls._executors.keys())}"
            )

        if definition.executor_type == "composite":
            executor = executor_cls(definition, tool_resolver=tool_resolver)
        else:
            executor = executor_cls(definition)

        return AdhocTool(definition=definition, executor=executor)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], tool_resolver=None) -> AdhocTool:
        """Create from a serialized dict."""
        definition = AdhocToolDefinition.from_dict(data)
        return cls.create(definition, tool_resolver=tool_resolver)
