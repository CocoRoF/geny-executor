"""Phase 7 — Security module tests.

Tests:
  - SSRF protection (URL validation, blocked IPs, schemes)
  - Script sandbox (AST-level forbidden imports, calls, attributes)
  - Import validator (size, stage, tool limits, script checks, MCP commands)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from geny_executor.security import SSRFError, validate_url, is_safe_url
from geny_executor.security.script_sandbox import (
    validate_script,
    check_script,
    ScriptSecurityError,
    FORBIDDEN_MODULES,
)
from geny_executor.security.import_validator import (
    validate_import,
    check_import,
    ImportValidationError,
    MAX_STAGES,
    MAX_ADHOC_TOOLS,
    MAX_MCP_SERVERS,
    MAX_SCRIPT_LENGTH,
)


# ═══════════════════════════════════════════════════════════
# SSRF Protection
# ═══════════════════════════════════════════════════════════


class TestSSRFValidation:
    def test_blocked_scheme_ftp(self):
        with pytest.raises(SSRFError, match="Blocked scheme"):
            validate_url("ftp://example.com/file")

    def test_blocked_scheme_file(self):
        with pytest.raises(SSRFError, match="Blocked scheme"):
            validate_url("file:///etc/passwd")

    def test_no_hostname(self):
        with pytest.raises(SSRFError, match="no hostname"):
            validate_url("http://")

    def test_localhost_blocked(self):
        with pytest.raises(SSRFError, match="Blocked"):
            validate_url("http://127.0.0.1/admin")

    def test_localhost_name_blocked(self):
        with pytest.raises(SSRFError, match="Blocked"):
            validate_url("http://localhost/admin")

    def test_private_10_blocked(self):
        with pytest.raises(SSRFError, match="Blocked"):
            validate_url("http://10.0.0.1/api")

    def test_private_172_blocked(self):
        with pytest.raises(SSRFError, match="Blocked"):
            validate_url("http://172.16.0.1/api")

    def test_private_192_blocked(self):
        with pytest.raises(SSRFError, match="Blocked"):
            validate_url("http://192.168.1.1/api")

    def test_link_local_blocked(self):
        with pytest.raises(SSRFError, match="Blocked"):
            validate_url("http://169.254.169.254/latest/meta-data/")

    def test_ipv6_loopback_blocked(self):
        with pytest.raises(SSRFError, match="Blocked"):
            validate_url("http://[::1]/api")

    def test_public_url_allowed(self):
        # github.com is a known public domain
        result = validate_url("https://api.github.com/repos")
        assert result == "https://api.github.com/repos"

    def test_is_safe_url_convenience(self):
        assert is_safe_url("https://example.com") is True
        assert is_safe_url("http://127.0.0.1") is False
        assert is_safe_url("ftp://evil.com") is False
        assert is_safe_url("not-a-url") is False


# ═══════════════════════════════════════════════════════════
# Script Sandbox
# ═══════════════════════════════════════════════════════════


class TestScriptSandboxImports:
    def test_import_os_blocked(self):
        v = validate_script("import os")
        assert any("os" in e for e in v)

    def test_import_subprocess_blocked(self):
        v = validate_script("import subprocess")
        assert any("subprocess" in e for e in v)

    def test_from_os_import_blocked(self):
        v = validate_script("from os import path")
        assert any("os" in e for e in v)

    def test_import_socket_blocked(self):
        v = validate_script("import socket")
        assert any("socket" in e for e in v)

    def test_safe_import_allowed(self):
        v = validate_script("import json\nimport math\nimport re")
        assert len(v) == 0

    def test_all_forbidden_modules_blocked(self):
        for mod in ["os", "sys", "subprocess", "pickle", "ctypes"]:
            v = validate_script(f"import {mod}")
            assert len(v) > 0, f"Module {mod} should be blocked"


class TestScriptSandboxCalls:
    def test_exec_blocked(self):
        v = validate_script("exec('print(1)')")
        assert any("exec" in e for e in v)

    def test_eval_blocked(self):
        v = validate_script("x = eval('1+1')")
        assert any("eval" in e for e in v)

    def test_compile_blocked(self):
        v = validate_script("compile('pass', '<x>', 'exec')")
        assert any("compile" in e for e in v)

    def test_dunder_import_blocked(self):
        v = validate_script("__import__('os')")
        assert any("__import__" in e for e in v)

    def test_open_blocked(self):
        v = validate_script("f = open('/etc/passwd')")
        assert any("open" in e for e in v)

    def test_print_allowed(self):
        v = validate_script("print('hello')")
        assert len(v) == 0


class TestScriptSandboxAttributes:
    def test_subclasses_blocked(self):
        v = validate_script("x.__subclasses__()")
        assert any("__subclasses__" in e for e in v)

    def test_globals_dunder_blocked(self):
        v = validate_script("x.__globals__")
        assert any("__globals__" in e for e in v)

    def test_builtins_dunder_blocked(self):
        v = validate_script("x.__builtins__")
        assert any("__builtins__" in e for e in v)


class TestScriptSandboxMisc:
    def test_syntax_error(self):
        v = validate_script("def incomplete(")
        assert any("Syntax error" in e for e in v)

    def test_safe_script(self):
        code = """
result = sum(range(10))
data = [x * 2 for x in range(5)]
output = str(result)
"""
        v = validate_script(code)
        assert len(v) == 0

    def test_check_script_raises(self):
        with pytest.raises(ScriptSecurityError):
            check_script("import os")

    def test_check_script_ok(self):
        check_script("x = 1 + 2")  # should not raise


# ═══════════════════════════════════════════════════════════
# Import Validator
# ═══════════════════════════════════════════════════════════


class TestImportValidator:
    def test_valid_import(self):
        data = {
            "version": "1.0",
            "stages": [{"order": i} for i in range(3)],
            "tools": {"adhoc": [], "mcp_servers": []},
        }
        errors = validate_import(data)
        assert len(errors) == 0

    def test_file_too_large(self):
        errors = validate_import({}, raw_size=20 * 1024 * 1024)
        assert any("too large" in e.lower() for e in errors)

    def test_unsupported_version(self):
        errors = validate_import({"version": "99.0"})
        assert any("version" in e.lower() for e in errors)

    def test_too_many_stages(self):
        data = {"stages": [{"order": i} for i in range(MAX_STAGES + 5)]}
        errors = validate_import(data)
        assert any("stages" in e.lower() for e in errors)

    def test_too_many_adhoc_tools(self):
        data = {"tools": {"adhoc": [{"name": f"t{i}"} for i in range(MAX_ADHOC_TOOLS + 1)]}}
        errors = validate_import(data)
        assert any("adhoc" in e.lower() for e in errors)

    def test_too_many_mcp_servers(self):
        data = {"tools": {"mcp_servers": [{"name": f"s{i}"} for i in range(MAX_MCP_SERVERS + 1)]}}
        errors = validate_import(data)
        assert any("mcp" in e.lower() for e in errors)

    def test_script_too_long(self):
        data = {"tools": {"adhoc": [{
            "name": "big_script",
            "executor_type": "script",
            "script_config": {"code": "x = 1\n" * (MAX_SCRIPT_LENGTH + 1)},
        }]}}
        errors = validate_import(data)
        assert any("too long" in e.lower() for e in errors)

    def test_script_security_violation(self):
        data = {"tools": {"adhoc": [{
            "name": "evil_tool",
            "executor_type": "script",
            "script_config": {"code": "import os\nos.system('rm -rf /')"},
        }]}}
        errors = validate_import(data)
        assert any("security" in e.lower() for e in errors)

    def test_dangerous_mcp_command(self):
        data = {"tools": {"mcp_servers": [{
            "name": "danger",
            "transport": "stdio",
            "command": "rm",
        }]}}
        errors = validate_import(data)
        assert any("dangerous" in e.lower() for e in errors)

    def test_safe_mcp_command(self):
        data = {"tools": {"mcp_servers": [{
            "name": "github",
            "transport": "stdio",
            "command": "npx",
        }]}}
        errors = validate_import(data)
        assert len(errors) == 0

    def test_check_import_raises(self):
        with pytest.raises(ImportValidationError):
            check_import({"version": "99.0"})

    def test_check_import_ok(self):
        check_import({"version": "1.0", "stages": []})  # no raise
