"""SSH built-in tools + per-session credential store.

Covers the security-critical invariants (secrets never surface to the agent;
password fed to sudo on stdin; the agent addresses servers by name only), the
file-backed per-session store, connection-kwargs assembly for password + key
auth, and the tool output/error shaping. Real network connects are mocked —
end-to-end auth is exercised at deploy time against a live server.
"""

from __future__ import annotations

import json

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools._ssh import (
    SSHConfigError,
    _connect_kwargs,
    _remote_command,
)
from geny_executor.tools.built_in._ssh_store import SSHServerStore
from geny_executor.tools.built_in import ssh_tools
from geny_executor.tools.built_in.ssh_tools import (
    SshDownloadTool,
    SshListServersTool,
    SshRunTool,
    SshUploadTool,
)


def _ctx(tmp_path, servers=None):
    extras = {"ssh": {"servers": servers}} if servers is not None else {}
    return ToolContext(session_id="s1", storage_path=str(tmp_path), extras=extras)


_PW_SERVER = {"name": "prod", "host": "1.2.3.4", "port": 2222, "user": "hrjang", "password": "secret", "description": "web"}


# ── SSHServerStore ───────────────────────────────────────────────────

def test_store_from_extras_writes_per_session_file(tmp_path):
    store = SSHServerStore.from_context(_ctx(tmp_path, [_PW_SERVER]))
    # Persisted to <storage_path>/ssh/servers.json (the "파일형태" record).
    f = tmp_path / "ssh" / "servers.json"
    assert f.is_file()
    assert json.loads(f.read_text())[0]["name"] == "prod"
    assert oct(f.stat().st_mode)[-3:] == "600"  # secrets → owner-only


def test_store_reads_file_when_no_injection(tmp_path):
    (tmp_path / "ssh").mkdir()
    (tmp_path / "ssh" / "servers.json").write_text(json.dumps([_PW_SERVER]))
    store = SSHServerStore.from_context(_ctx(tmp_path))  # no extras
    assert store.resolve("prod")["host"] == "1.2.3.4"


def test_list_public_hides_all_secrets(tmp_path):
    key_server = {"name": "k", "host": "h", "user": "u", "private_key": "PEM", "passphrase": "pp"}
    store = SSHServerStore.from_context(_ctx(tmp_path, [_PW_SERVER, key_server]))
    pub = store.list_public()
    blob = json.dumps(pub)
    assert "secret" not in blob and "PEM" not in blob and "pp" not in blob
    by = {s["name"]: s for s in pub}
    assert by["prod"]["auth"] == "password" and by["prod"]["port"] == 2222
    assert by["k"]["auth"] == "key"


def test_resolve_returns_full_record_with_secret(tmp_path):
    store = SSHServerStore.from_context(_ctx(tmp_path, [_PW_SERVER]))
    assert store.resolve("prod")["password"] == "secret"
    assert store.resolve("missing") is None


# ── _connect_kwargs / _remote_command ────────────────────────────────

def test_connect_kwargs_password():
    host, kw = _connect_kwargs(_PW_SERVER, connect_timeout=15)
    assert host == "1.2.3.4"
    assert kw["port"] == 2222 and kw["username"] == "hrjang"
    assert kw["password"] == "secret"
    assert kw["known_hosts"] is None  # relaxed host-key by default


def test_connect_kwargs_strict_host_key_opts_in():
    _, kw = _connect_kwargs({**_PW_SERVER, "strict_host_key": True}, connect_timeout=15)
    assert "known_hosts" not in kw


def test_connect_kwargs_private_key():
    import asyncssh

    pem = asyncssh.generate_private_key("ssh-rsa").export_private_key().decode()
    _, kw = _connect_kwargs(
        {"name": "k", "host": "h", "user": "u", "private_key": pem}, connect_timeout=15
    )
    assert "client_keys" in kw and len(kw["client_keys"]) == 1


def test_connect_kwargs_errors():
    with pytest.raises(SSHConfigError):
        _connect_kwargs({"user": "u", "password": "p"}, connect_timeout=15)  # no host
    with pytest.raises(SSHConfigError):
        _connect_kwargs({"host": "h", "user": "u"}, connect_timeout=15)  # no creds


def test_remote_command_plain_cwd_sudo():
    assert _remote_command("ls", cwd=None, sudo=False, password=None) == ("ls", None)
    cmd, inp = _remote_command("ls", cwd="/var/log", sudo=False, password=None)
    assert cmd == "cd /var/log && ls" and inp is None
    cmd, inp = _remote_command("whoami", cwd=None, sudo=True, password="pw")
    assert cmd.startswith("sudo -S -p '' /bin/sh -c ") and inp == "pw\n"


# ── tools ────────────────────────────────────────────────────────────

def test_ssh_tools_are_feature_gated():
    assert SshRunTool().required_config_keys() == ["feature:ssh_enabled"]
    assert SshListServersTool().required_config_keys() == ["feature:ssh_enabled"]


@pytest.mark.asyncio
async def test_list_servers_populated_and_empty(tmp_path):
    res = await SshListServersTool().execute({}, _ctx(tmp_path, [_PW_SERVER]))
    assert not res.is_error
    assert res.content["servers"][0]["name"] == "prod"
    assert "secret" not in json.dumps(res.content) and "secret" not in (res.display_text or "")

    empty = await SshListServersTool().execute({}, _ctx(tmp_path / "e", []))
    assert empty.content["servers"] == []


@pytest.mark.asyncio
async def test_ssh_run_unknown_server_and_no_command(tmp_path):
    ctx = _ctx(tmp_path, [_PW_SERVER])
    r1 = await SshRunTool().execute({"server": "nope", "command": "ls"}, ctx)
    assert r1.is_error and r1.content["error"]["code"] == "UNKNOWN_SERVER"
    r2 = await SshRunTool().execute({"server": "prod", "command": "  "}, ctx)
    assert r2.is_error and r2.content["error"]["code"] == "NO_COMMAND"


@pytest.mark.asyncio
async def test_ssh_run_sudo_requires_password(tmp_path):
    key_only = {"name": "k", "host": "h", "user": "u", "private_key": "x"}
    ctx = _ctx(tmp_path, [key_only])
    r = await SshRunTool().execute({"server": "k", "command": "id", "sudo": True}, ctx)
    assert r.is_error and r.content["error"]["code"] == "NO_SUDO_PASSWORD"


@pytest.mark.asyncio
async def test_ssh_run_happy_path_shapes_output(tmp_path, monkeypatch):
    captured = {}

    async def fake_exec(server, command, *, timeout, cwd, sudo, **kw):
        captured.update(server=server, command=command, sudo=sudo)
        return 0, "hello\n", ""

    monkeypatch.setattr(ssh_tools, "ssh_exec", fake_exec)
    res = await SshRunTool().execute(
        {"server": "prod", "command": "echo hello", "sudo": True}, _ctx(tmp_path, [_PW_SERVER])
    )
    assert not res.is_error
    assert res.content["exit_code"] == 0 and res.content["stdout"] == "hello\n"
    assert res.metadata["server"] == "prod"
    # The full record (with password) was handed to the connector, not the agent.
    assert captured["server"]["password"] == "secret" and captured["sudo"] is True
    assert "secret" not in (res.display_text or "")


@pytest.mark.asyncio
async def test_ssh_run_nonzero_is_error(tmp_path, monkeypatch):
    async def fake_exec(server, command, *, timeout, cwd, sudo, **kw):
        return 1, "", "boom"

    monkeypatch.setattr(ssh_tools, "ssh_exec", fake_exec)
    res = await SshRunTool().execute(
        {"server": "prod", "command": "false"}, _ctx(tmp_path, [_PW_SERVER])
    )
    assert res.is_error and res.content["exit_code"] == 1 and "boom" in res.display_text


@pytest.mark.asyncio
async def test_upload_path_guard_and_missing(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, [_PW_SERVER])
    # escape attempt
    esc = await SshUploadTool().execute(
        {"server": "prod", "local_path": "../../etc/passwd", "remote_path": "/tmp/x"}, ctx
    )
    assert esc.is_error and esc.content["error"]["code"] == "PATH_ESCAPE"
    # missing local file
    miss = await SshUploadTool().execute(
        {"server": "prod", "local_path": "nope.txt", "remote_path": "/tmp/x"}, ctx
    )
    assert miss.is_error and miss.content["error"]["code"] == "NOT_FOUND"
    # happy path (mock sftp)
    (tmp_path / "f.txt").write_text("data")

    async def fake_put(server, local, remote, **kw):
        assert remote == "/tmp/x"

    monkeypatch.setattr(ssh_tools, "sftp_put", fake_put)
    ok = await SshUploadTool().execute(
        {"server": "prod", "local_path": "f.txt", "remote_path": "/tmp/x"}, ctx
    )
    assert not ok.is_error and ok.content["uploaded"] == "/tmp/x"
