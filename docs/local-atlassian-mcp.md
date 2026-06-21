# Local Atlassian MCP Setup

This project can use the Atlassian MCP server as a local Codex developer tool.
It is not a Poly Chat runtime dependency.

## Install

Install the MCP server as a user-level `uv` tool:

```bash
uv tool install mcp-atlassian==0.21.1
```

If `uv` is not installed:

```bash
curl -L https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
sh /tmp/uv-install.sh
```

## Credentials

Copy the example environment file and fill in real Jira credentials:

```bash
cp .env.example .env
```

The `.env` file is ignored by git.

## Codex MCP Config

Use the template in `.codex/config.example.toml`, replacing paths for your
machine:

```toml
[mcp_servers.mcp-atlassian]
command = "/home/YOUR_USER/.local/bin/mcp-atlassian"
args = ["--env-file", "/absolute/path/to/poly-chat/.env"]
default_tools_approval_mode = "prompt"
```

Restart Codex after changing MCP config.
