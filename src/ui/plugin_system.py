from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable

# Approved read-only permissions for UI plugins
PERMITTED_PLUGIN_PERMISSIONS = {
    "read_market_scanner",
    "read_portfolio_balance",
    "read_model_predictions",
    "read_risk_metrics",
    "render_custom_widget",
}

# Forbidden permissions that plugins must NEVER be granted
FORBIDDEN_PERMISSIONS = {
    "access_credentials",
    "submit_live_orders",
    "bypass_risk_engine",
    "modify_authoritative_ledger",
}

@dataclass
class PluginManifest:
    plugin_id: str
    name: str
    version: str
    author: str
    requested_permissions: List[str]
    is_sandboxed: bool = True

class PluginSandbox:
    """
    Sandboxed Plugin Architecture for Extensible Developer Mode.
    Isolates third-party or user-created widgets and panels.
    Prevents direct execution of orders or access to secret stores.
    """
    def __init__(self):
        self.registered_plugins: Dict[str, PluginManifest] = {}
        self.active_plugins: Dict[str, bool] = {}
        self.workspaces: Dict[str, Dict[str, Any]] = {
            "Default": {
                "layout": ["scanner_panel", "chart_panel", "risk_monitor", "portfolio_balance"],
                "density": "dense",
            }
        }

    def register_plugin(self, manifest_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validates plugin manifest, rejects dangerous permissions."""
        plugin_id = manifest_dict.get("plugin_id")
        if not plugin_id or not isinstance(plugin_id, str) or not plugin_id.strip():
            return {"success": False, "reason": "Invalid or missing plugin_id in manifest"}
        plugin_id = plugin_id.strip()

        name = manifest_dict.get("name", "Unnamed Plugin")
        version = manifest_dict.get("version", "1.0.0")
        author = manifest_dict.get("author", "Unknown")
        req_perms = set(manifest_dict.get("requested_permissions", []))

        # Security check: Reject if any forbidden permission requested
        forbidden_found = req_perms.intersection(FORBIDDEN_PERMISSIONS)
        if forbidden_found:
            return {
                "success": False,
                "reason": f"Security violation: Plugin requested forbidden capabilities: {list(forbidden_found)}",
            }

        unauthorized_perms = req_perms - PERMITTED_PLUGIN_PERMISSIONS
        if unauthorized_perms:
            return {
                "success": False,
                "reason": f"Invalid permissions requested: {list(unauthorized_perms)}",
            }

        manifest = PluginManifest(
            plugin_id=plugin_id,
            name=name,
            version=version,
            author=author,
            requested_permissions=list(req_perms),
            is_sandboxed=True,
        )
        self.registered_plugins[plugin_id] = manifest
        self.active_plugins[plugin_id] = True

        return {"success": True, "plugin_id": plugin_id, "status": "REGISTERED_AND_ACTIVE"}

    def execute_plugin_safely(self, plugin_id: str, handler: Callable[[], Any]) -> Dict[str, Any]:
        """Executes plugin component with complete fault isolation."""
        if plugin_id not in self.registered_plugins or not self.active_plugins.get(plugin_id, False):
            return {"success": False, "error": f"Plugin {plugin_id} is inactive or not found"}

        try:
            result = handler()
            return {"success": True, "data": result}
        except Exception as exc:
            # Fault isolation: Disable misbehaving plugin without crashing host engine
            self.active_plugins[plugin_id] = False
            return {
                "success": False,
                "error": f"Plugin crashed and was automatically isolated: {type(exc).__name__}: {exc}",
            }

    def save_workspace_layout(self, workspace_name: str, layout_config: Dict[str, Any], config_path: str = "workspace_layout.json") -> None:
        self.workspaces[workspace_name] = layout_config
        try:
            from pathlib import Path
            Path(config_path).write_text(json.dumps(self.workspaces, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_workspace_layout(self, workspace_name: str, config_path: str = "workspace_layout.json") -> Dict[str, Any]:
        from pathlib import Path
        p = Path(config_path)
        if p.exists():
            try:
                loaded = json.loads(p.read_text(encoding="utf-8"))
                self.workspaces.update(loaded)
            except Exception:
                pass
        return self.workspaces.get(workspace_name, self.workspaces["Default"])
