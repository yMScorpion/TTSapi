#!/usr/bin/env python3
"""Verify documented API/config artifacts without loading model code or weights."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tree = ast.parse((root / "api_server.py").read_text())
routes = []
formats = []
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for item in node.decorator_list:
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and isinstance(item.func.value, ast.Name) and item.func.value.id == "app" and item.args and isinstance(item.args[0], ast.Constant):
                routes.append([item.func.attr.upper(), item.args[0].value])
    if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "AudioFormat" for x in node.targets):
        formats = [x.value for x in node.value.slice.elts]
dashboard = json.loads((root / "grafana/dashboards/styletts2_api.json").read_text())
assert len(routes) == len(set(map(tuple, routes))), "Duplicate HTTP route"
assert sorted(formats) == ["flac", "mp3", "ogg", "wav"], "Documented formats changed"
assert ["POST", "/v1/tts"] in routes and ["GET", "/healthz"] in routes
panels = [x for x in dashboard["panels"] if x["type"] != "row"]
assert all(x.get("targets") for x in panels), "Panel without configured query"
print(json.dumps({"kind": "static-api-config-inspection", "explicit_routes": routes, "route_count": len(routes), "formats": formats, "visualization_panels": len(panels), "inference_executed": False}, indent=2))
