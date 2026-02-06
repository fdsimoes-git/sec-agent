"""HTTP request tool."""

import httpx

from .base import Tool, ToolResult


class HttpRequestTool(Tool):
    """Make HTTP requests to URLs."""

    name = "http_request"
    description = (
        "Send an HTTP request and return the response. Useful for testing "
        "web endpoints, APIs, checking if services are up, "
        "and fetching web content."
    )
    parameters = {
        "method": {
            "type": "string",
            "description": "HTTP method: GET, POST, PUT, DELETE, HEAD, OPTIONS",
            "default": "GET",
        },
        "url": {
            "type": "string",
            "description": "The URL to request",
        },
        "headers": {
            "type": "object",
            "description": "Optional HTTP headers as key-value pairs",
            "default": {},
        },
        "body": {
            "type": "string",
            "description": "Optional request body",
            "default": "",
        },
    }
    requires_approval = True

    def execute(self, **kwargs) -> ToolResult:
        method = kwargs.get("method", "GET").upper()
        url = kwargs.get("url", "")
        headers = kwargs.get("headers", {})
        body = kwargs.get("body", "")

        if not url:
            return ToolResult(output="Error: no url provided", success=False)

        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=headers or {},
                    content=body if body else None,
                )

            lines = [
                f"HTTP {response.status_code} {response.reason_phrase}",
                "",
                "--- Response Headers ---",
            ]
            for key, value in response.headers.items():
                lines.append(f"{key}: {value}")

            lines.append("")
            lines.append("--- Response Body ---")

            body_text = response.text
            if len(body_text) > 5000:
                body_text = body_text[:5000] + "\n... (truncated)"
            lines.append(body_text)

            return ToolResult(output="\n".join(lines))
        except httpx.HTTPError as exc:
            return ToolResult(output=f"Error: {exc}", success=False)
