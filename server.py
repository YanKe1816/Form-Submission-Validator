import json
import os
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse


PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "form-submission-validator-mcp", "version": "0.1.0"}
TOOL_NAME = "validate_form_submission"
TOOL_TITLE = "Form Submission Validator"
TOOL_DESCRIPTION = (
    "Use this tool only when the request provides explicit form field values and "
    "an explicit list of required field names to validate. The tool validates a "
    "concrete form payload against concrete required fields and returns a fixed "
    "structured result. Do not use this tool when the user asks whether they "
    "should submit a form, asks for advice about submitting, asks to fill missing "
    "fields, asks to rewrite or improve form content, does not provide concrete "
    "form fields, does not provide required field names, or the request is "
    "open-ended, subjective, or asks for content generation. The tool must not "
    "infer missing fields, fill values, rewrite text, submit forms, or make "
    "subjective submit/no-submit decisions."
)

TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "object",
            "description": (
                "The form field values to validate. Keys are field names and "
                "values are the submitted values."
            ),
            "additionalProperties": True,
        },
        "required_fields": {
            "type": "array",
            "description": (
                "The field names that must be present and non-empty before submission."
            ),
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        },
    },
    "required": ["fields", "required_fields"],
    "additionalProperties": False,
}

TOOL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "can_submit": {
            "type": "boolean",
            "description": (
                "True only when all required fields are present, non-empty, and "
                "no invalid fields are found."
            ),
        },
        "missing_fields": {
            "type": "array",
            "description": (
                "Required field names that are missing, null, or empty strings."
            ),
            "items": {"type": "string"},
        },
        "invalid_fields": {
            "type": "array",
            "description": "Fields that are present but fail simple validation rules.",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["field", "reason"],
                "additionalProperties": False,
            },
        },
        "errors": {
            "type": "array",
            "description": (
                "Fixed contract errors for invalid input or out-of-scope requests."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["code", "message"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["can_submit", "missing_fields", "invalid_fields", "errors"],
    "additionalProperties": False,
}

TOOL_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "openWorldHint": False,
}

OUTPUT_KEYS = {"can_submit", "missing_fields", "invalid_fields", "errors"}
ERROR_CODES = {"missing_field", "invalid_value", "out_of_scope", "internal_error"}
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_PATTERN = re.compile(r"^[+()\-\s.\d]+$")
OUT_OF_SCOPE_PATTERNS = [
    "should i submit",
    "can i submit",
    "rewrite",
    "rewrite this form",
    "fill in",
    "fill in the missing fields",
    "fill the missing",
    "submit this",
    "what should i write",
    "improve this form",
]


app = FastAPI(title=TOOL_TITLE)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_output(
    can_submit: bool,
    missing_fields: list[str] | None = None,
    invalid_fields: list[dict[str, str]] | None = None,
    errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "can_submit": can_submit,
        "missing_fields": missing_fields or [],
        "invalid_fields": invalid_fields or [],
        "errors": errors or [],
    }


def contract_error(code: str, message: str) -> dict[str, Any]:
    if code not in ERROR_CODES:
        code = "internal_error"
        message = "internal error"
    return make_output(False, errors=[{"code": code, "message": message}])


def is_empty_required_value(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip() == ""


def format_issue_for_required_field(field: str, value: Any) -> dict[str, str] | None:
    normalized_field = field.strip().lower()
    if normalized_field == "email":
        if not isinstance(value, str) or not EMAIL_PATTERN.match(value.strip()):
            return {"field": field, "reason": "invalid_email_format"}
    if normalized_field == "phone":
        if not isinstance(value, str):
            return {"field": field, "reason": "invalid_phone_format"}
        digits = re.sub(r"\D", "", value)
        if not PHONE_PATTERN.match(value.strip()) or len(digits) < 7 or len(digits) > 15:
            return {"field": field, "reason": "invalid_phone_format"}
    return None


def is_out_of_scope(arguments: Any) -> bool:
    if not isinstance(arguments, dict):
        return False
    text_parts = []
    for key in ("request", "instruction", "prompt", "question", "task"):
        value = arguments.get(key)
        if isinstance(value, str):
            text_parts.append(value.lower())
    joined = " ".join(text_parts)
    return any(pattern in joined for pattern in OUT_OF_SCOPE_PATTERNS)


def validate_form_submission(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == {}:
        return contract_error(
            "missing_field", "fields and required_fields are required."
        )
    if not isinstance(arguments, dict):
        return contract_error("invalid_value", "arguments must be an object")
    if is_out_of_scope(arguments):
        return contract_error("out_of_scope", "request is out of scope")
    if "fields" not in arguments:
        return contract_error("missing_field", "fields is required")
    if "required_fields" not in arguments:
        return contract_error("missing_field", "required_fields is required")

    fields = arguments["fields"]
    required_fields = arguments["required_fields"]

    if not isinstance(fields, dict) or isinstance(fields, list):
        return contract_error("invalid_value", "fields must be an object")
    if not isinstance(required_fields, list) or not all(
        isinstance(field, str) for field in required_fields
    ):
        return contract_error(
            "invalid_value", "required_fields must be an array of strings"
        )
    if fields == {} and required_fields == []:
        return contract_error(
            "out_of_scope",
            "This tool requires explicit form fields and required field names.",
        )
    if required_fields == []:
        return contract_error(
            "invalid_value", "required_fields must contain at least one field name."
        )
    if any(field.strip() == "" for field in required_fields):
        return contract_error(
            "invalid_value", "required_fields must contain non-empty field names"
        )

    missing_fields = [
        field
        for field in required_fields
        if field not in fields or is_empty_required_value(fields[field])
    ]
    missing_field_set = set(missing_fields)
    invalid_fields = [
        issue
        for field in required_fields
        if field not in missing_field_set
        for issue in [format_issue_for_required_field(field, fields[field])]
        if issue is not None
    ]

    return make_output(
        can_submit=not missing_fields and not invalid_fields,
        missing_fields=missing_fields,
        invalid_fields=invalid_fields,
        errors=[],
    )


def tool_definition() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "title": TOOL_TITLE,
        "description": TOOL_DESCRIPTION,
        "inputSchema": TOOL_INPUT_SCHEMA,
        "outputSchema": TOOL_OUTPUT_SCHEMA,
        "annotations": TOOL_ANNOTATIONS,
    }


def initialize_result() -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": SERVER_INFO,
        "capabilities": {"tools": {}},
    }


def jsonrpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def jsonrpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def tool_call_result(structured: dict[str, Any]) -> dict[str, Any]:
    summary = json.dumps(structured, ensure_ascii=True, separators=(",", ":"))
    return {
        "content": [{"type": "text", "text": summary}],
        "structuredContent": structured,
        "isError": bool(structured["errors"]),
    }


def handle_mcp_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return jsonrpc_error(None, -32600, "Invalid Request")

    request_id = payload.get("id")
    method = payload.get("method")
    params = payload.get("params") or {}

    if method == "initialize":
        return jsonrpc_result(request_id, initialize_result())
    if method == "tools/list":
        return jsonrpc_result(request_id, {"tools": [tool_definition()]})
    if method == "tools/call":
        if not isinstance(params, dict):
            return jsonrpc_error(request_id, -32602, "Invalid params")
        if params.get("name") != TOOL_NAME:
            return jsonrpc_error(request_id, -32602, f"Unknown tool: {params.get('name')}")
        structured = validate_form_submission(params.get("arguments"))
        return jsonrpc_result(request_id, tool_call_result(structured))

    return jsonrpc_error(request_id, -32601, "Method not found")


def html_response(filename: str) -> FileResponse:
    return FileResponse(os.path.join(BASE_DIR, filename), media_type="text/html")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(
        os.path.join(BASE_DIR, "index.html"),
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/privacy")
def privacy() -> FileResponse:
    return FileResponse(
        os.path.join(BASE_DIR, "privacy.html"),
        media_type="text/html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/privacy-policy")
def privacy_policy() -> FileResponse:
    return FileResponse(
        os.path.join(BASE_DIR, "privacy.html"),
        media_type="text/html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/terms")
def terms() -> FileResponse:
    return html_response("terms.html")


@app.get("/support")
def support() -> FileResponse:
    return html_response("support.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/.well-known/openai-apps-challenge")
def openai_apps_challenge() -> PlainTextResponse:
    return PlainTextResponse(os.environ.get("OPENAI_APPS_CHALLENGE", "test"))


@app.post("/mcp")
async def mcp(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(jsonrpc_error(None, -32700, "Parse error"), status_code=400)
    return JSONResponse(handle_mcp_request(payload), status_code=200)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
