import copy
import os
import unittest

from fastapi.testclient import TestClient
from jsonschema import validate

import server


client = TestClient(server.app)


def rpc(method, params=None, request_id=1):
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    return client.post("/mcp", json=payload)


def call_tool(arguments, request_id=10):
    return rpc(
        "tools/call",
        {
            "name": server.TOOL_NAME,
            "arguments": arguments,
        },
        request_id=request_id,
    )


class FormSubmissionValidatorTests(unittest.TestCase):
    def assert_output_schema(self, structured):
        self.assertEqual(set(structured.keys()), server.OUTPUT_KEYS)
        validate(instance=structured, schema=server.TOOL_OUTPUT_SCHEMA)

    def test_server_starts_and_root_works(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("<title>Form Submission Validator</title>", response.text)
        self.assertIn("Read-Only MCP App", response.text)
        self.assertIn("POST /mcp", response.text)

    def test_review_pages_return_form_submission_validator_content(self):
        cases = [
            ("/privacy", "Privacy | Form Submission Validator", "does not store submitted form data"),
            ("/terms", "Terms | Form Submission Validator", "structured validation of form submission readiness"),
            ("/support", "Support | Form Submission Validator", "validation results, schema output"),
        ]
        for route, title, required_text in cases:
            response = client.get(route)
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers["content-type"])
            self.assertIn(title, response.text)
            self.assertIn(required_text, response.text)

    def test_health_returns_success(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_openai_apps_challenge_reads_environment_token(self):
        old_value = os.environ.get("OPENAI_APPS_CHALLENGE")
        os.environ["OPENAI_APPS_CHALLENGE"] = "review-shell-token"
        try:
            response = client.get("/.well-known/openai-apps-challenge")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "review-shell-token")
        finally:
            if old_value is None:
                os.environ.pop("OPENAI_APPS_CHALLENGE", None)
            else:
                os.environ["OPENAI_APPS_CHALLENGE"] = old_value

    def test_initialize_works(self):
        response = rpc("initialize", {}, 1)
        self.assertEqual(response.status_code, 200)
        result = response.json()["result"]
        self.assertIn("protocolVersion", result)
        self.assertIn("serverInfo", result)
        self.assertIn("capabilities", result)

    def test_tools_list_returns_exactly_one_complete_tool(self):
        response = rpc("tools/list", {}, 2)
        self.assertEqual(response.status_code, 200)
        tools = response.json()["result"]["tools"]
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool["name"], "validate_form_submission")
        self.assertEqual(tool["title"], "Form Submission Validator")
        self.assertIn("description", tool)
        self.assertIn("Do not use this tool when", tool["description"])
        self.assertIn("does not provide concrete form fields", tool["description"])
        self.assertIn("does not provide required field names", tool["description"])
        self.assertIn("inputSchema", tool)
        self.assertIn("outputSchema", tool)
        required_fields_schema = tool["inputSchema"]["properties"]["required_fields"]
        self.assertEqual(required_fields_schema["minItems"], 1)
        self.assertEqual(required_fields_schema["items"]["minLength"], 1)
        self.assertEqual(tool["outputSchema"], server.TOOL_OUTPUT_SCHEMA)
        self.assertEqual(
            tool["annotations"],
            {
                "readOnlyHint": True,
                "destructiveHint": False,
                "openWorldHint": False,
            },
        )

    def test_valid_full_form_can_submit_true(self):
        response = call_tool(
            {
                "fields": {
                    "name": "Alex",
                    "email": "alex@example.com",
                    "message": "Hello",
                },
                "required_fields": ["name", "email", "message"],
            }
        )
        self.assertEqual(response.status_code, 200)
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(
            structured,
            {
                "can_submit": True,
                "missing_fields": [],
                "invalid_fields": [],
                "errors": [],
            },
        )
        self.assert_output_schema(structured)

    def test_missing_required_field_can_submit_false(self):
        response = call_tool(
            {
                "fields": {"name": "Alex", "email": "alex@example.com"},
                "required_fields": ["name", "email", "message"],
            }
        )
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(structured["can_submit"], False)
        self.assertEqual(structured["missing_fields"], ["message"])
        self.assert_output_schema(structured)

    def test_empty_string_required_field_can_submit_false(self):
        response = call_tool(
            {
                "fields": {"name": "Alex", "email": "alex@example.com", "message": "   "},
                "required_fields": ["name", "email", "message"],
            }
        )
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(structured["can_submit"], False)
        self.assertEqual(structured["missing_fields"], ["message"])
        self.assert_output_schema(structured)

    def test_null_required_field_can_submit_false(self):
        response = call_tool(
            {
                "fields": {"name": "Alex", "email": "alex@example.com", "message": None},
                "required_fields": ["name", "email", "message"],
            }
        )
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(structured["can_submit"], False)
        self.assertEqual(structured["missing_fields"], ["message"])
        self.assert_output_schema(structured)

    def test_invalid_email_format_returns_invalid_fields(self):
        response = call_tool(
            {
                "fields": {"name": "Alex", "email": "not-an-email", "message": "Hello"},
                "required_fields": ["name", "email", "message"],
            }
        )
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(structured["can_submit"], False)
        self.assertEqual(
            structured["invalid_fields"],
            [{"field": "email", "reason": "invalid_email_format"}],
        )
        self.assert_output_schema(structured)

    def test_missing_fields_input_returns_fixed_error(self):
        response = call_tool({"required_fields": ["name"]})
        self.assertEqual(response.status_code, 200)
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(
            structured,
            {
                "can_submit": False,
                "missing_fields": [],
                "invalid_fields": [],
                "errors": [{"code": "missing_field", "message": "fields is required"}],
            },
        )
        self.assert_output_schema(structured)

    def test_empty_arguments_returns_combined_missing_field_error(self):
        response = call_tool({})
        self.assertEqual(response.status_code, 200)
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(
            structured,
            {
                "can_submit": False,
                "missing_fields": [],
                "invalid_fields": [],
                "errors": [
                    {
                        "code": "missing_field",
                        "message": "fields and required_fields are required.",
                    }
                ],
            },
        )
        self.assert_output_schema(structured)

    def test_null_arguments_returns_structured_error_not_http_500(self):
        response = call_tool(None)
        self.assertEqual(response.status_code, 200)
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(
            structured["errors"],
            [
                {
                    "code": "missing_field",
                    "message": "fields and required_fields are required.",
                }
            ],
        )
        self.assert_output_schema(structured)

    def test_missing_required_fields_input_returns_fixed_error(self):
        response = call_tool({"fields": {"name": "Alex"}})
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(
            structured["errors"],
            [{"code": "missing_field", "message": "required_fields is required"}],
        )
        self.assert_output_schema(structured)

    def test_invalid_input_types_return_fixed_errors(self):
        cases = [
            ({"fields": [], "required_fields": ["name"]}, "fields must be an object"),
            (
                {"fields": {}, "required_fields": ["name", 5]},
                "required_fields must be an array of strings",
            ),
        ]
        for arguments, message in cases:
            response = call_tool(arguments)
            self.assertEqual(response.status_code, 200)
            structured = response.json()["result"]["structuredContent"]
            self.assertEqual(
                structured["errors"],
                [{"code": "invalid_value", "message": message}],
            )
            self.assert_output_schema(structured)

    def test_empty_fields_and_empty_required_fields_returns_out_of_scope(self):
        response = call_tool({"fields": {}, "required_fields": []})
        self.assertEqual(response.status_code, 200)
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(structured["can_submit"], False)
        self.assertEqual(
            structured["errors"],
            [
                {
                    "code": "out_of_scope",
                    "message": (
                        "This tool requires explicit form fields and required field names."
                    ),
                }
            ],
        )
        self.assert_output_schema(structured)

    def test_empty_required_fields_with_fields_returns_invalid_value(self):
        response = call_tool({"fields": {"name": "Alex"}, "required_fields": []})
        self.assertEqual(response.status_code, 200)
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(structured["can_submit"], False)
        self.assertEqual(
            structured["errors"],
            [
                {
                    "code": "invalid_value",
                    "message": "required_fields must contain at least one field name.",
                }
            ],
        )
        self.assert_output_schema(structured)

    def test_blank_required_field_name_returns_invalid_value(self):
        response = call_tool({"fields": {"name": "Alex"}, "required_fields": [" "]})
        self.assertEqual(response.status_code, 200)
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(structured["can_submit"], False)
        self.assertEqual(structured["errors"][0]["code"], "invalid_value")
        self.assert_output_schema(structured)

    def test_repeated_same_call_returns_stable_structured_content(self):
        arguments = {
            "fields": {"name": "Alex", "email": "alex@example.com", "message": "Hello"},
            "required_fields": ["name", "email", "message"],
        }
        outputs = [
            call_tool(copy.deepcopy(arguments), request_id=index)
            .json()["result"]["structuredContent"]
            for index in range(3)
        ]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])
        for structured in outputs:
            self.assert_output_schema(structured)

    def test_structured_content_contains_no_extra_fields(self):
        structured = call_tool(
            {"fields": {"name": "Alex"}, "required_fields": ["name"]}
        ).json()["result"]["structuredContent"]
        self.assertEqual(set(structured.keys()), server.OUTPUT_KEYS)

    def test_out_of_scope_input_returns_structured_error(self):
        response = call_tool(
            {
                "request": "Should I submit this form?",
                "fields": {"name": "Alex"},
                "required_fields": ["name"],
            }
        )
        self.assertEqual(response.status_code, 200)
        structured = response.json()["result"]["structuredContent"]
        self.assertEqual(
            structured["errors"],
            [{"code": "out_of_scope", "message": "request is out of scope"}],
        )
        self.assert_output_schema(structured)

    def test_normal_validation_issues_do_not_return_http_500(self):
        responses = [
            call_tool({"required_fields": ["name"]}),
            call_tool({"fields": {"name": ""}, "required_fields": ["name"]}),
            call_tool(
                {
                    "fields": {"name": "Alex", "email": "bad"},
                    "required_fields": ["name", "email"],
                }
            ),
        ]
        self.assertTrue(all(response.status_code == 200 for response in responses))

    def test_get_mcp_is_not_endpoint(self):
        response = client.get("/mcp")
        self.assertEqual(response.status_code, 405)


if __name__ == "__main__":
    unittest.main()
