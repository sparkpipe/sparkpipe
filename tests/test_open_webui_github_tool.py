#!/usr/bin/env python3
"""SparkPipe Chat's GitHub tool only sends GET requests to the configured GitHub API, refuses malformed repositories and paths before any request, sends the token only when one is configured, decodes files, lists directories with the README, clips long replies, reads a pull request with its files and comments, and reports API errors instead of raising."""
import asyncio
import base64
import importlib.util
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("github_tool", ROOT / "deployment" / "hub" / "open_webui_tools" / "github.py")
github_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(github_tool)


def b64(data):
    return base64.b64encode(data).decode()


def fake_github(seen):
    def entry(request):
        seen.append((request.method, request.path, dict(request.query), request.headers.get("Authorization")))

    async def handler(request):
        entry(request)
        routes = {
            "/search/repositories": {"total_count": 1, "items": [{"full_name": "o/r", "stargazers_count": 5, "language": "C", "description": "demo", "html_url": "https://github.com/o/r"}]},
            "/search/code": {"total_count": 1, "items": [{"repository": {"full_name": "o/r"}, "path": "src/a.c", "html_url": "https://github.com/o/r/blob/main/src/a.c"}]},
            "/repos/o/r/contents/": [{"type": "file", "path": "README.md", "size": 14}, {"type": "dir", "path": "src"}],
            "/repos/o/r/readme": {"type": "file", "path": "README.md", "encoding": "base64", "content": b64(b"# Title\nhello")},
            "/repos/o/r/contents/src/big.txt": {"type": "file", "path": "src/big.txt", "encoding": "base64", "content": b64(b"x" * 500)},
            "/repos/o/r/contents/bin.dat": {"type": "file", "path": "bin.dat", "encoding": "base64", "content": b64(b"\xff\xfe\x00")},
            "/repos/o/r/issues": [
                {"number": 3, "state": "open", "title": "a bug", "user": {"login": "u"}, "updated_at": "2026-10-01T00:00:00Z"},
                {"number": 7, "state": "open", "title": "a change", "user": {"login": "u"}, "updated_at": "2026-10-02T00:00:00Z", "pull_request": {}},
            ],
            "/repos/o/r/issues/7": {"number": 7, "state": "open", "title": "a change", "user": {"login": "u"}, "created_at": "2026-10-02T00:00:00Z", "comments": 1, "body": "the description", "pull_request": {}},
            "/repos/o/r/pulls/7": {"head": {"label": "u:feature"}, "base": {"label": "o:main"}, "merged": False},
            "/repos/o/r/pulls/7/files": [{"status": "modified", "filename": "src/a.c", "additions": 3, "deletions": 1}],
            "/repos/o/r/issues/7/comments": [{"user": {"login": "reviewer"}, "created_at": "2026-10-03T00:00:00Z", "body": "looks right"}],
        }
        if request.path not in routes:
            return web.json_response({"message": "Not Found"}, status=404)
        return web.json_response(routes[request.path])

    async def refuse(request):
        entry(request)
        return web.json_response({"message": "method not allowed"}, status=405)

    app = web.Application()
    app.router.add_route("GET", "/{tail:.*}", handler)
    app.router.add_route("*", "/{tail:.*}", refuse)
    return app


async def run():
    seen = []
    runner = web.AppRunner(fake_github(seen))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    tool = github_tool.Tools()
    tool.valves.API_URL = f"http://127.0.0.1:{port}"
    try:
        root = await tool.github_read("o/r")
        assert "file README.md (14 bytes)" in root and "dir  src" in root and "# Title\nhello" in root, root

        tool.valves.MAX_CHARS = 100
        big = await tool.github_read("o/r", "src/big.txt")
        assert big.endswith("\n[truncated]") and len(big) == 100 + len("\n[truncated]"), big
        tool.valves.MAX_CHARS = 40000

        assert await tool.github_read("o/r", "bin.dat") == "bin.dat is a binary file of 3 bytes"

        before = len(seen)
        for repo, path in (("../x", ""), ("o/r/extra", ""), ("o/..", ""), ("o/r", "../../users"), ("o/r", "src/./a"), ("o/r", "a?b"), ("o/r", "a\\b")):
            reply = await tool.github_read(repo, path)
            assert reply.startswith(("repo must be", "path must be")), (repo, path, reply)
        assert await tool.github_item("o/r", "seven") == "number must be a positive integer"
        assert await tool.github_list("o/r", "secrets") == "kind must be one of issues, pulls, commits, releases"
        assert (await tool.github_search("x", "code")).startswith("GitHub code search needs a token")
        assert len(seen) == before, seen[before:]

        repos = await tool.github_search("demo")
        assert "o/r (5 stars, C): demo https://github.com/o/r" in repos, repos
        assert seen[-1][2] == {"q": "demo", "per_page": "10"} and seen[-1][3] is None, seen[-1]

        tool.valves.GITHUB_TOKEN = "test-token"
        code = await tool.github_search("main", "code")
        assert "o/r:src/a.c" in code and seen[-1][3] == "Bearer test-token", (code, seen[-1])
        tool.valves.GITHUB_TOKEN = ""

        issues = await tool.github_list("o/r")
        assert "#3 [open] a bug" in issues and "#7" not in issues, issues

        pull = await tool.github_item("o/r", 7)
        for part in ("Pull request o/r#7 [open] a change", "the description", "u:feature -> o:main, merged: False", "modified src/a.c (+3 -1)", "--- reviewer, 2026-10-03:\nlooks right"):
            assert part in pull, (part, pull)

        assert await tool.github_read("o/missing") == "GitHub API error 404: Not Found"
        assert all(method == "GET" for method, *_ in seen), seen
    finally:
        await runner.cleanup()


asyncio.run(run())
print("PASS test_open_webui_github_tool: GET-only GitHub reads, refused bad input before any request, token only when set, clipped replies, pull request with files and comments, API errors reported")
