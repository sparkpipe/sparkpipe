"""
title: GitHub
author: SparkPipe
version: 1.0.0
description: Read public GitHub repositories, files, issues, pull requests, commits and releases.
"""

import base64
import re
from urllib.parse import quote

import aiohttp
from pydantic import BaseModel, Field

REPO = re.compile(r"^[A-Za-z0-9-]{1,39}/[A-Za-z0-9._-]{1,100}$")
SEARCH_KINDS = ("repositories", "code", "issues")
LIST_KINDS = ("issues", "pulls", "commits", "releases")
STATES = ("open", "closed", "all")


class Tools:
    class Valves(BaseModel):
        GITHUB_TOKEN: str = Field(
            default="",
            description="Fine-grained token with read-only access to public repositories. Empty uses GitHub's unauthenticated limit of 60 requests an hour, and code search needs a token.",
        )
        API_URL: str = Field(default="https://api.github.com", description="GitHub REST API base URL")
        MAX_CHARS: int = Field(default=40000, description="Longest reply returned to the model, in characters")
        TIMEOUT_SECONDS: int = Field(default=20, description="Timeout of one GitHub request")

    def __init__(self):
        self.valves = self.Valves()

    async def github_search(self, query: str, kind: str = "repositories") -> str:
        """
        Search GitHub for repositories, code, or issues and pull requests.
        :param query: GitHub search syntax, for example "repo:owner/name parser language:python" or "topic:llm stars:>1000"
        :param kind: repositories, code, or issues (issues also finds pull requests)
        """
        if kind not in SEARCH_KINDS:
            return f"kind must be one of {', '.join(SEARCH_KINDS)}"
        if kind == "code" and not self.valves.GITHUB_TOKEN:
            return "GitHub code search needs a token and none is configured; search repositories or issues, or read files with github_read"
        status, data = await self._get(f"/search/{kind}", {"q": query, "per_page": "10"})
        if status != 200:
            return self._error(status, data)
        lines = [f"{data.get('total_count', 0)} results"]
        for item in data.get("items", []):
            if kind == "repositories":
                lines.append(
                    f"{item['full_name']} ({item.get('stargazers_count', 0)} stars, {item.get('language') or 'no language'}): "
                    f"{item.get('description') or ''} {item['html_url']}"
                )
            elif kind == "code":
                lines.append(f"{item['repository']['full_name']}:{item['path']} {item['html_url']}")
            else:
                label = "pull request" if "pull_request" in item else "issue"
                repo = item.get("repository_url", "").rsplit("/repos/", 1)[-1]
                lines.append(f"{repo}#{item['number']} [{label}, {item['state']}] {item['title']} {item['html_url']}")
        return self._clip("\n".join(lines))

    async def github_read(self, repo: str, path: str = "", ref: str = "") -> str:
        """
        Read a file or list a directory in a GitHub repository. An empty path lists the top level and adds the README.
        :param repo: owner/name, for example "sparkpipe/sparkpipe"
        :param path: file or directory inside the repository
        :param ref: branch, tag or commit; empty means the default branch
        """
        problem = self._check_repo(repo) or self._check_path(path)
        if problem:
            return problem
        params = {"ref": ref} if ref else None
        clean = path.strip("/")
        status, data = await self._get(f"/repos/{repo}/contents/{quote(clean, safe='/')}", params)
        if status != 200:
            return self._error(status, data)
        if isinstance(data, list):
            lines = []
            for entry in data:
                size = f" ({entry.get('size', 0)} bytes)" if entry.get("type") == "file" else ""
                lines.append(f"{entry.get('type', ''):4} {entry.get('path', '')}{size}")
            text = "\n".join(lines)
            if not clean:
                status, readme = await self._get(f"/repos/{repo}/readme", params)
                if status == 200:
                    text += "\n\n" + self._decode(readme)
            return self._clip(text)
        if data.get("type") != "file":
            return f"{clean} is a {data.get('type')}, not a file or directory"
        return self._clip(self._decode(data))

    async def github_list(self, repo: str, kind: str = "issues", state: str = "open") -> str:
        """
        List the latest issues, pull requests, commits or releases of a GitHub repository.
        :param repo: owner/name, for example "sparkpipe/sparkpipe"
        :param kind: issues, pulls, commits or releases
        :param state: open, closed or all; used for issues and pulls
        """
        problem = self._check_repo(repo)
        if problem:
            return problem
        if kind not in LIST_KINDS:
            return f"kind must be one of {', '.join(LIST_KINDS)}"
        if state not in STATES:
            return f"state must be one of {', '.join(STATES)}"
        params = {"per_page": "20"}
        if kind in ("issues", "pulls"):
            params["state"] = state
        status, data = await self._get(f"/repos/{repo}/{kind}", params)
        if status != 200:
            return self._error(status, data)
        lines = []
        for item in data:
            if kind == "issues":
                if "pull_request" in item:
                    continue
                lines.append(f"#{item['number']} [{item['state']}] {item['title']} ({item['user']['login']}, updated {item['updated_at'][:10]})")
            elif kind == "pulls":
                lines.append(
                    f"#{item['number']} [{item['state']}] {item['title']} ({item['user']['login']}, "
                    f"{item['head']['ref']} -> {item['base']['ref']}, updated {item['updated_at'][:10]})"
                )
            elif kind == "commits":
                commit = item["commit"]
                summary = (commit.get("message") or "").splitlines()[0] if commit.get("message") else ""
                lines.append(f"{item['sha'][:9]} {commit['author']['date'][:10]} {commit['author']['name']}: {summary}")
            else:
                lines.append(f"{item['tag_name']} {item.get('name') or ''} ({(item.get('published_at') or '')[:10]})")
        return self._clip("\n".join(lines) or f"no {kind}")

    async def github_item(self, repo: str, number: int) -> str:
        """
        Read one issue or pull request with its comments; for a pull request also its branches and changed files.
        :param repo: owner/name, for example "sparkpipe/sparkpipe"
        :param number: the issue or pull request number
        """
        problem = self._check_repo(repo)
        if problem:
            return problem
        try:
            number = int(number)
        except (TypeError, ValueError):
            return "number must be a positive integer"
        if number < 1:
            return "number must be a positive integer"
        status, item = await self._get(f"/repos/{repo}/issues/{number}")
        if status != 200:
            return self._error(status, item)
        is_pull = "pull_request" in item
        parts = [
            f"{'Pull request' if is_pull else 'Issue'} {repo}#{number} [{item['state']}] {item['title']}",
            f"by {item['user']['login']}, opened {item['created_at'][:10]}, {item.get('comments', 0)} comments",
            item.get("body") or "",
        ]
        if is_pull:
            status, pull = await self._get(f"/repos/{repo}/pulls/{number}")
            if status == 200:
                parts.append(f"{pull['head']['label']} -> {pull['base']['label']}, merged: {pull.get('merged', False)}")
            status, files = await self._get(f"/repos/{repo}/pulls/{number}/files", {"per_page": "100"})
            if status == 200:
                parts.append(
                    "Changed files:\n"
                    + "\n".join(f"{f['status']} {f['filename']} (+{f['additions']} -{f['deletions']})" for f in files)
                )
        status, comments = await self._get(f"/repos/{repo}/issues/{number}/comments", {"per_page": "50"})
        if status == 200:
            for comment in comments:
                parts.append(f"--- {comment['user']['login']}, {comment['created_at'][:10]}:\n{comment.get('body') or ''}")
        return self._clip("\n\n".join(parts))

    async def _get(self, path, params=None):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "SparkPipe-Chat",
        }
        if self.valves.GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {self.valves.GITHUB_TOKEN}"
        timeout = aiohttp.ClientTimeout(total=self.valves.TIMEOUT_SECONDS)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self.valves.API_URL.rstrip("/") + path, params=params, headers=headers, allow_redirects=False
                ) as response:
                    try:
                        data = await response.json(content_type=None)
                    except ValueError:
                        data = {"message": (await response.text())[:200]}
                    if response.status in (301, 302, 307) and isinstance(data, dict):
                        data["message"] = "moved; the repository was probably renamed, so use its new name"
                    return response.status, data
        except (aiohttp.ClientError, TimeoutError) as error:
            return 0, {"message": str(error) or type(error).__name__}

    def _error(self, status, data):
        message = data.get("message", "") if isinstance(data, dict) else ""
        if status in (403, 429) and "rate limit" in message.lower():
            message += "; the GitHub rate limit is reached, try again later"
            if not self.valves.GITHUB_TOKEN:
                message += " or ask the admin to configure a token"
        return f"GitHub API error {status}: {message}"

    def _check_repo(self, repo):
        if not isinstance(repo, str) or not REPO.match(repo) or repo.split("/")[1] in (".", ".."):
            return "repo must be owner/name, for example sparkpipe/sparkpipe"
        return None

    def _check_path(self, path):
        if not isinstance(path, str) or any(ch in path for ch in ("\\", "?", "#", "\n", "\r")):
            return "path must be a plain path inside the repository"
        if any(part in (".", "..") for part in path.split("/")):
            return "path must be a plain path inside the repository"
        return None

    def _decode(self, data):
        name = data.get("path", "")
        if data.get("encoding") != "base64":
            return f"{name} is {data.get('size', 0)} bytes, too large to read here; fetch_url can read {data.get('download_url')}"
        raw = base64.b64decode(data.get("content") or "")
        try:
            return f"{name}:\n{raw.decode('utf-8')}"
        except UnicodeDecodeError:
            return f"{name} is a binary file of {len(raw)} bytes"

    def _clip(self, text):
        if len(text) > self.valves.MAX_CHARS:
            return text[: self.valves.MAX_CHARS] + "\n[truncated]"
        return text
