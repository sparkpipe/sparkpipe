import asyncio
import json
import sys
from pathlib import Path

from open_webui.models.functions import FunctionForm, FunctionMeta, Functions
from open_webui.models.prompts import PromptForm, Prompts
from open_webui.models.tools import ToolForm, ToolMeta, Tools
from open_webui.models.users import Users
from open_webui.utils.plugin import load_function_module_by_id, load_tool_module_by_id, replace_imports
from open_webui.utils.tools import get_tool_specs

PUBLIC = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]


async def main(path):
    spec = json.load(open(path))
    users = await Users.get_users()
    users = users.get("users", users) if isinstance(users, dict) else users
    admin = [user for user in users if user.role == "admin"][0]
    for entry in spec["functions"]:
        content = replace_imports((Path(__file__).parent / entry["file"]).read_text())
        module, kind, frontmatter, _ = await load_function_module_by_id(entry["id"], content=content)
        meta = FunctionMeta(description=frontmatter.get("description", ""), manifest=frontmatter, toggle=kind == "filter" and bool(getattr(module, "toggle", False)))
        form = FunctionForm(id=entry["id"], name=frontmatter["title"], content=content, meta=meta)
        if await Functions.get_function_by_id(entry["id"]) is None:
            await Functions.insert_new_function(admin.id, kind, form, module=module)
            action = "created"
        else:
            await Functions.update_function_by_id(entry["id"], {**form.model_dump(exclude={"id"}), "type": kind}, user_id=admin.id, module=module)
            action = "updated"
        await Functions.update_function_by_id(entry["id"], {"is_active": True, "is_global": bool(entry["global"])})
        print(action, entry["id"], kind, "global" if entry["global"] else "per model")
    for entry in spec.get("tools", []):
        content = replace_imports((Path(__file__).parent / entry["file"]).read_text())
        module, frontmatter, _ = await load_tool_module_by_id(entry["id"], content=content)
        form = ToolForm(id=entry["id"], name=frontmatter["title"], content=content,
                        meta=ToolMeta(description=frontmatter.get("description", ""), manifest=frontmatter, has_user_valves=hasattr(module, "UserValves")),
                        access_grants=PUBLIC)
        if await Tools.get_tool_by_id(entry["id"]) is None:
            await Tools.insert_new_tool(admin.id, form, get_tool_specs(module), module=module)
            action = "created"
        else:
            await Tools.update_tool_by_id(entry["id"], {**form.model_dump(exclude={"id"}), "specs": get_tool_specs(module)}, user_id=admin.id, module=module)
            action = "updated"
        print(action, entry["id"], "tool")
    for entry in spec.get("prompts", []):
        form = PromptForm(command=entry["command"], name=entry["name"], content=entry["content"], access_grants=PUBLIC)
        existing = await Prompts.get_prompt_by_command(entry["command"])
        if existing is None:
            await Prompts.insert_new_prompt(admin.id, form)
            action = "created"
        else:
            await Prompts.update_prompt_by_command(entry["command"], form, admin.id)
            action = "updated"
        print(action, "/" + entry["command"], "prompt")


asyncio.run(main(sys.argv[1]))
