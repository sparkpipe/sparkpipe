import asyncio
import json
import sys
from pathlib import Path

from open_webui.models.functions import FunctionForm, FunctionMeta, Functions
from open_webui.models.users import Users
from open_webui.utils.plugin import load_function_module_by_id, replace_imports


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


asyncio.run(main(sys.argv[1]))
