import asyncio
import json
import sys

from open_webui.models.models import ModelForm, Models
from open_webui.models.users import Users


async def main(path):
    spec = json.load(open(path))
    users = await Users.get_users()
    users = users.get("users", users) if isinstance(users, dict) else users
    admin = [user for user in users if user.role == "admin"][0]
    for entry in spec["models"]:
        meta = dict(spec["defaults"], description=entry["description"])
        params = {"compact_token_threshold": entry["compact_token_threshold"]}
        grants = [{"principal_type": "user", "principal_id": "*", "permission": "read"}] if entry.get("public", True) else []
        form = ModelForm(id=entry["id"], base_model_id=None, name=entry["name"], meta=meta, params=params, is_active=True, access_grants=grants)
        if await Models.get_model_by_id(entry["id"]) is None:
            await Models.insert_new_model(form, admin.id)
            action = "created"
        else:
            await Models.update_model_by_id(entry["id"], form)
            action = "updated"
        print(action, entry["id"], "public" if grants else "admin only")


asyncio.run(main(sys.argv[1]))
