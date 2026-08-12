#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import os
import shutil


NODE_LAYOUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),"spark_node_layout.json")
MODEL_RESIDENT_ROLE = "model_resident"
MODEL_RESIDENT_COMMAND = "bin/sparkpipe_model_residentd"
REQUIRED_RELEASE_FILES = {
    MODEL_RESIDENT_COMMAND,
    "lib/model_serving_adapter.so",
    "lib/model_driver.so",
    "lib/hidden_transport.so",
    "config/model_resident.json",
}
MODEL_RESIDENT_CONFIGURATION = "config/model_resident.json"


def load_node_layout():
    with open(NODE_LAYOUT_PATH,"r",encoding="utf-8") as source:
        layout = json.load(source)
    if (set(layout) != {"schema_version","node_root_template","roots"}
            or layout["schema_version"] != 1
            or set(layout["roots"]) != {
                "sparkdata","srcdata","extnvme","kvcache"}):
        raise SystemExit("node layout schema is invalid")
    node_root = layout["node_root_template"]
    if (not isinstance(node_root,str) or node_root != "/home/{host}"
            or any(not isinstance(value,str) or "/" in value or value == ""
                   for value in layout["roots"].values())):
        raise SystemExit("node layout paths are invalid")
    return layout


def apply_node_layout(manifest,dataset):
    if (not isinstance(dataset,str) or dataset == ""
            or "/" in dataset or os.path.normpath(dataset) != dataset):
        raise SystemExit("install dataset must be one directory name")
    layout = load_node_layout()
    sparkdata = layout["node_root_template"] + "/" + layout["roots"]["sparkdata"]
    manifest["install_root"] = sparkdata + "/" + dataset
    manifest["state_root"] = (
        sparkdata + "/.layout/sparkpipe_state/" + dataset)


def apply_deployment_contract(root,manifest):
    path = os.path.join(root,MODEL_RESIDENT_CONFIGURATION)
    try:
        with open(path,"r",encoding="utf-8") as source:
            deployment = json.load(source)
    except (OSError,json.JSONDecodeError) as error:
        raise SystemExit("packaged model_resident deployment is invalid") from error
    limits = deployment.get("runtime_limits")
    nodes = deployment.get("nodes")
    max_active = limits.get("max_active_sequences") if isinstance(limits,dict) else None
    if (not isinstance(max_active,int) or isinstance(max_active,bool)
            or max_active < 1 or max_active > 0xffffffff):
        raise SystemExit("packaged deployment max_active_sequences is invalid")
    if not isinstance(nodes,list) or len(nodes) < 1 or len(nodes) > 0xffffffff:
        raise SystemExit("packaged deployment nodes are invalid")
    manifest["max_active_sequence_count"] = max_active
    manifest["rank_count"] = len(nodes)


def sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as source:
        while True:
            data = source.read(1024 * 1024)
            if not data:
                return digest.hexdigest()
            digest.update(data)


def normalized_relative_path(path):
    return (
        isinstance(path,str)
        and path != ""
        and not os.path.isabs(path)
        and os.path.normpath(path) == path
        and path != ".."
        and not path.startswith("../")
    )


def role_by_name(manifest,role_name):
    matches = [role for role in manifest["roles"]
               if role.get("name") == role_name]
    if len(matches) != 1:
        raise SystemExit(
            "release must contain exactly one role named " + role_name)
    return matches[0]


def validate_template(manifest):
    if manifest.get("schema_version") != 1:
        raise SystemExit("release template schema_version must be 1")
    files = manifest.get("files")
    roles = manifest.get("roles")
    if not isinstance(files,list) or not isinstance(roles,list):
        raise SystemExit("release template must contain files and roles arrays")
    file_paths = [entry.get("path") for entry in files
                  if isinstance(entry,dict)]
    if len(file_paths) != len(files) or len(set(file_paths)) != len(file_paths):
        raise SystemExit("release file paths must be present and unique")
    for path in file_paths:
        if not normalized_relative_path(path):
            raise SystemExit("release file path is not normalized: " + str(path))
    missing = sorted(REQUIRED_RELEASE_FILES - set(file_paths))
    if missing:
        raise SystemExit("release is missing required file: " + missing[0])
    role_names = [role.get("name") for role in roles
                  if isinstance(role,dict)]
    if len(role_names) != len(roles) or len(set(role_names)) != len(role_names):
        raise SystemExit("release role names must be present and unique")
    if set(role_names) != {MODEL_RESIDENT_ROLE}:
        raise SystemExit("release must contain only the model_resident role")
    role = role_by_name(manifest,MODEL_RESIDENT_ROLE)
    if role.get("command") != MODEL_RESIDENT_COMMAND:
        raise SystemExit("model_resident must run " + MODEL_RESIDENT_COMMAND)
    arguments = role.get("argv")
    if not isinstance(arguments,list):
        raise SystemExit("model_resident argv must be an array")
    matches = [index for index,value in enumerate(arguments)
               if value == "--deployment"]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise SystemExit("model_resident must name one --deployment file")
    if arguments[matches[0] + 1] != "{install_root}/config/model_resident.json":
        raise SystemExit("model_resident must load the packaged deployment")


def set_role_release_identity(manifest):
    values = {
        "SPARKPIPE_RELEASE_ID": manifest["release_id"],
        "SPARKPIPE_RELEASE_GIT_COMMIT": manifest["git_commit"],
        "SPARKPIPE_RELEASE_GENERATION": str(manifest["generation"]),
    }
    for role in manifest["roles"]:
        environment = role.setdefault("env",[])
        environment = [entry for entry in environment
                       if entry.split("=",1)[0] not in values]
        environment.extend(key + "=" + value for key,value in values.items())
        role["env"] = environment


def set_role_environment(manifest,specification):
    try:
        role_name,assignment = specification.split("=",1)
        name,value = assignment.split("=",1)
    except ValueError as error:
        raise SystemExit("role environment must be ROLE=NAME=VALUE") from error
    if not role_name or not name:
        raise SystemExit("role environment must be ROLE=NAME=VALUE")
    role = role_by_name(manifest,role_name)
    environment = role.setdefault("env",[])
    role["env"] = [entry for entry in environment
                   if entry.split("=",1)[0] != name]
    role["env"].append(name + "=" + value)


def unset_role_environment(manifest,specification):
    try:
        role_name,name = specification.split("=",1)
    except ValueError as error:
        raise SystemExit(
            "role environment removal must be ROLE=NAME") from error
    if not role_name or not name or "=" in name:
        raise SystemExit("role environment removal must be ROLE=NAME")
    role = role_by_name(manifest,role_name)
    environment = role.setdefault("env",[])
    role["env"] = [entry for entry in environment
                   if entry.split("=",1)[0] != name]


def apply_replacements(root,manifest,replacements):
    allowed = {entry["path"] for entry in manifest["files"]}
    seen = set()
    for specification in replacements:
        try:
            relative,source_path = specification.split("=",1)
        except ValueError as error:
            raise SystemExit(
                "replacement must be RELATIVE_PATH=SOURCE") from error
        if relative not in allowed:
            raise SystemExit("replacement is not in manifest: " + relative)
        if relative in seen:
            raise SystemExit("replacement occurs more than once: " + relative)
        if not os.path.isfile(source_path):
            raise SystemExit("missing replacement: " + source_path)
        destination = os.path.join(root,relative)
        os.makedirs(os.path.dirname(destination),exist_ok=True)
        shutil.copy2(source_path,destination)
        seen.add(relative)


def write_manifest(root,manifest):
    for entry in manifest["files"]:
        path = os.path.join(root,entry["path"])
        if not os.path.isfile(path):
            raise SystemExit("missing release file: " + entry["path"])
        entry["bytes"] = os.path.getsize(path)
        entry["sha256"] = sha256(path)
    path = os.path.join(root,"sparkpipe.json")
    with open(path,"w",encoding="utf-8") as target:
        json.dump(manifest,target,indent=2,sort_keys=True)
        target.write("\n")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--release-id",required=True)
    parser.add_argument("--git-commit",required=True)
    parser.add_argument("--install-dataset",required=True)
    parser.add_argument("--role-env",action="append",default=[])
    parser.add_argument("--role-env-unset",action="append",default=[])
    parser.add_argument("--replace",action="append",default=[])
    arguments = parser.parse_args()
    if not arguments.release_id:
        parser.error("--release-id must not be empty")
    if (len(arguments.git_commit) != 40
            or any(value not in "0123456789abcdef"
                   for value in arguments.git_commit)):
        parser.error("--git-commit must be an exact lowercase 40-hex commit")
    return arguments


def main():
    arguments = parse_arguments()
    temporary = arguments.output + ".assembling." + str(os.getpid())
    if os.path.exists(arguments.output) or os.path.exists(temporary):
        raise SystemExit("release output already exists")
    try:
        shutil.copytree(arguments.template,temporary)
        manifest_path = os.path.join(temporary,"sparkpipe.json")
        with open(manifest_path,"r",encoding="utf-8") as source:
            manifest = json.load(source)
        validate_template(manifest)
        apply_node_layout(manifest,arguments.install_dataset)
        apply_replacements(temporary,manifest,arguments.replace)
        apply_deployment_contract(temporary,manifest)
        manifest["release_id"] = arguments.release_id
        manifest["git_commit"] = arguments.git_commit
        manifest["generation"] = int(
            datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%d%H%M%S"))
        set_role_release_identity(manifest)
        for specification in arguments.role_env_unset:
            unset_role_environment(manifest,specification)
        for specification in arguments.role_env:
            set_role_environment(manifest,specification)
        write_manifest(temporary,manifest)
        os.rename(temporary,arguments.output)
    except BaseException:
        if os.path.isdir(temporary):
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
