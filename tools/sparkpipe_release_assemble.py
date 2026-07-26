#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import os
import shutil


DIAGNOSTIC_ENVIRONMENT_BY_ROLE = {
    "spark0_gateway": {
        "SPARKPIPE_PP13_TRACE": "1",
    },
    "pp13_cuda_residentd": {
        "SPARKPIPE_STAGE_COMPLETION_DEBUG": "1",
        "SPARKPIPE_STAGE_PHASE_HASH": "1",
        "SPARKPIPE_HIDDEN_DUMP_DIR": "{state_root}/hidden_dumps",
    },
    "pp13_rank_daemon": {
        "SPARKPIPE_STAGE_COMPLETION_DEBUG": "1",
        "SPARKPIPE_PP13_TRACE": "1",
    },
}
DIAGNOSTIC_ENVIRONMENT_NAMES = {
    key for values in DIAGNOSTIC_ENVIRONMENT_BY_ROLE.values()
    for key in values
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as source:
        while True:
            data = source.read(1024 * 1024)
            if not data:
                return digest.hexdigest()
            digest.update(data)


def set_role_max_active(manifest,max_active):
    manifest["max_active_sequence_count"] = max_active
    for role in manifest["roles"]:
        arguments = role.get("argv",[])
        for index in range(len(arguments) - 1):
            if arguments[index] == "--max-active":
                arguments[index + 1] = str(max_active)


def set_role_argument(manifest,role_name,argument,value):
    matching_roles = [role for role in manifest["roles"] if role.get("name") == role_name]
    if len(matching_roles) != 1:
        raise SystemExit("release must contain exactly one role named " + role_name)
    arguments = matching_roles[0].setdefault("argv",[])
    matches = [index for index,item in enumerate(arguments) if item == argument]
    if len(matches) > 1:
        raise SystemExit("role argument occurs more than once: " + argument)
    if matches:
        index = matches[0]
        if index + 1 >= len(arguments):
            raise SystemExit("role argument is missing its value: " + argument)
        arguments[index + 1] = str(value)
    else:
        arguments.extend([argument,str(value)])


def get_role_argument(manifest,role_name,argument):
    matching_roles = [role for role in manifest["roles"] if role.get("name") == role_name]
    if len(matching_roles) != 1:
        raise SystemExit("release must contain exactly one role named " + role_name)
    arguments = matching_roles[0].setdefault("argv",[])
    matches = [index for index,item in enumerate(arguments) if item == argument]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise SystemExit("role must contain exactly one valued argument: " + argument)
    return arguments[matches[0] + 1]


def remove_role_argument(manifest,role_name,argument):
    matching_roles = [role for role in manifest["roles"] if role.get("name") == role_name]
    if len(matching_roles) != 1:
        raise SystemExit("release must contain exactly one role named " + role_name)
    arguments = matching_roles[0].setdefault("argv",[])
    matches = [index for index,item in enumerate(arguments) if item == argument]
    if len(matches) > 1:
        raise SystemExit("role argument occurs more than once: " + argument)
    if matches:
        index = matches[0]
        if index + 1 >= len(arguments):
            raise SystemExit("role argument is missing its value: " + argument)
        del arguments[index:index + 2]


def set_model_arguments(
        manifest,model_quantization,moe_pack_root,stagepack_root):
    roles = ["spark0_gateway","pp13_cuda_residentd"]
    if model_quantization != "fp8" and stagepack_root is None:
        raise SystemExit("--stagepack-root is required for non-FP8 releases")
    if model_quantization != "fp8" and moe_pack_root is None:
        raise SystemExit("--moe-pack-root is required for non-FP8 releases")
    if stagepack_root is None:
        stagepack_root = get_role_argument(
            manifest,"pp13_cuda_residentd","--stagepack-root")
    if moe_pack_root is None:
        moe_pack_root = stagepack_root
    for role_name in roles:
        remove_role_argument(manifest,role_name,"--fp8-pack-root")
        set_role_argument(
            manifest,role_name,"--model-quantization",model_quantization)
        set_role_argument(manifest,role_name,"--stagepack-root",stagepack_root)
        set_role_argument(manifest,role_name,"--moe-pack-root",moe_pack_root)


def set_dspark_arguments(
        manifest,enabled,model_dir,manifest_path,maximum_context_tokens):
    gateway_role = "spark0_gateway"
    resident_role = "pp13_cuda_residentd"
    set_role_switch(manifest,gateway_role,"--dspark",enabled)
    set_role_switch(manifest,resident_role,"--dspark",enabled)
    for argument in (
            "--dspark-config",
            "--dspark-manifest",
            "--dspark-safetensors",
            "--dspark-max-context"):
        remove_role_argument(manifest,resident_role,argument)
    if not enabled:
        if model_dir is not None or manifest_path is not None:
            raise SystemExit(
                "--dspark-model-dir and --dspark-manifest require --dspark")
        return
    if model_dir is None:
        raise SystemExit("--dspark-model-dir is required with --dspark")
    if manifest_path is None:
        raise SystemExit("--dspark-manifest is required with --dspark")
    if maximum_context_tokens < 1:
        raise SystemExit("--dspark-max-context must be positive")
    set_role_argument(
        manifest,resident_role,"--dspark-config",
        os.path.join(model_dir,"config.json"))
    set_role_argument(
        manifest,resident_role,"--dspark-manifest",manifest_path)
    set_role_argument(
        manifest,resident_role,"--dspark-safetensors",
        os.path.join(model_dir,"model.safetensors"))
    set_role_argument(
        manifest,resident_role,"--dspark-max-context",
        maximum_context_tokens)


def set_role_switch(manifest,role_name,argument,enabled):
    matching_roles = [role for role in manifest["roles"] if role.get("name") == role_name]
    if len(matching_roles) != 1:
        raise SystemExit("release must contain exactly one role named " + role_name)
    arguments = matching_roles[0].setdefault("argv",[])
    matches = [index for index,item in enumerate(arguments) if item == argument]
    if len(matches) > 1:
        raise SystemExit("role switch occurs more than once: " + argument)
    if matches:
        arguments.pop(matches[0])
    if enabled:
        arguments.append(argument)


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


def set_runtime_diagnostics(manifest,enabled):
    for role in manifest["roles"]:
        environment = role.get("env",[])
        role["env"] = [entry for entry in environment
                       if entry.split("=",1)[0] not in
                       DIAGNOSTIC_ENVIRONMENT_NAMES]
    for role_name,values in DIAGNOSTIC_ENVIRONMENT_BY_ROLE.items():
        matching_roles = [role for role in manifest["roles"]
                          if role.get("name") == role_name]
        if len(matching_roles) != 1:
            raise SystemExit(
                "release must contain exactly one role named " + role_name)
        role = matching_roles[0]
        if enabled:
            role["env"].extend(key + "=" + value
                               for key,value in values.items())


def set_role_environment(manifest,specification):
    try:
        role_name,assignment = specification.split("=",1)
        name,value = assignment.split("=",1)
    except ValueError:
        raise SystemExit("role environment must be ROLE=NAME=VALUE")
    if not role_name or not name:
        raise SystemExit("role environment must be ROLE=NAME=VALUE")
    matching_roles = [role for role in manifest["roles"]
                      if role.get("name") == role_name]
    if len(matching_roles) != 1:
        raise SystemExit(
            "release must contain exactly one role named " + role_name)
    environment = matching_roles[0].setdefault("env",[])
    environment = [entry for entry in environment
                   if entry.split("=",1)[0] != name]
    environment.append(name + "=" + value)
    matching_roles[0]["env"] = environment


def unset_role_environment(manifest,specification):
    try:
        role_name,name = specification.split("=",1)
    except ValueError:
        raise SystemExit("role environment removal must be ROLE=NAME")
    if not role_name or not name or "=" in name:
        raise SystemExit("role environment removal must be ROLE=NAME")
    matching_roles = [role for role in manifest["roles"]
                      if role.get("name") == role_name]
    if len(matching_roles) != 1:
        raise SystemExit(
            "release must contain exactly one role named " + role_name)
    environment = matching_roles[0].setdefault("env",[])
    matching_roles[0]["env"] = [entry for entry in environment
                                        if entry.split("=",1)[0] != name]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--release-id",required=True)
    parser.add_argument("--git-commit",required=True)
    parser.add_argument("--max-active",type=int)
    parser.add_argument("--kv-pool-tokens",type=int)
    parser.add_argument("--kv-logical-blocks",type=int,required=True)
    parser.add_argument(
        "--model-quantization",choices=["fp8","nvfp4"],default="fp8")
    parser.add_argument("--stagepack-root")
    parser.add_argument("--moe-pack-root")
    decode_mode = parser.add_mutually_exclusive_group(required=True)
    decode_mode.add_argument("--mtp",action="store_true")
    decode_mode.add_argument("--plain-decode",action="store_true")
    parser.add_argument("--dspark",action="store_true")
    parser.add_argument("--dspark-model-dir")
    parser.add_argument("--dspark-manifest")
    parser.add_argument("--dspark-max-context",type=int,default=2048)
    parser.add_argument("--without-diagnostics",action="store_true")
    parser.add_argument("--role-env",action="append",default=[])
    parser.add_argument("--role-env-unset",action="append",default=[])
    parser.add_argument("--replace",action="append",default=[])
    arguments = parser.parse_args()
    temporary = arguments.output + ".assembling." + str(os.getpid())
    if os.path.exists(arguments.output) or os.path.exists(temporary):
        raise SystemExit("release output already exists")
    shutil.copytree(arguments.template,temporary)
    manifest_path = os.path.join(temporary,"sparkpipe.json")
    with open(manifest_path,"r",encoding="utf-8") as source:
        manifest = json.load(source)
    allowed = {entry["path"] for entry in manifest["files"]}
    for replacement in arguments.replace:
        relative,source = replacement.split("=",1)
        if relative not in allowed:
            raise SystemExit("replacement is not in manifest: " + relative)
        if not os.path.isfile(source):
            raise SystemExit("missing replacement: " + source)
        shutil.copy2(source,os.path.join(temporary,relative))
    manifest["release_id"] = arguments.release_id
    manifest["git_commit"] = arguments.git_commit
    manifest["generation"] = int(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S"))
    set_role_release_identity(manifest)
    if arguments.max_active is not None:
        if arguments.max_active < 1 or arguments.max_active > 1024:
            raise SystemExit("max-active must be in 1..1024")
        set_role_max_active(manifest,arguments.max_active)
    if arguments.kv_pool_tokens is not None:
        if arguments.kv_pool_tokens < 1:
            raise SystemExit("kv-pool-tokens must be positive")
        set_role_argument(
            manifest,"pp13_cuda_residentd","--kv-pool-tokens",
            arguments.kv_pool_tokens)
    if arguments.kv_logical_blocks < 1:
        raise SystemExit("kv-logical-blocks must be positive")
    set_role_argument(
        manifest,"spark0_gateway","--kv-logical-blocks",
        arguments.kv_logical_blocks)
    remove_role_argument(
        manifest,"spark0_gateway","--decode-batch-target")
    set_model_arguments(
        manifest,
        arguments.model_quantization,
        arguments.moe_pack_root,
        arguments.stagepack_root)
    set_role_switch(manifest,"spark0_gateway","--mtp",arguments.mtp)
    set_role_switch(manifest,"pp13_cuda_residentd","--mtp",arguments.mtp)
    set_dspark_arguments(
        manifest,
        arguments.dspark,
        arguments.dspark_model_dir,
        arguments.dspark_manifest,
        arguments.dspark_max_context)
    set_runtime_diagnostics(manifest,not arguments.without_diagnostics)
    for specification in arguments.role_env_unset:
        unset_role_environment(manifest,specification)
    for specification in arguments.role_env:
        set_role_environment(manifest,specification)
    write_manifest(temporary,manifest)
    os.rename(temporary,arguments.output)


if __name__ == "__main__":
    main()
