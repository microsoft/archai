# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Mapping, MutableMapping, Optional

COPY_NODE_KEY = "_copy"  # For copying the content of a node (must be dict)
COPY_VALUE_PREFIX = "_copy:"  # For copying the value of a node (must be scalar)


def _merge_dicts(source: Mapping, destination: MutableMapping) -> None:
    # Copy anything that source has but destination does not have
    for source_key in source:
        if source_key not in destination:
            destination[source_key] = source[source_key]
        else:
            source_value = source[source_key]
            destination_value = destination[source_key]

            # Recursively merge child nodes
            if isinstance(source_value, Mapping) and isinstance(destination_value, MutableMapping):
                _merge_dicts(source_value, destination_value)


def _concatenate_paths(path1: str, path2: str) -> str:
    def _normalize_path(path: str) -> str:
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return path

    split_point = 1 if path1.endswith("/") else 0
    split_point += 1 if path2.startswith("/") else 0

    if split_point == 0:
        concat_path = path1 + "/" + path2
    elif split_point == 1:
        concat_path = path1 + path2
    else:
        concat_path = path1[:-1] + path2

    return _normalize_path(concat_path)


def _is_path_valid(path: str) -> bool:
    return path.startswith("/") and (len(path) == 1 or not path.endswith("/"))


def _get_absolute_path(src_folder: str, rel_path: str) -> str:
    assert len(src_folder) > 0 and src_folder.startswith("/"), "`src_folder` must be an absolute path"

    rel_paths = rel_path.split("/")
    if rel_path.startswith("/"):
        src_folder_paths = []  # `rel_path` is absolute path so ignore `src_folder`
    else:
        src_folder_paths = src_folder.split("/")
    full_paths = src_folder_paths + rel_paths

    final_paths = []
    for i in range(len(full_paths)):
        path = full_paths[i].strip()
        if not path or path == ".":  # Remove blank strings and single dots
            continue

        if path == "..":
            if len(final_paths):
                final_paths.pop()
            else:
                raise RuntimeError(f"cannot create abs path for src_folder={src_folder} and rel_path={rel_path}")
        else:
            final_paths.append(path)

    final_path = "/" + "/".join(final_paths)  # Should work even when `final_paths` is empty
    assert ".." not in final_path and _is_path_valid(final_path)

    return final_path


def _get_path_to_resolve(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.startswith(COPY_VALUE_PREFIX):
        # Almost always have space after _copy command
        return value[len(COPY_VALUE_PREFIX) :].strip()

    return None


def _resolve_path(root_dict: MutableMapping, path: str, visited_paths: set) -> Any:
    assert _is_path_valid(path)

    # Traverse path in root dict hierarchy
    current_path = "/"  # Path at each iteration of for loop
    current_dict = root_dict
    for path in path.split("/"):
        if not path:
            continue  # There will be blank vals at start

        # For each path, we need to be able find key in dict but some dics may not be fully resolved yet
        # For last key, `current_dict` will be either dict or other value
        if isinstance(current_dict, Mapping):
            # For this section, make sure everything is resolved before we search for the key
            _resolve_dict(root_dict, current_dict, current_path, visited_paths)

            if path in current_dict:
                # "cd" into child node
                current_dict = current_dict[path]
                current_path = _concatenate_paths(current_path, path)
            else:
                raise RuntimeError(f"Path `{path}` could not be found in specified dictionary at `{path}`.")
        else:
            raise KeyError(
                f"Path `{path}` cannot be resolved because `{current_path}` is not a dictionary so `{path}` cannot exist in it."
            )

    # last child is our answer
    resolved_path = _get_path_to_resolve(current_dict)
    if resolved_path:
        next_path = _get_absolute_path(current_path, resolved_path)
        if next_path == path:
            raise RuntimeError(f"Cannot resolve path `{path}` because it is circular reference.")
        current_dict = _resolve_path(root_dict, next_path, visited_paths)

    return current_dict


def _resolve_dict(
    root_dict: MutableMapping, current_dict: MutableMapping, current_path: str, visited_paths: set
) -> None:
    assert _is_path_valid(current_path)

    if current_path in visited_paths:
        return  # Avoids infinite recursion
    visited_paths.add(current_path)

    child_path = current_dict.get(COPY_NODE_KEY, None)
    if child_path and isinstance(child_path, str):
        child_dict = _resolve_path(root_dict, _get_absolute_path(current_path, child_path), visited_paths)
        if not isinstance(child_dict, Mapping):
            raise RuntimeError(f"Path `{child_path}` should be dictionary but its instead `{child_dict}`.")

        _merge_dicts(child_dict, current_dict)
        del current_dict[COPY_NODE_KEY]

    for key in current_dict.keys():
        resolved_path = _get_path_to_resolve(current_dict[key])
        if resolved_path:
            current_dict[key] = _resolve_path(
                root_dict, _get_absolute_path(_concatenate_paths(current_path, key), resolved_path), visited_paths
            )

        # Recursively resolve values in nested dicts
        if isinstance(current_dict[key], MutableMapping):
            _resolve_dict(root_dict, current_dict[key], _concatenate_paths(current_path, key), visited_paths)


def resolve_dict(root_dict: MutableMapping) -> None:
    """Resolve all values in a dictionary recursively.

    Args:
        root_dict: Root dictionary.

    """

    _resolve_dict(root_dict, root_dict, "/", set())
