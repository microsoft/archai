# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Mapping, MutableMapping, Optional

COPY_NODE_KEY = "_copy"  # For copying the content of a node (must be dict)
COPY_VALUE_PREFIX = "_copy:"  # For copying the value of a node (must be scalar)


def merge_dicts(source: Mapping, destination: MutableMapping) -> None:
    """Recursively merge dictionaries.
    
    If a key is present in both `source` and `destination`, the value in `destination` is
    overwritten with the value in `source`.
    
    Args:
        source: Source dictionary.
        destination: Destination dictionary.
        
    """

    # Copy anything that source has but destination doesn't
    for source_key in source:
        if source_key not in destination:
            destination[source_key] = source[source_key]
        else:
            source_value = source[source_key]
            destination_value = destination[source_key]

            # Recursively merge child nodes
            if isinstance(source_value, Mapping) and isinstance(destination_value, MutableMapping):
                merge_dicts(source_value, destination_value)


def concatenate_paths(path1: str, path2: str) -> str:
    """Concatenates two paths.
    
    For example, `path1=/a/b/c` and `path2=d/e` should return `/a/b/c/d/e`.
    
    Args:
        path1: First path.
        path2: Second path.
        
    Returns:
        Concatenated path.
        
    """

    def _normalize_path(path: str) -> str:
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return path

    mid = 1 if path1.endswith("/") else 0
    mid += 1 if path2.startswith("/") else 0

    # only 3 possibilities
    if mid == 0:
        res = path1 + "/" + path2
    elif mid == 1:
        res = path1 + path2
    else:
        res = path1[:-1] + path2

    return _normalize_path(res)


def is_path_valid(path: str) -> bool:
    """Checks if a path is valid.
    
    Args:
        path: Path to check.
        
    Returns:
        `True` if path is valid, `False` otherwise.
        
    """

    return path.startswith("/") and (len(path) == 1 or not path.endswith("/"))


def get_absolute_path(current_working_directory: str, relative_path: str) -> str:
    """Returns an absolute path given a current working directory and a relative path.
    
    Args:
        current_working_directory: Current working directory.
        relative_path: Relative path.
        
    Returns:
        Absolute path.
        
    """

    assert len(current_working_directory) > 0 and current_working_directory.startswith(
        "/"
    ), "current_working_directory must be an absolute path"

    relative_parts = relative_path.split("/")
    if relative_path.startswith("/"):
        current_working_directory_parts = []  # relative_path is absolute path so ignore current_working_directory
    else:
        current_working_directory_parts = current_working_directory.split("/")
    full_parts = current_working_directory_parts + relative_parts

    final_parts = []
    for i in range(len(full_parts)):
        part = full_parts[i].strip()
        if not part or part == ".":  # remove blank strings and single dots
            continue
        if part == "..":
            if len(final_parts):
                final_parts.pop()
            else:
                raise RuntimeError(
                    f"cannot create abs path for current_working_directory={current_working_directory} and relative_path={relative_path}"
                )
        else:
            final_parts.append(part)

    final_path = "/" + "/".join(final_parts)  # should work even when final_parts is empty
    assert ".." not in final_path and is_path_valid(final_path)  # make algorithm indeed worked
    return final_path


def get_path_to_resolve(value: Any) -> Optional[str]:
    """Returns path to resolve if value is a copy node, otherwise returns None.
    
    Args:
        value: Value to check.
        
    Returns:
        Path to resolve if value is a copy node, otherwise returns None.
    
    """

    if isinstance(value, str) and value.startswith(COPY_VALUE_PREFIX):
        # we will almost always have space after _copy command
        return value[len(COPY_VALUE_PREFIX) :].strip()
    return None


def resolve_path(root_dict: MutableMapping, path: str, visited_paths: set) -> Any:
    """Resolves a path in a dictionary.
    
    Args:
        root_dict: Root dictionary.
        path: Path to resolve.
        visited_paths: Set of paths that have already been visited.
        
    Returns:
        Value at path.
        
    """

    assert is_path_valid(path)

    # traverse path in root dict hierarchy
    current_path = "/"  # path at each iteration of for loop
    current_dict = root_dict
    for part in path.split("/"):
        if not part:
            continue  # there will be blank vals at start

        # For each part, we need to be able find key in dict but some dics may not
        # be fully resolved yet. For last key, current_dict will be either dict or other value.
        if isinstance(current_dict, Mapping):
            # for this section, make sure everything is resolved
            # before we prob for the key
            resolve_values_recursively(root_dict, current_dict, current_path, visited_paths)

            if part in current_dict:
                # "cd" into child node
                current_dict = current_dict[part]
                current_path = concatenate_paths(current_path, part)
            else:
                raise RuntimeError(f'Path {path} could not be found in specified dictionary at "{part}"')
        else:
            raise KeyError(
                f'Path "{path}" cannot be resolved because "{current_path}" is not a dictionary so "{part}" cannot exist in it'
            )

    # last child is our answer
    resolved_path = get_path_to_resolve(current_dict)
    if resolved_path:
        next_path = get_absolute_path(current_path, resolved_path)
        if next_path == path:
            raise RuntimeError(f'Cannot resolve path "{path}" because it is circular reference')
        current_dict = resolve_path(root_dict, next_path, visited_paths)
    return current_dict


def resolve_values_recursively(
    root_dict: MutableMapping, current_dict: MutableMapping, current_path: str, visited_paths: set
) -> None:
    """Resolves values in a dictionary recursively.
    
    Args:
        root_dict: Root dictionary.
        current_dict: Current dictionary.
        current_path: Current path.
        visited_paths: Set of paths that have already been visited.
        
    """

    assert is_path_valid(current_path)

    if current_path in visited_paths:
        return  # to avoid infinite recursion
    visited_paths.add(current_path)

    # check if current dict has a copy node key
    child_path = current_dict.get(COPY_NODE_KEY, None)
    if child_path and isinstance(child_path, str):
        # resolve the path to get the source dict
        child_dict = resolve_path(root_dict, get_absolute_path(current_path, child_path), visited_paths)
        # ensure the target path points to a dict that can be merged
        if not isinstance(child_dict, Mapping):
            raise RuntimeError(f'Path "{child_path}" should be dictionary but its instead "{child_dict}"')
        # merge keys that have not been overridden
        merge_dicts(child_dict, current_dict)
        # remove the copy node key
        del current_dict[COPY_NODE_KEY]

    for key in current_dict.keys():
        # check if this key needs path resolution
        resolved_path = get_path_to_resolve(current_dict[key])
        if resolved_path:
            current_dict[key] = resolve_path(
                root_dict, get_absolute_path(concatenate_paths(current_path, key), resolved_path), visited_paths
            )
        # recursively resolve values in nested dicts
        if isinstance(current_dict[key], MutableMapping):
            resolve_values_recursively(
                root_dict, current_dict[key], concatenate_paths(current_path, key), visited_paths
            )


def resolve_all_values(root_dict: MutableMapping) -> None:
    """Resolves all values in a dictionary recursively.
    
    Args:
        root_dict: Root dictionary.
        
    """

    resolve_values_recursively(root_dict, root_dict, "/", set())
