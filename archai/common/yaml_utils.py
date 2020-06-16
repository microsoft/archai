# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import Mapping, MutableMapping, Any, Optional


_PREFIX_NODE = '_copy' # for copy node content command (must be dict)
_PREFIX_PATH = '_copy:' # for copy node value command (must be scaler)


def resolve_all(root_d:MutableMapping):
    _resolve_all(root_d, root_d, '/', set())

def _resolve_all(root_d:MutableMapping, cur:MutableMapping, cur_path:str, prev_paths:set):
    assert is_proper_path(cur_path)

    if cur_path in prev_paths:
        return # else we get in to infinite recursion
    prev_paths.add(cur_path)

    # if cur dict has '_copy' node with path in it
    child_path = cur.get(_PREFIX_NODE, None)
    if child_path and isinstance(child_path, str):
        # resolve this path to get source dict
        child_d = _resolve_path(root_d, _rel2full_path(cur_path, child_path), prev_paths)
        # we expect target path to point to dict so we can merge its keys
        if not isinstance(child_d, Mapping):
            raise RuntimeError(f'Path "{child_path}" should be dictionary but its instead "{child_d}"')
        # replace keys that have not been overriden
        _merge_source(child_d, cur)
        # remove command key
        del cur[_PREFIX_NODE]

    for k in cur.keys():
        # if this key needs path resolution, get target and replace the value
        rpath = _req_resolve(cur[k])
        if rpath:
            cur[k] = _resolve_path(root_d,
                        _rel2full_path(_join_path(cur_path, k), rpath), prev_paths)
        # if replaced value is again dictionary, recurse on it
        if isinstance(cur[k], MutableMapping):
            _resolve_all(root_d, cur[k], _join_path(cur_path, k), prev_paths)

def _merge_source(source:Mapping, dest:MutableMapping)->None:
    # for anything that source has but dest doesn't, just do copy
    for sk in source:
        if sk not in dest:
            dest[sk] = source[sk]
        else:
            sv = source[sk]
            dv = dest[sk]

            # recursively merge child nodes
            if isinstance(sv, Mapping) and isinstance(dv, MutableMapping):
                _merge_source(source[sk], dest[sk])
            # else at least dest value is not dict and should not be overriden

def _req_resolve(v:Any)->Optional[str]:
    """If the value is actually a path we need resolve then return that path or return None"""
    if isinstance(v, str) and v.startswith(_PREFIX_PATH):
        # we will almost always have space after _copy command
        return v[len(_PREFIX_PATH):].strip()
    return None

def _join_path(path1:str, path2:str):
    mid = 1 if path1.endswith('/') else 0
    mid += 1 if path2.startswith('/') else 0

    # only 3 possibilities
    if mid==0:
        res = path1 + '/' + path2
    elif mid==1:
        res = path1 + path2
    else:
        res = path1[:-1] + path2

    return _norm_ended(res)

def _norm_ended(path:str)->str:
    if len(path) > 1 and path.endswith('/'):
        path = path[:-1]
    return path

def is_proper_path(path:str)->bool:
    return path.startswith('/') and (len(path)==1 or not path.endswith('/'))

def _rel2full_path(cwd:str, rel_path:str)->str:
    """Given current directory and path, we return abolute path. For example,
    cwd='/a/b/c' and rel_path='../d/e' should return '/a/b/d/e'. Note that rel_path
    can hold absolute path in which case it will start with '/'
    """
    assert len(cwd) > 0 and cwd.startswith('/'), 'cwd must be absolute path'

    rel_parts = rel_path.split('/')
    if rel_path.startswith('/'):
        cwd_parts = [] # rel_path is absolute path so ignore cwd
    else:
        cwd_parts = cwd.split('/')
    full_parts = cwd_parts + rel_parts

    final = []
    for i in range(len(full_parts)):
        part = full_parts[i].strip()
        if not part or part == '.': # remove blank strings and single dots
            continue
        if part == '..':
            if len(final):
                final.pop()
            else:
                raise RuntimeError(f'cannot create abs path for cwd={cwd} and rel_path={rel_path}')
        else:
            final.append(part)

    final = '/' + '/'.join(final)  # should work even when final is empty
    assert not '..' in final and is_proper_path(final) # make algo indeed worked
    return final


def _resolve_path(root_d:MutableMapping, path:str, prev_paths:set)->Any:
    """For given path returns value or node from root_d"""

    assert is_proper_path(path)

    # traverse path in root dict hierarchy
    cur_path = '/' # path at each iteration of for loop
    d = root_d
    for part in path.split('/'):
        if not part:
            continue # there will be blank vals at start

        # For each part, we need to be able find key in dict but some dics may not
        # be fully resolved yet. For last key, d will be either dict or other value.
        if isinstance(d, Mapping):
            # for this section, make sure everything is resolved
            # before we prob for the key
            _resolve_all(root_d, d, cur_path, prev_paths)

            if part in d:
                # "cd" into child node
                d = d[part]
                cur_path = _join_path(cur_path, part)
            else:
                raise RuntimeError(f'Path {path} could not be found in specified dictionary at "{part}"')
        else:
            raise KeyError(f'Path "{path}" cannot be resolved because "{cur_path}" is not a dictionary so "{part}" cannot exist in it')

    # last child is our answer
    rpath = _req_resolve(d)
    if rpath:
        next_path = _rel2full_path(cur_path, rpath)
        if next_path == path:
            raise RuntimeError(f'Cannot resolve path "{path}" because it is circular reference')
        d = _resolve_path(root_d, next_path, prev_paths)
    return d