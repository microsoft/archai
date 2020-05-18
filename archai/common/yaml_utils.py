from typing import Mapping, MutableMapping, Any, Optional

_PREFIX_NODE = '_copy'
_PREFIX_PATH = '_copy:'


def resolve_all(root_d:MutableMapping, cur:MutableMapping):
    # if cur dict has '_copy' node with path in it
    child_path = cur.get(_PREFIX_NODE, None)
    if child_path and isinstance(child_path, str):
        # resolve this path to get source dict
        child_d = _resolve_path(root_d, child_path)
        if not isinstance(child_d, Mapping):
            raise RuntimeError(f'Path "{child_path}" should be dictionary but its instead "{child_d}"')
        # replace keys that have not been overriden
        _merge_source(child_d, cur)
        del cur[_PREFIX_NODE]

    for k in cur.keys():
        rpath = _req_resolve(cur[k])
        if rpath:
            cur[k] = _resolve_path(root_d, rpath)
        if isinstance(cur[k], MutableMapping):
            resolve_all(root_d, cur[k])

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
    if isinstance(v, str) and v.startswith(_PREFIX_PATH):
        return v[len(_PREFIX_PATH):]
    return None

def _resolve_path(root_d:MutableMapping, path:str)->Any:
    assert path # otherwise we end up returning root for null paths
    d = root_d
    cur_path = '' # maintained for debugging

    parts = path.strip().split('/')
    # if path starts with '/' remove the first part
    if len(parts)>0 and parts[0]=='':
        parts = parts[1:]
    if len(parts)>0 and parts[-1]=='':
        parts = parts[:-1]

    # traverse path in root dict hierarchy
    for part in parts:
        # make sure current node is dict so we can "cd" into it
        if isinstance(d, Mapping):
            # if path doesn't exit in current dir, see if there are any copy commands here
            if part not in d:
                resolve_all(root_d, d)
            # if path do exist but is string with copy command and then resolve it first
            else:
                rpath = _req_resolve(d[part])
                if rpath:
                    d[part] = _resolve_path(root_d, rpath)
                # else resolution already done

        # at this point we should have dict to "cd" into otherwise its an error
        if  isinstance(d, Mapping) and part in d:
            # "cd" into child node
            d = d[part]
            cur_path += '/' + part
        else:
            raise KeyError(f'Path {path} cannot be resolved because {part} in {cur_path} does not exist or is not a dictionary')

    # last child is our answer
    rpath = _req_resolve(d)
    if rpath:
        d = _resolve_path(root_d, rpath)
    return d