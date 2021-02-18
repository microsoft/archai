from typing import Iterable, List, Mapping, OrderedDict, Tuple

from archai.common import utils

class DelimitedText:
    def __init__(self)->None:
        self._data:OrderedDict[str,List[str]] = OrderedDict()

    def add_from_file(self, filepath:str, has_header:bool, delimiter:str='\t')->None:
        filepath = utils.full_path(filepath)
        header = None if has_header else []
        with open(filepath, 'r') as f:
            line = f.readline()
            while line:
                cols = line.rstrip('\n').split(sep=delimiter)
                if header is None:
                    header = cols
                else:
                    self.add_from_cols(cols, header)

    def add_from_text(self, text:str, has_header:bool, delimiter:str='\t')->None:
        header = None if has_header else []
        for line in text.splitlines():
            cols = line.rstrip('\n').split(sep=delimiter)
            if header is None:
                header = cols
            else:
                self.add_from_cols(cols, header)

    def add_from_cols(self, cols:Iterable, header:List[str])->None:
        for i, col in enumerate(cols):
            key = header[i] if len(header) > i else str(i)
            if key not in self._data:
                self._data[key] = []
            self._data[key].append(str(col))

    def get_col(self, col_name:str)->List[str]:
        return self._data[col_name]

    def set_col(self, col_name:str, vals:List[str])->None:
        self._data[col_name] = vals

    def set_data(self, d:Mapping[str, List[str]])->None:
        self._data = OrderedDict(d)

    def add_from_cols_list(self, cols_list:Iterable[Iterable], header:List[str])->None:
        for cols in cols_list:
            self.add_from_cols(cols, header)

    def save(self, filepath:str, has_header=True, delimiter:str='\t')->None:
        keys = list(self._data.keys())
        with open(filepath, 'w') as f:
            if has_header:
                f.write(delimiter.join(keys) + '\n')
            for vals in zip(*(self._data[key] for key in keys)):
                f.write(delimiter.join(vals) + '\n')

    def __len__(self)->int:
        return len(self._data)