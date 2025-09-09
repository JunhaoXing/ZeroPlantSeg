import os
import glob

def filename_wo_ext(path: str) -> str:
    """return os.path.splitext(os.path.basename(path))[0]

    Args:
        path (str): _description_

    Returns:
        str: _description_
    """
    return os.path.splitext(os.path.basename(path))[0]

def basename(path: str) -> str:
    if path[-1] == "/":
        return os.path.basename(os.path.dirname(path))
    else:
        return os.path.basename(path)

def cat_ref_paths(trg_dir:str, ref_paths: list, WO_EXT=False) -> list:
    if WO_EXT:
        return [os.path.join(trg_dir, filename_wo_ext(ref_path)) for ref_path in ref_paths]
    else:
        return [os.path.join(trg_dir, basename(ref_path)) for ref_path in ref_paths]

def find_samefiles(dir1: str, dir2: str) -> list:
    """find same filenames between dir1 and dir2

    Args:
        dir1 (str): _description_
        dir2 (str): _description_

    Returns:
        list: _description_
    """
    dir1_paths = glob.glob(os.path.join(dir1, "*"))
    
    res_filenames = []
    for dir1_path in dir1_paths:
        filename = basename(dir1_path)
        if os.path.isfile(os.path.join(dir2, filename)):
            res_filenames.append(filename)
    
    return res_filenames