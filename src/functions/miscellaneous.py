from shutil import rmtree
from os.path import exists, isdir
from os import mkdir, makedirs

def ensure_clean_dir(path):
    if exists(path):
        rmtree(path)
    mkdir(path)
    
def ensure_dir(path):
    if not isdir(path):
        makedirs(path)