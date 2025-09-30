import os
def check_svpaths(svpath):
    if not os.path.exists(svpath):
        print("Warning: Path save directory does not exist!")
        os.makedirs(svpath)
        print("Path created!")
    else:
        print("Save path exists: %s"%svpath)    
