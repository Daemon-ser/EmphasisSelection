import torch.distributed as dist
import os
def del_file(filename):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    if os.path.isfile(filename):
        os.remove(filename)
        return True
    return False


def cleanup():
    dist.destroy_process_group()
