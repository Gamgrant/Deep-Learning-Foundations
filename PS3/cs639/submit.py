import os
import zipfile


_PS3_FILES = [
    "fcos.py",
    "fcos.ipynb",
    "fcos_detector.pt",
]



def make_ps3_submission(assignment_path, uniquename=None):
    _make_submission(assignment_path, _PS3_FILES, "PS3", uniquename)


def _make_submission(
    assignment_path, file_list, assignment_no, uniquename=None):
    if uniquename is None:
        uniquename = _get_user_info()
    zip_path = "{}_{}.zip".format(uniquename, assignment_no)
    zip_path = os.path.join(assignment_path, zip_path)
    print("Writing zip file to: ", zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for filename in file_list:
            in_path = os.path.join(assignment_path, filename)
            if not os.path.isfile(in_path):
                raise ValueError('Could not find file "%s"' % filename)
            zf.write(in_path, filename)


def _get_user_info():
    unq = input("Enter firstname1_firstname2 as your uniquename (e.g., Josef Pieper and Mel Gibson -> joseph_mel): ")
    return unq
