# -*- coding: utf-8 -*-
import os
import sys
import shutil


def main():
    home = os.path.expanduser("~")
    mnt = '/mnt/'
    media = '/media/'
    sympy_import = ''
    # Check if ipython3 is installed
    if not shutil.which("ipython3"):
        print("Error: ipython3 is not installed or not in your PATH.")
        sys.exit(1)
    # if -h/--help is passed, show help message
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
GrasPy DOCUMENTATION
`graspy` is a command-line tool that calls an interactive Python 
shell (ipython3) with the option to pass a path to wehre store the
data (or read any existing data).

Options:
--------
no option : Initialize an ipython3 --pylab='qt' shell

-f <path> : Option to pass a base data folder path.
              
-s |--sympy : Option to initiate a sympy profile.

-h |--help : Shows this help message

        """)
        sys.exit(0)
    elif len(sys.argv) > 2 and sys.argv[1] == '-f' and any([sys.argv[2] != '', sys.argv[2] != None]):
        config_path = sys.argv[2]
        if not any([config_path.startswith(home), config_path.startswith(mnt), config_path.startswith(media)]):
            config_path = os.path.join(home, config_path)
        try:
            if any(['-s' in sys.argv, '--sympy' in sys.argv]):
                sympy_import = ';from sympy import *; init_session(quiet=True, use_latex=True)'
            print("\n Initiating IPython Shell. Importing GRASP...\n")
            os.system(f"export GRASPDATA={config_path} && ipython3 --pylab='qt' -i -c 'import grasp{sympy_import}'")
        except OSError as ose:
            print(f"Error: {ose}")
            sys.exit(1)
    elif len(sys.argv) == 1:
        os.system("ipython3 --pylab='qt'")
    else: # Handle invalid arguments
        print("Error: Invalid use. Use -h or --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()