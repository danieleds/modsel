import sys
import modsel


def main():
    r = modsel.main()
    if r != 0:
        sys.exit(r)
