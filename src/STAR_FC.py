import sys
import getopt
from Settings import Settings


def usage():
    print('Usage: python3 STAR_FC.py [options]')
    print('STAR_FC v1.1')
    print('Application for predicting human fixations on static images')
    print('Options:')
    print('-h, --help\t\t', 'Displays this help')
    print('-c <configFilePath>\t', 'Full path to confi file')
    print('-v,\t\t\t', 'Visualize results')

def main(argv):
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:v', ['help','configFile', 'verbose'])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)

    visualize = False
    iniFilePath = None

    for o, a in opts:
        if o == "-v":
            visualize = True
        elif o in ["-h", "--help"]:
            usage()
            sys.exit(2)
        elif o == "-c":
            iniFilePath = a

    if not iniFilePath:
        print('ERROR: .ini config file not provided!')
        usage()
        sys.exit(2)

    from Controller import Controller

    settings = Settings(iniFilePath, visualize)
    controller = Controller(settings)
    controller.run()

if __name__ == '__main__':
    main(sys.argv)
