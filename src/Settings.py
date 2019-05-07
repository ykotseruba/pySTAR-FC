#parse settings from .ini file

import configparser

class Settings:
    def __init__(self, iniPath, visualize):
        self.iniPath = iniPath
        self.visualize = visualize

        #input params
        self.input = None
        self.batch = None

        #attention map params
        self.PeriphSalAlgorithm = ''
        self.AIMBasis = ''
        self.CentralSalAlgorithm = ''
        self.pgain = -1
        self.cgain = -1
        self.blendingStrategy = -1
        self.nextFixAsMax = False
        self.nextFixThresh = -1
        self.pSizeDeg = -1
        self.cSizeDeg = -1
        self.iorSizeDeg = -1
        self.iorDecayRate = -1

        # viewing params
        self.inputSizeDeg = -1
        self.viewDist = -1
        self.maxNumFixations = -1
        self.maxSubjects = -1
        self.paddingRGB = [-1, -1, -1]
        self.foveate = False

        #log params
        self.saveDir = ''
        self.saveFix = False
        self.saveScreen = False
        self.overwrite = False #overwrite previously saved results of computation

        self.readINI()

    def readINI(self):

        iniReader = configparser.ConfigParser()
        iniReader.read(self.iniPath)

        self.input = iniReader['input_params'].get('input', fallback=None)
        self.batch = iniReader['input_params'].get('batch', fallback=None)


        if self.input and self.batch:
            raise ValueError('Conflicting options provided. Cannot run with both input and batch in the .ini file.')

        if not self.input and not self.batch:
            raise ValueError('No input options provided. Provide either single image input or directory in the .ini file.')

        self.PeriphSalAlgorithm = iniReader['attention_map_params'].get('PeriphSalAlgorithm', fallback='AIM')
        self.AIMBasis = iniReader['attention_map_params'].get('AIMBasis', fallback='data/21infomax950.mat')
        self.CentralSalAlgorithm = iniReader['attention_map_params'].get('CentralSalAlgorithm', fallback='DeepGazeII')
        self.pgain = iniReader['attention_map_params'].getfloat('pgain')
        self.cgain = iniReader['attention_map_params'].getfloat('cgain', fallback=1.0)
        self.blendingStrategy = iniReader['attention_map_params'].getint('blendingStrategy', fallback=2)
        self.nextFixAsMax = iniReader['attention_map_params'].getboolean('nextFixAsMax', fallback=True)
        self.nextFixThresh = iniReader['attention_map_params'].getfloat('nextFixThresh')
        self.pSizeDeg = iniReader['attention_map_params'].getfloat('pSizeDeg', fallback=9.5)
        self.cSizeDeg = iniReader['attention_map_params'].getfloat('cSizeDeg', fallback=9.6)
        self.iorSizeDeg = iniReader['attention_map_params'].getfloat('iorSizeDeg', fallback=1.5)
        self.iorDecayRate = iniReader['attention_map_params'].getint('iorDecayRate', fallback=10)

        if not self.pgain:
            raise ValueError('pgain value is not provided in the .ini file!')

        if not self.nextFixAsMax and not self.nextFixThresh:
            raise ValueError('nextFixThresh is not provided in the .ini file')

        self.pix2deg = iniReader['viewing_params'].getint('pix2deg')
        self.inputSizeDeg = iniReader['viewing_params'].getfloat('inputSizeDeg')
        self.viewDist = iniReader['viewing_params'].getfloat('viewDist')
        self.foveate = iniReader['viewing_params'].getboolean('foveate', fallback=False)
        self.rodsAndCones = iniReader['viewing_params'].getboolean('rodsAndCones', fallback=False)
        self.maxNumFixations = iniReader['viewing_params'].getint('maxNumFixations')
        self.numSubjects = iniReader['viewing_params'].getint('numSubjects')

        paddingR = iniReader['viewing_params'].getint('paddingR')
        paddingG = iniReader['viewing_params'].getint('paddingG')
        paddingB = iniReader['viewing_params'].getint('paddingB')

        if paddingR and paddingG and paddingB:
            self.paddingRGB = [int(paddingR), int(paddingG), int(paddingB)]

        if self.nextFixAsMax and self.numSubjects > 1:
            raise ValueError('STAR-FC is running in deterministic mode (nextFixAsMax=True) but numSubjects is > 1')

        if self.pix2deg and self.inputSizeDeg:
            raise ValueError('conflicting options (pix2deg and inputSizeDeg) provided. Only one should be present in the .ini file!')

        if not self.viewDist or not (self.pix2deg or self.inputSizeDeg):
            raise ValueError('viewing parameters (viewDist, pix2deg, inputSizeDeg) are not set! Specify the viewing distance and either pix2deg or inputSizeDeg in the .ini file!')

        self.saveDir = iniReader['log_params'].get('saveDir', fallback=None)
        self.saveFix = iniReader['log_params'].getboolean('saveFix', fallback = False)
        #self.saveScreen = iniReader['log_params'].getboolean('saveScreen', fallback = False) #GUI not implemented

        self.overwrite = iniReader['log_params'].getboolean('overwrite', fallback = False)

        if self.saveFix and not self.saveDir:
            raise ValueError('saveFix set to true but save directory path (saveDir) is not provided. Set saveDir in the .ini file!')
